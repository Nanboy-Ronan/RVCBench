"""Portable sample identities, atomic artifacts and explicit coverage policy."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path

SCHEMA_VERSION = 2
METRIC_COLUMNS = {
    'mcd': 'mcd', 'wer': 'wer', 'sim': 'sim', 'sva': 'sva',
    'speechmos': 'speechmos_mos', 'dnsmos': 'dnsmos_ovrl', 'emotion': 'emotion_match',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sample_id(sample):
    extra = getattr(sample, 'extra', {}) or {}
    identity = {k: str(extra.get(k) or '') for k in ('dataset_name', 'split', 'pair_id')}
    identity['speaker_id'] = str(sample.speaker_id)
    if extra.get('manifest_variant'):
        identity['manifest_variant'] = str(extra['manifest_variant'])
    if not identity['pair_id']:
        # Legacy manifests: relative names plus transcripts, never machine-specific roots.
        identity.update(prompt=str(extra.get('prompt_file_name') or Path(str(sample.prompt_path)).name),
                        target=str(extra.get('target_file_name') or Path(str(sample.target_path)).name),
                        prompt_text=sample.prompt_text, target_text=sample.target_text)
    return digest(identity)


def output_path(root, sample, suffix='cloned'):
    speaker = re.sub(r'[^0-9A-Za-z_.-]', '_', str(sample.speaker_id)) or 'unknown'
    return Path(root) / speaker / f'{sample_id(sample)}_{suffix}.wav'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def provenance(root):
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata['Name']
        if name:
            packages.setdefault(name, distribution.version)
    import torch
    hardware = {'machine': platform.machine(), 'processor': platform.processor(),
                'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpus': []}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            hardware['gpus'].append({'visible_index': index, 'name': props.name,
                                    'total_memory': props.total_memory,
                                    'compute_capability': [props.major, props.minor]})
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
            'source_sha256': digest({str(p.relative_to(root)): file_hash(p) for p in sorted((Path(root) / 'src').rglob('*.py'))}),
            'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
            'hardware': hardware}


def input_records(samples):
    rows = []
    seen = set()
    cache = {}
    for sample in samples:
        sid = sample_id(sample)
        if sid in seen:
            raise ValueError(f'Duplicate sample identity: {sid}; fix pair_id in the input manifest')
        seen.add(sid)
        row = {'sample_id': sid, 'pair_id': str(sample.extra.get('pair_id') or ''),
               'speaker_id': str(sample.speaker_id), 'target_text': sample.target_text,
               'prompt_text': sample.prompt_text, 'language': sample.target_language,
               'prompt_language': sample.prompt_language, 'target_language': sample.target_language,
               'source_index': sample.index,
               'manifest_variant': sample.extra.get('manifest_variant', 'default'),
               'status': 'pending', 'error': None}
        for name in ('prompt', 'target'):
            path = getattr(sample, name + '_path')
            row[name + '_path'] = str(path) if path else None
            key = str(path)
            if key not in cache:
                cache[key] = file_hash(path) if path and Path(path).is_file() else None
            row[name + '_sha256'] = cache[key]
        rows.append(row)
    return rows


def input_fingerprint(rows):
    return digest([{k: r[k] for k in ('sample_id', 'prompt_sha256', 'target_sha256',
                                     'prompt_text', 'target_text', 'prompt_language', 'target_language', 'source_index')} for r in rows])


def finite(value):
    if value is None or value == '':
        return False
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return True
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def metric_value_valid(metric, value):
    if not finite(value):
        return False
    if metric in ('sva', 'emotion'):
        if isinstance(value, str) and value.lower() in ('true', 'false'):
            return True
        return float(value) in (0, 1)
    if isinstance(value, bool) or (isinstance(value, str) and value.lower() in ('true', 'false')):
        return False
    number = float(value)
    if metric in ('mcd', 'wer'):
        return number >= 0
    if metric == 'sim':
        return -1.000001 <= number <= 1.000001
    return True


def coverage(rows, required_metrics, *, evaluated):
    requested = len(rows)
    generated = sum(r.get('generated_sha256') is not None for r in rows)
    counts = {m: sum(metric_value_valid(m, r.get('metrics', {}).get(METRIC_COLUMNS[m])) and m not in r.get('metric_errors', {}) for r in rows)
              for m in METRIC_COLUMNS}
    complete = bool(requested) and generated == requested and evaluated and all(counts[m] == requested for m in required_metrics)
    return {'requested': requested, 'generated': generated,
            'generation_failed': sum(r.get('status') == 'generation_failed' for r in rows),
            'input_failed': sum(r.get('status') == 'input_failed' for r in rows),
            'pending': sum(r.get('status') == 'pending' for r in rows),
            'generating': sum(r.get('status') == 'generating' for r in rows),
            'metric_valid': counts, 'required_metrics': list(required_metrics),
            'status': 'complete' if complete else ('generated' if not evaluated and generated == requested and requested else 'partial'),
            'eligible_for_comparison': complete}


def validate_report(manifest, *, verify_files=True):
    if manifest.get('schema_version') not in (1, SCHEMA_VERSION):
        raise ValueError('Unsupported run schema')
    rows = manifest['samples']
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    if manifest['schema_version'] >= 2 and input_fingerprint(rows) != manifest.get('input_fingerprint'):
        raise ValueError('Input fingerprint does not match sample records')
    expected = coverage(rows, manifest['coverage']['required_metrics'], evaluated=manifest['evaluated'])
    if manifest['schema_version'] == 1:
        for name in ('input_failed', 'pending', 'generating'):
            expected.pop(name)
        expected['generation_failed'] = expected['requested'] - expected['generated']
    if expected != manifest['coverage']:
        raise ValueError('Coverage does not match sample records')
    if not expected['eligible_for_comparison'] or manifest['status'] != 'complete':
        raise ValueError('Run is incomplete or generation-only; not eligible for comparison')
    if verify_files:
        for row in rows:
            for kind in ('prompt', 'target', 'generated'):
                if file_hash(row[kind + '_path']) != row[kind + '_sha256']:
                    raise ValueError(kind + ' audio has changed: ' + row['sample_id'])
    return expected


def metric_means(rows, required_metrics):
    means = {}
    for metric in required_metrics:
        column = METRIC_COLUMNS[metric]
        values = [r.get('metrics', {}).get(column) for r in rows]
        if not all(finite(v) for v in values):
            raise ValueError('Incomplete metric: ' + metric)
        numbers = [float(v.lower() == 'true') if isinstance(v, str) and v.lower() in ('true', 'false') else float(v) for v in values]
        means[metric] = sum(numbers) / len(numbers)
    return means


def append_sample_event(path, row):
    """Durable O(1) progress updates; avoid rewriting the full run per sample."""
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def load_run(directory):
    """Recover completed sample events, ignoring only an interrupted final line."""
    directory = Path(directory)
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    journal = directory / 'sample_events.jsonl'
    if journal.is_file() and manifest['status'] in ('running', 'interrupted', 'failed'):
        rows = {r['sample_id']: r for r in manifest['samples']}
        with journal.open() as f:
            for line in f:
                if not line.endswith('\n'):
                    break
                event = json.loads(line)
                if event['sample_id'] not in rows:
                    raise ValueError('Journal contains an unknown sample ID')
                rows[event['sample_id']] = event
        manifest['samples'] = [rows[r['sample_id']] for r in manifest['samples']]
        manifest['coverage'] = coverage(manifest['samples'], manifest['coverage']['required_metrics'],
                                        evaluated=manifest['evaluated'])
    return manifest
