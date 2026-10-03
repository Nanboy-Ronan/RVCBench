"""Score audio generated outside RVCBench against a fixed, versioned suite.

``export_prompts`` writes the reference audio, transcripts and expected output file
names of a suite. Anyone can then generate speech with their own code, and
``score_submission`` scores the files with the suite's fixed evaluation protocol.
Target recordings are never exported; they are only read while scoring.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from .artifacts import (SCHEMA_VERSION, METRIC_COLUMNS, atomic_json, append_sample_event, coverage, digest, file_hash,
                        input_fingerprint, input_records, metric_means, provenance, runtime_root)

PROTOCOL = 'rvcbench-submission-v1'
SUITES_DIR = Path(__file__).resolve().parents[1] / 'suites'
CONFIGS_DIR = Path(__file__).resolve().parents[1] / 'configs'
_SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')


def available_suites():
    return sorted(path.parent.name.replace('_', '-') for path in SUITES_DIR.glob('*/suite.json'))


def load_suite(name_or_path):
    """Load a packaged suite by name (``onboarding-v1``) or a ``suite.json`` path."""
    path = Path(str(name_or_path))
    if path.suffix != '.json':
        path = SUITES_DIR / str(name_or_path).replace('-', '_') / 'suite.json'
        if not path.is_file():
            raise ValueError(f"Unknown suite '{name_or_path}'. Available: {', '.join(available_suites())}")
    raw = path.read_bytes()
    spec = json.loads(raw)
    for key in ('suite', 'version', 'leaderboard', 'label', 'evaluation', 'tasks'):
        if key not in spec:
            raise ValueError(f'Suite {path} is missing {key!r}')
    tasks = [task['task'] for task in spec['tasks']]
    if not tasks or len(set(tasks)) != len(tasks) or not all(_SAFE_NAME.fullmatch(t) for t in tasks):
        raise ValueError(f'Suite {path} needs unique, file-name-safe task names')
    generated = {task['task'] for task in spec['tasks'] if not task.get('derived_from')}
    for task in spec['tasks']:
        if task.get('derived_from') and (task['derived_from'] not in generated or 'transform' not in task):
            raise ValueError(f"Task {task['task']}: derived_from must name a generated task and come with a transform")
        if task.get('anchor') and task['anchor'] not in tasks:
            raise ValueError(f"Task {task['task']}: anchor {task['anchor']!r} is not a task of this suite")
        unknown = set(task.get('required_metrics') or ()) - set(METRIC_COLUMNS)
        if unknown:
            raise ValueError(f"Task {task['task']}: unknown metrics {sorted(unknown)}")
    spec['_directory'] = path.parent
    spec['_sha256'] = file_hash(path)
    return spec


def suite_record(spec):
    return {'suite': spec['suite'], 'version': spec['version'], 'leaderboard': bool(spec['leaderboard']),
            'label': spec['label'], 'suite_sha256': spec['_sha256']}


def _frozen_selection(spec, task):
    selection = json.loads((spec['_directory'] / task['selection']).read_text())
    return selection, {sample['sample_id']: sample for sample in selection['samples']}


def _task_root_name(task):
    # "paths": "repo" means manifest paths are relative to the dataset repository root,
    # which lets a task pair files from different folders (e.g. protected references).
    return '' if task.get('paths') == 'repo' else task['hf_config_name']


def _fetch_from_hub(spec, task, manifest_rows, *, attempts=4, wait_seconds=30):
    import time
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
    prefix = _task_root_name(task)
    names = sorted({row[key] for row in manifest_rows for key in ('prompt_file_name', 'target_file_name')})
    for attempt in range(1, attempts + 1):
        try:
            local = snapshot_download(repo_id=spec['hf_dataset_id'], repo_type='dataset', revision=spec['hf_revision'],
                                      allow_patterns=[f'{prefix}/{name}' if prefix else name for name in names],
                                      max_workers=4)
            return Path(local) / prefix
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status != 429:
                raise
            if attempt == attempts:
                raise RuntimeError('The Hugging Face Hub is rate-limiting anonymous downloads. Log in with '
                                   '`huggingface-cli login` (or set HF_TOKEN) and rerun; files already '
                                   'downloaded are reused.') from exc
            time.sleep(wait_seconds * attempt)


def _task_samples(spec, task, data_root, logger):
    """Load one task's frozen samples and verify every input against the frozen hashes."""
    from rvcbench.datasets.zero_shot import ZeroShotDataset
    manifest = (spec['_directory'] / task['manifest']).resolve()
    rows = json.loads(manifest.read_text())
    pair_ids = [str(row.get('pair_id') or '') for row in rows]
    if len(set(pair_ids)) != len(pair_ids) or not all(_SAFE_NAME.fullmatch(p) for p in pair_ids):
        raise ValueError(f"Task {task['task']}: pair_id values must be unique and file-name safe")
    root = (Path(data_root) / _task_root_name(task)) if data_root else _fetch_from_hub(spec, task, rows)
    dataset_config = OmegaConf.load(CONFIGS_DIR / 'dataset' / f"{task['dataset_config']}.yaml")
    dataset_config.root_path = str(root)
    dataset_config.use_hf_dataset = False
    dataset_config.manifest_filename = str(manifest)
    dataset_config.speaker_id = None
    dataset = ZeroShotDataset(OmegaConf.create({}), dataset_config, logger)
    samples = dataset.get_zero_shot_samples()
    records = input_records(samples)
    selection, frozen = _frozen_selection(spec, task)
    mismatched = [r['pair_id'] for r in records
                  if r['sample_id'] not in frozen
                  or r['prompt_sha256'] != frozen[r['sample_id']]['prompt_sha256']
                  or r['target_sha256'] != frozen[r['sample_id']]['target_sha256']]
    if mismatched or input_fingerprint(records) != selection['input_fingerprint']:
        raise ValueError(f"Task {task['task']}: inputs under {root} differ from the frozen suite "
                         f"(mismatched pairs: {', '.join(mismatched) or 'transcripts or metadata'})")
    return dataset, samples, records


def export_prompts(suite, output, *, data_root=None, logger=None):
    """Write reference audio, transcripts and expected output names; never target audio."""
    logger = logger or logging.getLogger('rvcbench.prompts')
    spec = load_suite(suite)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; choose a new directory')
    entries = []
    for task in spec['tasks']:
        if task.get('derived_from'):
            continue  # computed from another task's submitted audio
        _, samples, records = _task_samples(spec, task, data_root, logger)
        for sample, record in zip(samples, records):
            # Hub files are symlinks into a content-addressed cache; take the suffix from the manifest.
            suffix = Path(str(sample.extra.get('prompt_file_name') or '')).suffix or '.wav'
            reference = Path('references') / task['task'] / f"{record['pair_id']}{suffix}"
            (output / reference).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sample.prompt_path, output / reference)
            entries.append({'task': task['task'], 'pair_id': record['pair_id'], 'speaker_id': record['speaker_id'],
                            'reference_audio': reference.as_posix(), 'reference_sha256': record['prompt_sha256'],
                            'reference_text': record['prompt_text'], 'text': record['target_text'],
                            'language': record['target_language'],
                            'output_file': f"{task['task']}/{record['pair_id']}.wav"})
    with (output / 'prompts.jsonl').open('w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    atomic_json(output / 'suite.json', {**suite_record(spec), 'tasks': [t['task'] for t in spec['tasks'] if not t.get('derived_from')],
                                        'prompts': len(entries)})
    (output / 'README.md').write_text(_PROMPTS_README.format(
        suite=spec['suite'], label=spec['label'], count=len(entries)), encoding='utf-8')
    return {'suite': spec['suite'], 'prompts': len(entries), 'output': str(output.resolve())}


_PROMPTS_README = """# {suite} prompts

{label}

`prompts.jsonl` lists {count} utterances. For each line:

1. Synthesize `text` in the voice of `reference_audio` (its transcript is `reference_text`).
2. Save the result as a mono WAV file at `<your output directory>/<output_file>`.

Then score the directory:

    rvcbench score --suite {suite} --generated <your output directory> --model <model name> --output <results directory>

A missing or invalid file counts as a failed sample. Do not use the target recordings of
the dataset; a file identical to a target recording that was not exported is rejected.
"""


def _metrics_for(spec, task):
    return list(task.get('required_metrics') or spec['evaluation']['required_metrics'])


def _submitted_rows(spec, task, generated, audio_dir, data_root, logger):
    from .runner import check_audio
    dataset, samples, rows = _task_samples(spec, task, data_root, logger)
    for sample, row in zip(samples, rows):
        submitted = Path(generated) / task['task'] / f"{row['pair_id']}.wav"
        row.update(metrics={}, attempts=0, submitted_path=str(submitted))
        for column in _group_columns(task):
            row.setdefault('groups', {})[column] = str(sample.extra.get(column) or '')
        if not submitted.is_file():
            row.update(status='generation_failed', error=f"Missing submitted audio: {task['task']}/{row['pair_id']}.wav")
            continue
        try:
            check_audio(submitted)
        except Exception as exc:
            row.update(status='generation_failed', error=f'Invalid submitted audio: {type(exc).__name__}: {exc}')
            continue
        sha = file_hash(submitted)
        # A target recording is never exported, so submitting it is a leak. Some pairs
        # use the reference recording as their target; that file is public anyway.
        if sha == row['target_sha256'] and row['target_sha256'] != row['prompt_sha256']:
            row.update(status='generation_failed', error='Submitted audio is the target recording')
            continue
        stored = audio_dir / submitted.name
        shutil.copyfile(submitted, stored)
        row.update(status='generated', error=None, generated_path=str(stored.resolve()), generated_sha256=sha)
    return rows, {'dataset': OmegaConf.to_container(dataset.dataset_config),
                  'variant_selection': dataset.variant_selection}


_CODECS = {'mp3': ('libmp3lame', '.mp3'), 'aac': ('aac', '.m4a'), 'opus': ('libopus', '.ogg')}


def _ffmpeg():
    path = shutil.which('ffmpeg')
    if not path:
        raise RuntimeError('Post-processing tasks require ffmpeg on PATH')
    return path


def transform_audio(source, destination, transform, workdir):
    """Apply one post-processing condition and decode back to 16-bit mono WAV."""
    import subprocess
    ffmpeg = _ffmpeg()
    rate = str(int(transform.get('sample_rate', 24000)))
    quiet = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y']
    if transform['kind'] == 'codec':
        encoder, suffix = _CODECS[transform['codec']]
        intermediate = Path(workdir) / (Path(destination).stem + suffix)
        encode = [*quiet, '-i', str(source), '-ac', '1', '-ar', rate, '-c:a', encoder,
                  '-b:a', str(transform['bitrate']), str(intermediate)]
    elif transform['kind'] == 'narrowband':
        # Telephone channel: 300-3400 Hz band at 8 kHz, resampled back to the evaluation rate.
        intermediate = Path(workdir) / (Path(destination).stem + '_8k.wav')
        encode = [*quiet, '-i', str(source), '-ac', '1', '-af', 'highpass=f=300,lowpass=f=3400',
                  '-ar', '8000', '-c:a', 'pcm_s16le', str(intermediate)]
    else:
        raise ValueError(f"Unknown transform kind: {transform['kind']!r}")
    decode = [*quiet, '-i', str(intermediate), '-ac', '1', '-ar', rate, '-c:a', 'pcm_s16le', str(destination)]
    for command in (encode, decode):
        subprocess.run(command, check=True, capture_output=True)
    intermediate.unlink(missing_ok=True)


def _derived_rows(task, source_rows, audio_dir):
    """Rows comparing each processed clone with the unprocessed clone it came from."""
    rows = []
    work = audio_dir.parent / 'transform_work'
    work.mkdir()
    for source in source_rows:
        row = {key: value for key, value in source.items()
               if key not in ('metrics', 'metric_errors', 'generated_path', 'generated_sha256', 'error', 'status')}
        row.update(metrics={}, attempts=0, derived_from=task['derived_from'], transform=task['transform'])
        if not source.get('generated_sha256'):
            row.update(status='generation_failed', error=f"Source sample failed in {task['derived_from']}")
        else:
            destination = audio_dir / Path(source['generated_path']).name
            transform_audio(source['generated_path'], destination, task['transform'], work)
            row.update(target_path=source['generated_path'], target_sha256=source['generated_sha256'],
                       generated_path=str(destination.resolve()), generated_sha256=file_hash(destination),
                       status='generated', error=None)
        rows.append(row)
    work.rmdir()
    return rows


def _group_columns(task):
    columns = task.get('group_by') or []
    return [columns] if isinstance(columns, str) else list(columns)


def _group_means(rows, required, column):
    groups = {}
    for row in rows:
        groups.setdefault(row.get('groups', {}).get(column, ''), []).append(row)
    return {group: metric_means(members, required) for group, members in sorted(groups.items())}


def _run_task(spec, task, rows, details, task_dir, label, device, logger):
    evaluation = spec['evaluation']
    events = task_dir / 'sample_events.jsonl'
    for row in rows:
        append_sample_event(events, row)
    required = _metrics_for(spec, task)
    generation_config = {'model': label, 'adversary': {}, 'sample_seed_policy': 'external_submission_v1'}
    manifest = {'schema_version': SCHEMA_VERSION, 'protocol': PROTOCOL, 'status': 'running', 'evaluated': False,
                'suite': {**suite_record(spec), 'task': task['task']},
                'config': {'vc': {'mode': 'ots', 'model': label}, 'dataset': details.get('dataset'),
                           'evaluation': {**evaluation, 'required_metrics': required},
                           'task': {k: v for k, v in task.items() if not k.startswith('_')}},
                'provenance': provenance(runtime_root()), 'generation_config': generation_config,
                'generation_provenance': None, 'generation_fingerprint': digest(generation_config),
                'input_fingerprint': input_fingerprint(rows), 'samples': rows,
                'variant_selection': details.get('variant_selection'), 'reference_stage': None,
                'comparison_status': 'requires_pairwise_protocol_check',
                'coverage': coverage(rows, required, evaluated=False)}
    manifest_path = task_dir / 'run_manifest.json'
    atomic_json(manifest_path, manifest)
    metrics = {}
    try:
        if any(row.get('generated_sha256') for row in rows):
            from rvcbench.evaluation.pipeline import evaluate_run
            manifest['evaluation_device'] = str(device)
            metrics = evaluate_run(rows, required, task_dir, device, logger, seed=int(evaluation.get('seed', 42)),
                                   cap=evaluation.get('generated_audio_max_seconds'),
                                   bootstrap_config=evaluation.get('bootstrap'),
                                   wer_normalization=evaluation.get('wer_normalization', 'ascii_punctuation_removed_v2'),
                                   on_sample=lambda row: append_sample_event(events, row))
            manifest['evaluated'] = True
            manifest['evaluation_fingerprint'] = metrics['evaluation_fingerprint']
        manifest['coverage'] = coverage(rows, required, evaluated=manifest['evaluated'])
        manifest['status'] = manifest['coverage']['status']
        manifest['metrics'] = metrics
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        atomic_json(manifest_path, manifest)
    result = {key: task[key] for key in ('dimension', 'evaluation', 'anchor', 'derived_from') if key in task}
    result.update({'status': manifest['status'], 'required_metrics': required, 'coverage': manifest['coverage'],
                   'run_manifest': f"{task['task']}/run_manifest.json", 'run_manifest_sha256': file_hash(manifest_path),
                   'evaluation_fingerprint': manifest.get('evaluation_fingerprint'),
                   'failures': {row['pair_id']: row['error'] for row in rows if row['status'] == 'generation_failed'},
                   'generated_sha256': {row['pair_id']: row.get('generated_sha256') for row in rows}})
    if manifest['status'] == 'complete':
        result['means'] = metric_means(rows, required)
        if _group_columns(task):
            result['group_means'] = {column: _group_means(rows, required, column) for column in _group_columns(task)}
    return result, rows


def relative_change(task_means, anchor_means):
    """Percentage change of each shared metric relative to the anchor task, as in the paper."""
    return {metric: 100.0 * (value - anchor_means[metric]) / abs(anchor_means[metric])
            for metric, value in task_means.items()
            if metric in anchor_means and anchor_means[metric] not in (0, 0.0)}


def score_submission(suite, generated, output, *, model, device='cpu', data_root=None, logger=None):
    """Score externally generated audio; writes one run per task and ``submission.json``."""
    import rvcbench
    logger = logger or logging.getLogger('rvcbench.score')
    spec = load_suite(suite)
    if not str(model).strip():
        raise ValueError('A model name is required')
    if not Path(generated).is_dir():
        raise ValueError(f'Generated audio directory not found: {generated}')
    if any(task.get('derived_from') for task in spec['tasks']):
        _ffmpeg()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tasks, task_rows = {}, {}
    for task in sorted(spec['tasks'], key=lambda t: bool(t.get('derived_from'))):
        audio_dir = output / task['task'] / 'generated_audio'
        audio_dir.mkdir(parents=True)
        if task.get('derived_from'):
            rows, details = _derived_rows(task, task_rows[task['derived_from']], audio_dir), {}
        else:
            rows, details = _submitted_rows(spec, task, generated, audio_dir, data_root, logger)
        tasks[task['task']], task_rows[task['task']] = _run_task(
            spec, task, rows, details, output / task['task'], str(model), device, logger)
    for name, result in tasks.items():
        anchor = tasks.get(result.get('anchor'))
        if anchor and 'means' in result and 'means' in anchor:
            result['relative_change_percent'] = relative_change(result['means'], anchor['means'])
    ordered = {task['task']: tasks[task['task']] for task in spec['tasks']}
    complete = all(task['status'] == 'complete' for task in ordered.values())
    submission = {'schema_version': 1, 'protocol': PROTOCOL, **suite_record(spec), 'model': str(model),
                  'rvcbench_version': rvcbench.__version__, 'status': 'complete' if complete else 'partial',
                  'tasks': ordered}
    atomic_json(output / 'submission.json', submission)
    return submission
