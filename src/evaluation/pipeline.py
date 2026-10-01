"""Durable sample-by-metric evaluation with bounded model residency."""
from __future__ import annotations

import csv
import gc
import importlib.metadata
import inspect
import json
import math
from pathlib import Path

from src.benchmark.artifacts import METRIC_COLUMNS, atomic_json, digest, file_hash, finite, metric_value_valid as _valid
from .scorers import ScoreInput, create_scorer
from src.utils.runtime_errors import invalid_cuda_context


def _scorer_provenance(scorer, seed, cap, device=None):
    from .scorers import text
    implementation = Path(inspect.getfile(type(scorer)))
    paths = [implementation, Path(__file__), Path(text.__file__),
             Path(__file__).parents[1] / 'utils/runtime_errors.py']
    if implementation.stem == 'auxiliary':
        paths += [Path(__file__).with_name('generation.py'), Path(__file__).with_name('fidelity.py')]
    packages = {}
    for package in scorer.dependencies:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    execution = None
    if device is not None:
        import torch
        resolved = torch.device(device)
        execution = {'device': str(resolved), 'torch': torch.__version__,
                     'cuda': torch.version.cuda, 'cpu_threads': torch.get_num_threads()}
        if resolved.type == 'cuda':
            properties = torch.cuda.get_device_properties(resolved)
            execution.update(gpu=properties.name,
                             compute_capability=[properties.major, properties.minor],
                             cudnn=torch.backends.cudnn.version())
    return {'version': scorer.version, 'source': {p.name: file_hash(p) for p in paths},
            'packages': packages, 'model': scorer.model_provenance, 'seed': seed,
            'sample_seed_policy': 'metric_seed_plus_sample_hash_v1', 'generated_audio_max_seconds': cap,
            'execution': execution}


def _request(row, output, cap):
    generated = Path(row['generated_path'])
    if cap is not None:
        import soundfile as sf
        info = sf.info(str(generated))
        frames = int(cap * info.samplerate)
        if info.frames > frames:
            trimmed = output / '_eval_trimmed' / (row['generated_sha256'] + f'_{cap}.wav')
            trimmed.parent.mkdir(parents=True, exist_ok=True)
            # Recreate from the verified source; do not trust an old partial trim.
            audio, rate = sf.read(str(generated), frames=frames)
            sf.write(str(trimmed), audio, rate, subtype='PCM_16')
            generated = trimmed
    return ScoreInput(Path(row['target_path']), generated, row['target_text'],
                      row.get('target_language') or row.get('prompt_language'))


def evaluate_run(rows, required, output, device, logger, *, seed=42, cap=None,
                 bootstrap_config=None, cache_sources=(), on_sample=None,
                 wer_normalization='ascii_punctuation_removed_v2'):
    """Mutate only score fields on rows; journal each finished sample/metric.

    Successful cached values are reused only after the active scorer's code,
    dependencies, weights, configuration, and exact input hashes match.
    """
    import torch
    from src.utils.seeding import configure_seeds
    if not required or set(required) - METRIC_COLUMNS.keys():
        raise ValueError('Unknown or empty metric selection')
    if cap is not None and (not math.isfinite(float(cap)) or float(cap) <= 0):
        raise ValueError('generated_audio_max_seconds must be finite and positive')
    cap = float(cap) if cap is not None else None
    output = Path(output)
    cache = output / 'metric_cache'
    cache.mkdir(parents=True, exist_ok=True)
    sources = [cache] + [Path(p) / 'metric_cache' for p in cache_sources]
    active = [r for r in rows if r.get('generated_sha256')]
    for row in active:
        row.setdefault('metrics', {})
        row.setdefault('metric_errors', {})
        # Verify source content once, before accepting any cached score.
        for kind in ('target', 'generated'):
            if file_hash(row[kind + '_path']) != row[kind + '_sha256']:
                raise ValueError(f'{kind} artifact changed before evaluation: {row["sample_id"]}')
    requests = {r['sample_id']: _request(r, output, cap) for r in active}
    groups = {}
    for metric in required:
        groups.setdefault('sim' if metric in ('sim', 'sva') else metric, []).append(metric)
    provenance = {}
    for group, metrics in groups.items():
        scorer = None
        fatal_error = False
        try:
            configure_seeds(seed, logger=None)
            scorer = create_scorer(group, device, logger)
            if group == 'wer' and wer_normalization != 'ascii_punctuation_removed_v2':
                scorer.set_normalization(wer_normalization)
            scorer.prepare()
            spec = _scorer_provenance(scorer, seed, cap, device)
            fingerprint = digest(spec)
            provenance[group] = {'fingerprint': fingerprint, **spec}
            atomic_json(output / 'scoring_manifest.json', provenance)
            for index, row in enumerate(active):
                key = digest({'scorer': fingerprint, 'target': row['target_sha256'],
                              'generated': row['generated_sha256'], 'text': row['target_text'],
                              'language': requests[row['sample_id']].language,
                              'sample_id': row['sample_id']})
                record = None
                for directory in sources:
                    path = directory / (key + '.json')
                    if path.is_file():
                        candidate = json.loads(path.read_text())
                        if (candidate.get('key') == key and candidate.get('status') == 'complete'
                                and all(_valid(m, candidate.get('values', {}).get(METRIC_COLUMNS[m])) for m in metrics)):
                            record = candidate
                            break
                try:
                    if record is None:
                        configure_seeds((seed + int(row['sample_id'][:8], 16)) % (2**32 - 1), logger=None)
                        values = scorer.score(requests[row['sample_id']])
                        missing = [m for m in metrics if not _valid(m, values.get(METRIC_COLUMNS[m]))]
                        if missing:
                            raise ValueError(f'Missing or invalid metric values: {missing}')
                        record = {'key': key, 'status': 'complete', 'values': values,
                                  'scorer_fingerprint': fingerprint, 'sample_id': row['sample_id']}
                    row['metrics'].update(record['values'])
                    for metric in metrics:
                        row['metric_errors'].pop(metric, None)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    record = {'key': key, 'status': 'failed', 'error': error, 'sample_id': row['sample_id']}
                    for metric in metrics:
                        row['metrics'].pop(METRIC_COLUMNS[metric], None)
                        row['metric_errors'][metric] = error
                    logger.warning('Scoring %s failed for %s: %s', group, row['sample_id'], error)
                    if invalid_cuda_context(exc):
                        fatal_error = True
                        row.update(status='metric_failed', error=error, fatal_runtime_error=True)
                        atomic_json(cache / (key + '.json'), record)
                        if on_sample:
                            on_sample(row)
                        raise
                atomic_json(cache / (key + '.json'), record)
                row['status'] = 'scoring'
                if on_sample:
                    on_sample(row)
                logger.info('[%s %d/%d] %s', group, index + 1, len(active), record['status'])
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            provenance[group] = {**provenance.get(group, {}), 'status': 'failed', 'error': error}
            logger.error('Scorer %s unavailable: %s', group, error)
            if fatal_error or invalid_cuda_context(exc):
                fatal_error = True
                provenance[group]['fatal_runtime_error'] = True
                atomic_json(output / 'scoring_manifest.json', provenance)
                raise
            for row in active:
                for metric in metrics:
                    row['metric_errors'][metric] = error
                if on_sample:
                    on_sample(row)
        finally:
            if scorer is not None:
                try:
                    scorer.close()
                except Exception as exc:
                    if not fatal_error:
                        raise
                    logger.error('Scorer cleanup after fatal runtime error failed: %s', exc)
                del scorer
            gc.collect()
            if not fatal_error and torch.cuda.is_available():
                torch.cuda.empty_cache()
        atomic_json(output / 'scoring_manifest.json', provenance)
    for row in active:
        missing = [m for m in required if not _valid(m, row['metrics'].get(METRIC_COLUMNS[m])) or m in row['metric_errors']]
        row.update(status='metric_failed' if missing else 'complete',
                   error=f'Missing/invalid metrics: {missing}' if missing else None)
        if on_sample:
            on_sample(row)
    # Cached and fresh scoring must produce the same resampling distribution.
    configure_seeds(seed, logger=None)
    return _export(rows, required, output, provenance, bootstrap_config)


def _export(rows, required, output, provenance, bootstrap_config):
    import soundfile as sf
    from omegaconf import OmegaConf
    from .bootstrap import create_bootstrapper
    bootstrapper = create_bootstrapper(bootstrap_config)
    csv_path = output / 'generation_sample_metrics.csv'
    fields = ['sample_id', 'speaker_id', 'ground_truth_path', 'generated_path', 'ground_truth_text',
              'predicted_text', 'reference_emotion', 'generated_emotion'] + list(METRIC_COLUMNS.values()) + ['dnsmos_sig', 'dnsmos_bak',
              'generated_duration_sec', 'synthesis_time_sec', 'timing_scope']
    timings = {}
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            duration = None
            if row.get('generated_sha256'):
                info = sf.info(row['generated_path'])
                duration = info.frames / info.samplerate
                elapsed = row.get('synthesis_time_sec')
                if elapsed is not None and finite(elapsed) and float(elapsed) >= 0:
                    scope = row.get('timing_scope') or 'unspecified'
                    group = timings.setdefault(scope, {'synthesis_time_sec': 0., 'generated_duration_sec': 0., 'samples': 0})
                    group['synthesis_time_sec'] += float(elapsed)
                    group['generated_duration_sec'] += duration
                    group['samples'] += 1
            writer.writerow({'sample_id': row['sample_id'], 'speaker_id': row['speaker_id'],
                'ground_truth_path': row['target_path'], 'generated_path': row.get('generated_path'),
                'ground_truth_text': row['target_text'], 'generated_duration_sec': duration,
                'synthesis_time_sec': row.get('synthesis_time_sec'), 'timing_scope': row.get('timing_scope'),
                **row.get('metrics', {})})
    aggregation = OmegaConf.to_container(bootstrap_config, resolve=True) if OmegaConf.is_config(bootstrap_config) else bootstrap_config
    result = {'sample_metrics_csv': str(csv_path), 'scorers': provenance,
              'evaluation_fingerprint': digest({'scorers': provenance, 'bootstrap': aggregation,
                                                'required_metrics': required}),
              'evaluated_pairs': sum(bool(r.get('generated_sha256')) for r in rows)}
    for group in timings.values():
        group['rtf'] = group['synthesis_time_sec'] / group['generated_duration_sec'] if group['generated_duration_sec'] else None
    result['rtf_by_scope'] = timings
    only = next(iter(timings.values())) if len(timings) == 1 else None
    result['rtf'] = only['rtf'] if only and only['samples'] == len(rows) and 'unspecified' not in timings else None
    for metric in required:
        values = [float(r['metrics'][METRIC_COLUMNS[metric]]) for r in rows
                  if _valid(metric, r.get('metrics', {}).get(METRIC_COLUMNS[metric])) and metric not in r.get('metric_errors', {})]
        result[f'{metric}_pairs'] = len(values)
        result[f'avg_{metric}'] = sum(values) / len(values) if values else None
        if bootstrapper and values:
            bootstrapper.maybe_add_interval(result, f'avg_{metric}', values)
    return result
