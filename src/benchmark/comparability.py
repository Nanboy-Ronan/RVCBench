"""Pairwise protocol checks; a complete run alone does not establish comparability."""
import json
from pathlib import Path

from .artifacts import METRIC_COLUMNS, digest, file_hash, load_run, validate_report


def check_comparability(left_dir, right_dir, metrics):
    metrics = list(metrics)
    if not metrics or set(metrics) - METRIC_COLUMNS.keys():
        raise ValueError('Comparison requires a nonempty supported metric selection')
    directories = [Path(left_dir), Path(right_dir)]
    runs = [load_run(p) for p in directories]
    reasons = []
    for index, run in enumerate(runs):
        validate_report(run)
        if run['config']['vc']['model'] == 'smoke':
            reasons.append(f'run {index}: synthetic smoke results')
        for metric in metrics:
            if run['coverage']['metric_valid'][metric] != len(run['samples']):
                reasons.append(f'run {index}: incomplete {metric}')
    left, right = runs
    for key in ('protocol', 'input_fingerprint'):
        if not left.get(key) or left.get(key) != right.get(key):
            reasons.append(f'{key} differs or is missing')
    policies = [(r.get('generation_config') or {}).get('sample_seed_policy') for r in runs]
    if policies[0] is None or policies[0] != policies[1]:
        reasons.append('sample seed policies differ or are missing')
    scorers = []
    for directory in directories:
        path = directory / 'scoring_manifest.json'
        scorers.append(json.loads(path.read_text()) if path.is_file() else {})
    for metric in metrics:
        group = 'sim' if metric in ('sim', 'sva') else metric
        records = [s.get(group, {}) for s in scorers]
        fingerprints = [s.get('fingerprint') for s in records]
        for index, record in enumerate(records):
            spec = {k: v for k, v in record.items() if k != 'fingerprint'}
            if record.get('fingerprint') != digest(spec):
                reasons.append(f'run {index}: {metric} scorer fingerprint is inconsistent')
        if fingerprints[0] is None or fingerprints[0] != fingerprints[1]:
            reasons.append(f'{metric} scorer code, dependencies, assets or settings differ or are missing')
    return {'status': 'comparable' if not reasons else 'incompatible', 'reasons': reasons,
            'metrics': metrics, 'requested_pairs': [len(r['samples']) for r in runs],
            'models': [r['config']['vc']['model'] for r in runs],
            'sources': [{'manifest': str((p / 'run_manifest.json').resolve()),
                         'sha256': file_hash(p / 'run_manifest.json')} for p in directories],
            'scope': 'Matched quality metrics across models; does not authorize timing, intervention or historical equivalence claims.'}
