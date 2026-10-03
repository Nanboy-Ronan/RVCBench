"""Compare several models' submissions for the same suite in one table.

``compare_submissions`` reads ``submission.json`` files written by ``rvcbench score`` and
writes ``comparison.json``, a long-format ``comparison.csv`` and a ``comparison.md`` table
with one column per model. Submissions of different suites, or of different versions of
a suite, are refused.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .artifacts import atomic_json, file_hash

HIGHER_IS_BETTER = {'sim': True, 'sva': True, 'speechmos': True, 'dnsmos': True, 'emotion': True, 'stoi': True,
                    'wer': False, 'mcd': False}
LABELS = {'sim': 'SIM', 'sva': 'SVA', 'speechmos': 'MOS', 'dnsmos': 'DNSMOS', 'emotion': 'EMC', 'stoi': 'STOI',
          'wer': 'WER', 'mcd': 'MCD'}
DECIMALS = {'speechmos': 2, 'dnsmos': 2, 'mcd': 2}
CSV_FIELDS = ['task', 'evaluation', 'metric', 'model', 'mean', 'relative_change_percent', 'anchor', 'task_status']


def load_submission(path):
    """Read a ``submission.json`` file, or the one inside a results directory."""
    path = Path(path)
    if path.is_dir():
        path = path / 'submission.json'
    if not path.is_file():
        raise ValueError(f'No submission.json at {path}')
    submission = json.loads(path.read_text())
    if not str(submission.get('protocol', '')).startswith('rvcbench-submission-'):
        raise ValueError(f'{path} is not an RVCBench submission')
    return path, submission


def _metric_label(metric):
    arrow = '↑' if HIGHER_IS_BETTER.get(metric, True) else '↓'
    return f'{LABELS.get(metric, metric)} {arrow}'


def _format(metric, value, change):
    if value is None:
        return '—'
    text = f'{value:.{DECIMALS.get(metric, 3)}f}'
    return text if change is None else f'{text} ({change:+.1f}%)'


def _markdown(comparison, models, rows):
    lines = [f"# {comparison['suite']}: model comparison", '', f"> {comparison['label']}", '']
    statuses = ', '.join(f"{m['model']} ({m['status']})" for m in comparison['models'])
    lines += [f'Models: {statuses}. Values are task means; percentages are the change against the',
              'task\'s clean counterpart. **Bold** marks the best value when two or more models have one;',
              '— marks a task that is not complete for that model.', '']
    lines += ['| Task | Metric | ' + ' | '.join(models) + ' |', '| --- | --- | ' + ' | '.join('---:' for _ in models) + ' |']
    by_key = {}
    for row in rows:
        by_key.setdefault((row['task'], row['metric']), {})[row['model']] = row
    for (task, metric), cells in by_key.items():
        values = {m: c['mean'] for m, c in cells.items() if c['mean'] is not None}
        best = None
        if len(values) >= 2:
            best = (max if HIGHER_IS_BETTER.get(metric, True) else min)(values.values())
        rendered = []
        for model in models:
            cell = cells.get(model, {})
            text = _format(metric, cell.get('mean'), cell.get('relative_change_percent'))
            rendered.append(f'**{text}**' if best is not None and cell.get('mean') == best else text)
        lines.append(f'| {task} | {_metric_label(metric)} | ' + ' | '.join(rendered) + ' |')
    return '\n'.join(lines) + '\n'


def compare_submissions(paths, output=None):
    """Build the comparison of several submissions; write it under ``output`` when given."""
    loaded = [load_submission(p) for p in paths]
    if not loaded:
        raise ValueError('No submissions to compare')
    first = loaded[0][1]
    for path, submission in loaded[1:]:
        for key in ('suite', 'version', 'suite_sha256'):
            if submission.get(key) != first.get(key):
                raise ValueError(f"{path} is for {submission.get('suite')} (version {submission.get('version')}, "
                                 f"{key} differs from {first.get('suite')} version {first.get('version')}); "
                                 'only submissions of the same suite can be compared')
    models = [submission['model'] for _, submission in loaded]
    if len(set(models)) != len(models):
        raise ValueError(f'Duplicate model names: {models}')
    rows = []
    for name, reference in first['tasks'].items():
        for metric in reference['required_metrics']:
            for _, submission in loaded:
                task = submission['tasks'].get(name, {})
                rows.append({'task': name, 'evaluation': reference.get('evaluation'), 'metric': metric,
                             'model': submission['model'],
                             'mean': (task.get('means') or {}).get(metric),
                             'relative_change_percent': (task.get('relative_change_percent') or {}).get(metric),
                             'anchor': reference.get('anchor'), 'task_status': task.get('status', 'missing')})
    comparison = {'schema_version': 1, 'suite': first['suite'], 'version': first['version'],
                  'suite_sha256': first['suite_sha256'], 'leaderboard': first['leaderboard'], 'label': first['label'],
                  'models': [{'model': s['model'], 'status': s['status'], 'submission': str(p),
                              'submission_sha256': file_hash(p), 'rvcbench_version': s.get('rvcbench_version')}
                             for p, s in loaded],
                  'rows': rows}
    markdown = _markdown(comparison, models, rows)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'comparison.json', comparison)
        with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (output / 'comparison.md').write_text(markdown, encoding='utf-8')
    return {**comparison, 'markdown': markdown}
