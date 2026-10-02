#!/usr/bin/env python3
"""Rescore exact historical subset audio without claiming new generation provenance."""
import argparse
import copy
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rvcbench.benchmark.artifacts import atomic_json, file_hash, load_run
from rvcbench.benchmark.reproduction import match_historical, infer_english_wer_protocol
from rvcbench.evaluation.pipeline import evaluate_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=Path, help='Modern subset run defining the exact population')
parser.add_argument('historical_csv', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
logging.basicConfig(level=logging.INFO)
run = load_run(args.run_dir)
_, matches = match_historical(run, args.historical_csv)
rows = []
for new, old in matches:
    row = copy.deepcopy(new)
    row.update(generated_path=old['generated_path'], generated_sha256=file_hash(old['generated_path']),
               metrics={}, metric_errors={}, status='generated', error=None, synthesis_time_sec=None, timing_scope=None)
    rows.append(row)
wer_protocol = infer_english_wer_protocol(args.historical_csv)
metrics = evaluate_run(rows, ['mcd', 'wer', 'sim'], args.output, args.device, logging.getLogger('replay'),
                       wer_normalization=wer_protocol['normalization'])
differences = {}
for metric in ('mcd', 'wer', 'sim'):
    complete = all(metric in r['metrics'] and metric not in r.get('metric_errors', {}) for r in rows)
    delta = [float(r['metrics'][metric]) - float(old[metric]) for r, (_, old) in zip(rows, matches)] if complete else []
    differences[metric] = {'valid': len(delta), 'requested': len(rows),
        'mean_delta': sum(delta) / len(delta) if delta else None,
        'max_absolute_delta': max(map(abs, delta)) if delta else None}
atomic_json(args.output / 'replay_report.json', {
    'kind': 'historical_audio_metric_replay', 'historical_csv': str(args.historical_csv.resolve()),
    'historical_csv_sha256': file_hash(args.historical_csv), 'differences': differences,
    'historical_wer_protocol': wer_protocol,
    'metrics': metrics, 'samples': rows,
    'interpretation': 'Tests scorer reproduction on historical audio; does not prove generation reproducibility.'})
print(json.dumps(differences, indent=2))
