#!/usr/bin/env python3
"""Identify English WER normalization from saved predictions, without rerunning ASR."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.artifacts import atomic_json, file_hash
from src.benchmark.reproduction import infer_english_wer_protocol

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('manifest_audit', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = json.loads(args.manifest_audit.read_text())
results = []
for entry in source['historical_results']:
    if entry['rows'] < 1999:
        continue
    path = Path(entry['csv'])
    if file_hash(path) != entry['sha256']:
        raise ValueError(f'Historical CSV changed: {path}')
    try:
        protocol = infer_english_wer_protocol(path)
        result = {'status': 'verified_formula', **protocol}
    except ValueError as exc:
        result = {'status': 'unresolved', 'reason': str(exc)}
    results.append({'csv': str(path), 'sha256': entry['sha256'], **result})
atomic_json(args.output, {'kind': 'historical_wer_formula_audit', 'results': results,
    'scope': 'English LibriTTS; formula verification from stored predictions only.',
    'limitations': 'Does not establish ASR weight, decode, runtime or generation equivalence.'})
print(args.output)
