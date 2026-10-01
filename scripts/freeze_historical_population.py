#!/usr/bin/env python3
"""Freeze an entire historical LibriTTS speaker population, checking CSV identity.

This selects no rows based on scores. Run from the repository root in an
environment with the repository installed.
"""
import argparse
import csv
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from src.benchmark.artifacts import atomic_json, file_hash, input_fingerprint, input_records
from src.datasets.manifest_utils import canonicalize_records
from src.datasets.zero_shot import ZeroShotDataset


def freeze(source_root, audio_root, historical_csv, output, expected_pairs):
    source_root, audio_root = Path(source_root), Path(audio_root).resolve()
    historical_csv, output = Path(historical_csv), Path(output)
    if expected_pairs <= 0:
        raise ValueError('expected_pairs must be positive')
    if output.exists():
        raise ValueError('Use a new output directory; frozen populations are immutable')
    files = sorted((source_root / 'filelists').glob('*.json'))
    files = [p for p in files if (source_root / 'audios' / p.stem).is_dir()
             or (audio_root / 'audios' / p.stem).is_dir()]
    metadata = []
    for path in files:
        records = json.loads(path.read_text())
        frame = canonicalize_records(records, dataset_name='LibriTTS', source_manifest=path.name)
        for row in frame.to_dict(orient='records'):
            row.update(manifest_variant='speaker', source_index=len(metadata))
            metadata.append(row)
    with historical_csv.open(newline='') as handle:
        history = list(csv.DictReader(handle))
    if len(metadata) != expected_pairs or len(history) != expected_pairs:
        raise ValueError('Entire manifest population and historical CSV must match expected_pairs')
    # Resolve through the canonical audio root without modifying source files.
    # Validate before creating the frozen artifact directory.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='rvcbench-freeze-') as temporary:
        manifest = Path(temporary) / 'metadata.json'
        atomic_json(manifest, metadata)
        conf = OmegaConf.create({'dataset': {'root_path': str(audio_root),
            'use_hf_dataset': False, 'manifest_filename': str(manifest), 'manifest_variant': 'speaker'}})
        dataset = ZeroShotDataset(conf, conf.dataset, logging.getLogger(__name__))
        samples = dataset.get_zero_shot_samples()
        inputs = input_records(samples)
        for sample, old, record in zip(samples, history, inputs):
            expected_name = f'{sample.prompt_path.stem}_to_{sample.target_path.stem}_cloned.wav'
            if (str(old['speaker_id']) != sample.speaker_id
                    or Path(old['generated_path']).name != expected_name
                    or Path(old['ground_truth_path']).name != sample.target_path.name
                    or old['ground_truth_text'] != sample.target_text):
                raise ValueError(f'Historical pair/order/transcript mismatch at index {sample.index}')
            old_target = Path(old['ground_truth_path'])
            old_prompt = old_target.parent / sample.prompt_path.name
            if (not record['prompt_sha256'] or not record['target_sha256']
                    or file_hash(old_target) != record['target_sha256']
                    or file_hash(old_prompt) != record['prompt_sha256']):
                raise ValueError(f'Historical audio content mismatch at index {sample.index}')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'metadata.json', metadata)
    atomic_json(output / 'selection.json', {
        'schema_version': 1, 'selection': 'entire_historical_speaker_manifest_population_v1',
        'population': len(samples), 'requested': len(samples), 'speakers': len(files),
        'variant_selection': dataset.variant_selection,
        'source_manifest_sha256': {p.name: file_hash(p) for p in files},
        'historical_csv': str(historical_csv.resolve()), 'historical_csv_sha256': file_hash(historical_csv),
        'metadata_sha256': file_hash(output / 'metadata.json'),
        'input_fingerprint': input_fingerprint(inputs),
        'samples': [{k: v for k, v in r.items() if k not in ('prompt_path', 'target_path', 'status', 'error')}
                    for r in inputs],
        'interpretation': 'Entire legacy population; no outcome-based selection. Original cohort selection '
                          'and historical generation seed/weight revision may be unknown. '
                          'Does not establish full-paper reproduction.',
    })
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--historical-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-pairs', type=int, required=True)
    args = parser.parse_args()
    print(freeze(args.source_root, args.audio_root, args.historical_csv, args.output, args.expected_pairs))
