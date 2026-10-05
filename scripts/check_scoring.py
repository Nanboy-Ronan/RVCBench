#!/usr/bin/env python3
"""Real speech regression for the installed package; no cloning model is required.

The fixture is an explicitly transformed public LibriTTS recording, not a model
submission or benchmark result. --record is for maintainer review of new golden
values; CI only reads the checked-in baseline.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from rvcbench import metrics
from rvcbench.benchmark.artifacts import atomic_json, file_hash, input_fingerprint
from rvcbench.benchmark.submission import load_suite, read_json, score_submission
from rvcbench.evaluation.locked_assets import fetch_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record', action='store_true', help='Write a candidate golden file under --output for review')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)
    packaged = load_suite('onboarding-v1')
    task = packaged['tasks'][0]
    row = read_json(packaged['_directory'] / task['manifest'])[0]
    frozen = read_json(packaged['_directory'] / task['selection'])['samples'][0]
    data = args.output / 'data'
    for kind in ('prompt', 'target'):
        relative = f"{task['hf_config_name']}/{row[kind + '_file_name']}"
        url = f"https://huggingface.co/datasets/{packaged['hf_dataset_id']}/resolve/{packaged['hf_revision']}/{relative}"
        fetch_url(url, data / relative, frozen[kind + '_sha256'])
    target = data / task['hf_config_name'] / row['target_file_name']
    audio, rate = sf.read(target, dtype='float32')
    generated = args.output / 'generated'
    generated.mkdir(exist_ok=True)
    wav = generated / f"fixture__{row['pair_id']}.wav"
    # Deterministic, non-identical speech for checking real scorer outputs.
    transformed = .9 * audio.astype('float64') + .005 * np.sin(np.arange(len(audio)) * .13)
    sf.write(wav, transformed, rate, subtype='PCM_16')
    selected = metrics.available()
    suite_dir = args.output / 'suite'
    suite_dir.mkdir(exist_ok=True)
    atomic_json(suite_dir / 'metadata.json', [row])
    atomic_json(suite_dir / 'selection.json', {'input_fingerprint': input_fingerprint([frozen]), 'samples': [frozen]})
    spec = {'suite': 'scorer-regression-v1', 'version': 1, 'leaderboard': False,
            'label': 'Transformed real speech scorer regression; not a model benchmark.',
            'evaluation': {'required_metrics': selected, 'seed': 42, 'bootstrap': {'enabled': False}},
            'tasks': [{'task': 'fixture', 'dataset_config': 'libritts', 'hf_config_name': task['hf_config_name'],
                       'manifest': 'metadata.json', 'selection': 'selection.json'}]}
    atomic_json(suite_dir / 'suite.json', spec)
    sample_seed = (42 + int(frozen['sample_id'][:8], 16)) % (2**32 - 1)
    with metrics.Evaluator('all', device='cpu', seed=sample_seed) as evaluator:
        scores = evaluator.score(wav, reference=target, target=target, text=row['target_text'], language='en')
    output = args.output / 'scored'
    result = score_submission(suite_dir / 'suite.json', generated, output, model='regression-fixture',
                              data_root=data, device='cpu', resume=output.exists())
    if result['status'] != 'complete':
        raise AssertionError(f'Suite scoring failed: {result}')
    means = result['tasks']['fixture']['means']
    for name in selected:
        if abs(float(scores[name]) - means[name]) > 1e-5:
            raise AssertionError(f'API and suite disagree on {name}: {scores[name]} != {means[name]}')
    resumed = score_submission(suite_dir / 'suite.json', generated, output, model='regression-fixture',
                               data_root=data, device='cpu', resume=True)
    assert resumed['tasks']['fixture']['means'] == means
    actual = {'fixture': 'onboarding-v1 LibriTTS-121-000039 target, gain 0.9 plus sine noise',
              'generated_sha256': file_hash(wav), 'scores': scores}
    atomic_json(args.output / 'actual.json', actual)
    golden_path = Path(__file__).with_name('scoring_golden.json')
    if args.record:
        actual['absolute_tolerance'] = {'sim': .002, 'sva': 0, 'wer': .001, 'speechmos': .02,
                                        'mcd': .02, 'stoi': .002, 'emotion': 0}
        atomic_json(args.output / 'scoring_golden.json', actual)
    else:
        golden = read_json(golden_path)
        assert actual['generated_sha256'] == golden['generated_sha256'], 'Regression audio changed'
        for name, expected in golden['scores'].items():
            if abs(float(scores[name]) - float(expected)) > golden['absolute_tolerance'][name]:
                raise AssertionError(f'{name}: {scores[name]} differs from golden {expected}')
    print(json.dumps(actual, indent=2))


if __name__ == '__main__':
    main()
