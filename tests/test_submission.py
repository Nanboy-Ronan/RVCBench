import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pytest
import soundfile as sf

from rvcbench.benchmark import submission
from rvcbench.benchmark.artifacts import file_hash, input_fingerprint, input_records, validate_report
from rvcbench.datasets.zero_shot import ZeroShotDataset


ROOT = Path(__file__).resolve().parents[1]
PAIRS = [('s1', 'a', 'b', 'hello there'), ('s1', 'b', 'c', 'good morning'), ('s2', 'd', 'e', 'see you soon'),
         ('s2', 'd', 'd', 'a different sentence')]  # the reference doubles as the target, as in Robotcall


def fake_evaluate(rows, required, output, device, logger, **kwargs):
    for row in rows:
        if row.get('generated_sha256'):
            row['metrics'] = {'sim': 0.5}
            row['status'] = 'complete'
    return {'evaluation_fingerprint': 'test-only-fingerprint', 'avg_sim': 0.5}


@pytest.fixture
def suite(tmp_path):
    data = tmp_path / 'data' / 'Toy'
    for index, name in enumerate('abcde'):
        speaker = 's2' if name in 'de' else 's1'
        path = data / 'audios' / speaker / f'{name}.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, 0.1 * np.sin(np.arange(1600) * 0.05 * (index + 1)), 16000)
    rows = [dict(dataset_name='Toy', split='default', pair_id=f'Toy-{speaker}-{i:06d}', speaker_id=speaker,
                 manifest_variant='speaker', source_manifest=f'{speaker}.json', source_row=i,
                 prompt_file_name=f'audios/{speaker}/{prompt}.wav', target_file_name=f'audios/{speaker}/{target}.wav',
                 prompt_text='reference words', target_text=text, prompt_language='EN', target_language='EN',
                 source_index=i)
            for i, (speaker, prompt, target, text) in enumerate(PAIRS)]
    directory = tmp_path / 'suite'
    directory.mkdir()
    (directory / 'toy.metadata.json').write_text(json.dumps(rows))
    config = OmegaConf.load(submission.CONFIGS_DIR / 'dataset' / 'libritts.yaml')
    config.update({'root_path': str(data), 'use_hf_dataset': False,
                   'manifest_filename': str(directory / 'toy.metadata.json'), 'speaker_id': None})
    records = input_records(ZeroShotDataset(OmegaConf.create({}), config, logging.getLogger('t')).get_zero_shot_samples())
    (directory / 'toy.selection.json').write_text(json.dumps(
        {'input_fingerprint': input_fingerprint(records), 'samples': records}))
    (directory / 'suite.json').write_text(json.dumps({
        'suite': 'toy-v1', 'version': 1, 'leaderboard': False, 'label': 'Toy suite for tests.',
        'hf_dataset_id': 'unused', 'hf_revision': 'unused',
        'evaluation': {'required_metrics': ['sim'], 'seed': 42, 'bootstrap': {'enabled': False}},
        'tasks': [{'task': 'toy', 'dataset_config': 'libritts', 'hf_config_name': 'Toy',
                   'manifest': 'toy.metadata.json', 'selection': 'toy.selection.json'}]}))
    return directory / 'suite.json', tmp_path / 'data', tmp_path


def export(suite):
    spec, data_root, tmp = suite
    submission.export_prompts(spec, tmp / 'prompts', data_root=data_root)
    entries = [json.loads(line) for line in (tmp / 'prompts' / 'prompts.jsonl').read_text().splitlines()]
    return entries, tmp / 'prompts'


def echo(entries, prompts, directory):
    for entry in entries:
        destination = directory / entry['output_file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prompts / entry['reference_audio'], destination)


def score(suite, generated, name='scored'):
    spec, data_root, tmp = suite
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=fake_evaluate):
        return submission.score_submission(spec, generated, tmp / name, model='echo', data_root=data_root)


def test_export_contains_references_and_never_targets(suite):
    entries, prompts = export(suite)
    assert [entry['output_file'] for entry in entries] == [f'toy/Toy-{s}-{i:06d}.wav' for i, (s, *_) in enumerate(PAIRS)]
    assert [entry['text'] for entry in entries] == [text for *_, text in PAIRS]
    for entry in entries:
        assert file_hash(prompts / entry['reference_audio']) == entry['reference_sha256']
        assert entry['reference_audio'].endswith('.wav') and entry['reference_text'] == 'reference words'
    exported = {file_hash(path) for path in prompts.rglob('*.wav')}
    targets = {file_hash(path) for path in suite[1].rglob('*.wav')} - {e['reference_sha256'] for e in entries}
    assert targets and not exported & targets
    assert 'toy-v1' in (prompts / 'README.md').read_text()
    assert json.loads((prompts / 'suite.json').read_text())['leaderboard'] is False


def test_complete_submission_is_reportable(suite):
    entries, prompts = export(suite)
    echo(entries, prompts, suite[2] / 'generated')
    result = score(suite, suite[2] / 'generated')
    assert result['status'] == 'complete' and result['leaderboard'] is False and result['model'] == 'echo'
    task = result['tasks']['toy']
    assert task['means'] == {'sim': 0.5} and task['failures'] == {}
    # Echoing a reference that is also the pair's target is allowed: that file was exported.
    assert task['coverage']['generated'] == len(PAIRS)
    manifest = json.loads((suite[2] / 'scored' / task['run_manifest']).read_text())
    assert manifest['protocol'] == 'rvcbench-submission-v1' and manifest['generation_provenance'] is None
    assert validate_report(manifest)['eligible_for_comparison']
    assert json.loads((suite[2] / 'scored' / 'submission.json').read_text()) == result
    for row in manifest['samples']:
        assert Path(row['generated_path']).parent == suite[2] / 'scored' / 'toy' / 'generated_audio'


def test_missing_invalid_and_copied_target_files_fail_their_samples(suite):
    entries, prompts = export(suite)
    generated = suite[2] / 'generated'
    echo(entries[1:], prompts, generated)
    sf.write(generated / entries[1]['output_file'], np.full(800, np.nan), 16000, subtype='FLOAT')
    target = suite[1] / 'Toy' / 'audios' / 's2' / 'e.wav'
    shutil.copyfile(target, generated / entries[2]['output_file'])
    result = score(suite, generated)
    task = result['tasks']['toy']
    assert result['status'] == 'partial' and task['status'] == 'partial' and 'means' not in task
    failures = task['failures']
    assert 'Missing submitted audio' in failures[entries[0]['pair_id']]
    assert 'Invalid submitted audio' in failures[entries[1]['pair_id']]
    assert failures[entries[2]['pair_id']] == 'Submitted audio is the target recording'
    manifest = json.loads((suite[2] / 'scored' / task['run_manifest']).read_text())
    with pytest.raises(ValueError, match='incomplete'):
        validate_report(manifest)


@pytest.mark.parametrize('command', ['prompts', 'score'])
def test_changed_inputs_are_rejected(suite, command):
    spec, data_root, tmp = suite
    sf.write(data_root / 'Toy' / 'audios' / 's1' / 'a.wav', np.zeros(1600) + 0.01, 16000)
    with pytest.raises(ValueError, match='differ from the frozen suite'):
        if command == 'prompts':
            submission.export_prompts(spec, tmp / 'prompts', data_root=data_root)
        else:
            (tmp / 'generated').mkdir()
            score(suite, tmp / 'generated')


def test_output_directories_must_be_new(suite):
    entries, prompts = export(suite)
    with pytest.raises(ValueError, match='not empty'):
        submission.export_prompts(suite[0], prompts, data_root=suite[1])
    echo(entries, prompts, suite[2] / 'generated')
    score(suite, suite[2] / 'generated')
    with pytest.raises(FileExistsError):
        score(suite, suite[2] / 'generated')


def test_unknown_suite_lists_the_packaged_ones():
    with pytest.raises(ValueError, match='Available: core-v1, onboarding-v1'):
        submission.load_suite('core-v0')


@pytest.mark.parametrize('name', ['libritts16_v1', 'vctk16_v1', 'robotcall20_v1'])
def test_onboarding_suite_is_a_byte_copy_of_the_frozen_subsets(name):
    packaged = submission.SUITES_DIR / 'onboarding_v1'
    frozen = ROOT / 'reproduction' / 'subsets' / name
    assert (packaged / f'{name}.metadata.json').read_bytes() == (frozen / 'metadata.json').read_bytes()
    assert (packaged / f'{name}.selection.json').read_bytes() == (frozen / 'selection.json').read_bytes()


def test_onboarding_suite_is_not_a_leaderboard_suite():
    spec = submission.load_suite('onboarding-v1')
    assert spec['leaderboard'] is False and 'not RVCBench benchmark results' in spec['label']
    assert len(spec['hf_revision']) == 40
    assert [task['task'] for task in spec['tasks']] == ['libritts', 'vctk', 'robotcall']


def test_score_command_reports_status_and_label(suite, monkeypatch, capsys):
    from rvcbench.benchmark import cli
    entries, prompts = export(suite)
    echo(entries[1:], prompts, suite[2] / 'generated')
    spec, data_root, tmp = suite
    monkeypatch.setattr(sys, 'argv', ['rvcbench', 'score', '--suite', str(spec), '--generated', str(tmp / 'generated'),
                                      '--model', 'echo', '--output', str(tmp / 'cli'), '--data-root', str(data_root)])
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=fake_evaluate), pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code is True or stopped.value.code == 1
    printed = capsys.readouterr().out
    assert '"status": "partial"' in printed and 'Note: Toy suite for tests.' in printed


def test_prompts_command_runs_from_any_directory(suite):
    spec, data_root, tmp = suite
    result = subprocess.run([sys.executable, '-m', 'rvcbench.benchmark.cli', 'prompts', '--suite', str(spec),
                             '--output', str(tmp / 'cli-prompts'), '--data-root', str(data_root)],
                            cwd=tmp, capture_output=True, text=True,
                            env={**__import__('os').environ, 'PYTHONPATH': str(ROOT / 'src')})
    assert result.returncode == 0, result.stderr
    assert len((tmp / 'cli-prompts' / 'prompts.jsonl').read_text().splitlines()) == len(PAIRS)
