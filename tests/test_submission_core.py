import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from rvcbench.benchmark import submission
from rvcbench.benchmark.artifacts import validate_report
from test_submission import PAIRS, echo, export, suite  # noqa: F401  (fixture)


VALUES = {'sim': 0.5, 'wer': 0.2, 'mcd': 5.0, 'speechmos': 3.0, 'sva': 1}


def fake_evaluate(rows, required, output, device, logger, **kwargs):
    """Deterministic scores; the noisy task scores worse so relative change is visible."""
    factor = 0.8 if Path(output).name.endswith('noisy') else 1.0
    for row in rows:
        if row.get('generated_sha256'):
            row['metrics'] = {metric: (VALUES[metric] * factor if metric != 'sva' else 1) for metric in required}
            row['status'] = 'complete'
    return {'evaluation_fingerprint': 'test-only-fingerprint'}


def rewrite(suite, tasks):
    path = suite[0]
    spec = json.loads(path.read_text())
    base = spec['tasks'][0]
    spec['tasks'] = [{**base, **task} for task in tasks]
    path.write_text(json.dumps(spec))
    return path


def score(suite, generated):
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=fake_evaluate):
        return submission.score_submission(suite[0], generated, suite[2] / 'scored', model='echo', data_root=suite[1])


def test_task_metrics_anchor_and_groups(suite):
    rewrite(suite, [
        {'task': 'toy', 'dimension': 'generation', 'evaluation': 'English-VC', 'required_metrics': ['sim', 'wer', 'mcd'],
         'group_by': 'speaker_id'},
        {'task': 'toy-noisy', 'dimension': 'perturbation', 'evaluation': 'Background', 'anchor': 'toy',
         'required_metrics': ['sim', 'wer', 'sva']},
    ])
    entries, prompts = export(suite)
    assert {entry['task'] for entry in entries} == {'toy', 'toy-noisy'}
    echo(entries, prompts, suite[2] / 'generated')
    result = score(suite, suite[2] / 'generated')
    clean, noisy = result['tasks']['toy'], result['tasks']['toy-noisy']
    assert result['status'] == 'complete'
    assert clean['required_metrics'] == ['sim', 'wer', 'mcd'] and noisy['required_metrics'] == ['sim', 'wer', 'sva']
    assert clean['dimension'] == 'generation' and noisy['evaluation'] == 'Background'
    assert set(clean['group_means']['speaker_id']) == {'s1', 's2'}
    assert clean['group_means']['speaker_id']['s1']['sim'] == pytest.approx(0.5)
    assert noisy['relative_change_percent'] == {'sim': pytest.approx(-20.0), 'wer': pytest.approx(-20.0)}
    assert 'relative_change_percent' not in clean


@pytest.mark.parametrize('transform', [
    {'kind': 'codec', 'codec': 'mp3', 'bitrate': '32k'},
    {'kind': 'codec', 'codec': 'aac', 'bitrate': '64k'},
    {'kind': 'codec', 'codec': 'opus', 'bitrate': '16k'},
    {'kind': 'narrowband'},
])
def test_transforms_decode_to_mono_pcm_at_the_evaluation_rate(tmp_path, transform):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg not installed')
    source = tmp_path / 'clone.wav'
    sf.write(source, 0.2 * np.sin(2 * np.pi * 440 * np.arange(24000) / 24000), 24000)
    destination = tmp_path / 'out.wav'
    submission.transform_audio(source, destination, transform, tmp_path)
    info = sf.info(destination)
    assert (info.samplerate, info.channels, info.subtype) == (24000, 1, 'PCM_16')
    assert abs(info.frames - 24000) < 2400
    assert sorted(path.name for path in tmp_path.iterdir()) == ['clone.wav', 'out.wav']


def test_narrowband_removes_energy_above_the_telephone_band(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg not installed')
    t = np.arange(24000) / 24000
    source = tmp_path / 'clone.wav'
    sf.write(source, 0.2 * np.sin(2 * np.pi * 1000 * t) + 0.2 * np.sin(2 * np.pi * 6000 * t), 24000)
    submission.transform_audio(source, tmp_path / 'phone.wav', {'kind': 'narrowband'}, tmp_path)
    audio, rate = sf.read(tmp_path / 'phone.wav')
    spectrum = np.abs(np.fft.rfft(audio))
    frequency = np.fft.rfftfreq(audio.size, 1 / rate)
    band = lambda f: spectrum[(frequency > f - 50) & (frequency < f + 50)].max()  # noqa: E731
    assert band(6000) < 0.01 * band(1000)


def test_derived_tasks_compare_processed_and_unprocessed_clones(suite):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg not installed')
    rewrite(suite, [
        {'task': 'toy', 'required_metrics': ['sim']},
        {'task': 'toy-mp3-32k', 'derived_from': 'toy', 'transform': {'kind': 'codec', 'codec': 'mp3', 'bitrate': '32k'},
         'required_metrics': ['mcd', 'sim', 'wer'], 'dimension': 'output'},
    ])
    entries, prompts = export(suite)
    assert {entry['task'] for entry in entries} == {'toy'}
    generated = suite[2] / 'generated'
    echo(entries[1:], prompts, generated)
    result = score(suite, generated)
    derived = result['tasks']['toy-mp3-32k']
    assert derived['derived_from'] == 'toy' and derived['status'] == 'partial'
    assert derived['failures'] == {entries[0]['pair_id']: 'Source sample failed in toy'}
    manifest = json.loads((suite[2] / 'scored' / derived['run_manifest']).read_text())
    clones = json.loads((suite[2] / 'scored' / 'toy' / 'run_manifest.json').read_text())['samples']
    for row, clone in zip(manifest['samples'][1:], clones[1:]):
        assert row['target_path'] == clone['generated_path'] and row['target_sha256'] == clone['generated_sha256']
        assert Path(row['generated_path']).parent == suite[2] / 'scored' / 'toy-mp3-32k' / 'generated_audio'
        assert row['generated_sha256'] != clone['generated_sha256']


def test_complete_derived_task_is_reportable(suite):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg not installed')
    rewrite(suite, [{'task': 'toy', 'required_metrics': ['sim']},
                    {'task': 'toy-phone', 'derived_from': 'toy', 'transform': {'kind': 'narrowband'},
                     'required_metrics': ['sim', 'wer']}])
    entries, prompts = export(suite)
    echo(entries, prompts, suite[2] / 'generated')
    result = score(suite, suite[2] / 'generated')
    assert result['status'] == 'complete'
    manifest = json.loads((suite[2] / 'scored' / 'toy-phone' / 'run_manifest.json').read_text())
    assert validate_report(manifest)['eligible_for_comparison']


@pytest.mark.parametrize('tasks,message', [
    ([{'task': 'a'}, {'task': 'b', 'derived_from': 'missing', 'transform': {'kind': 'narrowband'}}], 'derived_from'),
    ([{'task': 'a'}, {'task': 'b', 'derived_from': 'a'}], 'transform'),
    ([{'task': 'a', 'anchor': 'missing'}], 'anchor'),
    ([{'task': 'a', 'required_metrics': ['sim', 'utmos']}], 'unknown metrics'),
])
def test_suite_definition_errors(suite, tasks, message):
    with pytest.raises(ValueError, match=message):
        submission.load_suite(rewrite(suite, tasks))


def test_hub_rate_limit_is_retried_then_explained(monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError
    import huggingface_hub

    import requests
    response = requests.Response()
    response.status_code = 429
    response.request = requests.Request('GET', 'https://huggingface.co/api/datasets/x/y').prepare()
    calls = []

    def limited(**kwargs):
        calls.append(kwargs)
        raise HfHubHTTPError('429 Too Many Requests', response=response)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', limited)
    spec = {'hf_dataset_id': 'x/y', 'hf_revision': 'r'}
    task = {'task': 't', 'hf_config_name': 'Libritts'}
    with pytest.raises(RuntimeError, match='hf auth login'):
        submission._fetch_from_hub(spec, task, [{'prompt_file_name': 'a.wav', 'target_file_name': 'b.wav'}],
                                   attempts=3, wait_seconds=0)
    assert len(calls) == 3 and calls[0]['max_workers'] == 4
    assert calls[0]['allow_patterns'] == ['Libritts/a.wav', 'Libritts/b.wav']



def test_an_incomplete_cached_snapshot_is_not_accepted(monkeypatch, tmp_path):
    import huggingface_hub
    (tmp_path / 'Libritts').mkdir()
    (tmp_path / 'Libritts' / 'a.wav').write_bytes(b'a')
    calls = []

    def partial_then_complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            (tmp_path / 'Libritts' / 'b.wav').write_bytes(b'b')
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', partial_then_complete)
    spec = {'hf_dataset_id': 'x/y', 'hf_revision': 'r'}
    task = {'task': 't', 'hf_config_name': 'Libritts'}
    rows = [{'prompt_file_name': 'a.wav', 'target_file_name': 'b.wav'}]
    assert submission._fetch_from_hub(spec, task, rows, attempts=3, wait_seconds=0) == tmp_path / 'Libritts'
    assert len(calls) == 2
    (tmp_path / 'Libritts' / 'b.wav').unlink()
    calls.clear()
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **kwargs: calls.append(kwargs) or str(tmp_path))
    with pytest.raises(RuntimeError, match='--data-root'):
        submission._fetch_from_hub(spec, task, rows, attempts=2, wait_seconds=0)
    assert len(calls) == 2
