"""Legacy reference identity must be proved, including target-only file names."""
from pathlib import Path

import pytest

from src.benchmark.reproduction import historical_prompt_ledger


def test_robotcall_conditions_share_one_bootstrap_cluster():
    from src.benchmark.reproduction import speaker_cluster
    assert speaker_cluster('Robotcall', 'p227robocall') == speaker_cluster('Robotcall', 'p227vctk')
    assert speaker_cluster('Robotcall', 'p232vctk') != speaker_cluster('Robotcall', 'p227vctk')
    assert speaker_cluster('VCTK', 'p227') == 'p227'
    with pytest.raises(ValueError, match='speaker-condition'):
        speaker_cluster('Robotcall', 'unverified_alias')


def test_target_only_ledger_requires_complete_ordered_log(tmp_path):
    rows = [dict(speaker_id='one', ground_truth_text='First sentence.', generated_path='first.wav'),
            dict(speaker_id='two', ground_truth_text='Second sentence.', generated_path='second.wav')]
    log = tmp_path / 'generation.log'
    log.write_text('[FishSpeech] [1/2] speaker=one prompt=ref1.wav (sr=24000) text="First sentence."\n'
                   '[FishSpeech] [2/2] speaker=two prompt=ref2.wav (sr=24000) text="Second sentence."\n')
    ledger = historical_prompt_ledger(tmp_path / 'scores.csv', rows)
    assert ledger['first.wav']['prompt_name'] == 'ref1.wav'
    assert ledger['first.wav']['source_log_sha256']
    with pytest.raises(ValueError, match='complete, CSV-aligned'):
        historical_prompt_ledger(tmp_path / 'scores.csv', list(reversed(rows)))
    log.write_text(log.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='complete, CSV-aligned'):
        historical_prompt_ledger(tmp_path / 'scores.csv', rows)


def test_wer_protocol_inferred_from_all_transcripts(tmp_path):
    import csv
    from src.benchmark.reproduction import infer_english_wer_protocol
    path = tmp_path / 'scores.csv'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ground_truth_text', 'predicted_text', 'wer'])
        writer.writeheader()
        writer.writerows([dict(ground_truth_text='Hello, world.', predicted_text='Hello world', wer=1),
                          dict(ground_truth_text='Same.', predicted_text='Same.', wer=0)])
    assert infer_english_wer_protocol(path)['normalization'] == 'lowercase_v1'
    with path.open('a') as handle:
        handle.write('different,words,0\n')
    with pytest.raises(ValueError, match='unsupported or ambiguous'):
        infer_english_wer_protocol(path)


@pytest.mark.parametrize('dataset', ['LibriTTS', 'VCTK', 'Robotcall'])
def test_pair_match_requires_dataset_identity_and_immutable_reference_audio(tmp_path, dataset):
    import csv
    import json
    from src.benchmark.artifacts import file_hash
    from src.benchmark.reproduction import match_historical
    audio = tmp_path / 'audios' / 'one'
    audio.mkdir(parents=True)
    prompt, target = audio / 'prompt.wav', audio / 'target.wav'
    prompt.write_bytes(b'prompt')
    target.write_bytes(b'target')
    config = {'vc': {'model': 'smoke'}, 'dataset': {'name': dataset, 'root_path': str(tmp_path)}}
    (tmp_path / 'metrics.json').write_text(json.dumps({'config': config}))
    csv_path = tmp_path / 'generation_sample_metrics.csv'
    old = {'speaker_id': 'one', 'ground_truth_text': 'Target.', 'ground_truth_path': str(target),
           'generated_path': 'prompt_to_target_cloned.wav'}
    with csv_path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(old))
        writer.writeheader()
        writer.writerow(old)
    row = dict(sample_id='fixture', speaker_id='one', prompt_path=str(prompt), target_path=str(target),
               target_text='Target.', prompt_sha256=file_hash(prompt), target_sha256=file_hash(target))
    run = {'config': config, 'samples': [row]}
    assert len(match_historical(run, csv_path)[1]) == 1
    run['config'] = {**config, 'dataset': {**config['dataset'], 'name': 'other'}}
    with pytest.raises(ValueError, match='dataset identity'):
        match_historical(run, csv_path)
    run['config'] = config
    prompt.write_bytes(b'changed reference')
    with pytest.raises(ValueError, match='input/content mismatch'):
        match_historical(run, csv_path)


def test_external_audio_reference_requires_unique_manifest_and_target_hash(tmp_path):
    import json
    from src.benchmark.reproduction import historical_manifest_prompt
    root = tmp_path / 'short10'
    (root / 'filelists').mkdir(parents=True)
    audio = tmp_path / 'Libritts' / 'audios' / 'one'
    audio.mkdir(parents=True)
    prompt, target = audio / 'prompt.wav', audio / 'target.wav'
    prompt.write_bytes(b'prompt')
    target.write_bytes(b'target')
    path = root / 'filelists' / 'one.json'
    record = {'ori_pth': 'Libritts/audios/one/prompt.wav',
              'gt_pth': 'Libritts/audios/one/target.wav', 'gt_text': 'Target.'}
    path.write_text(json.dumps([record]))
    row = {'speaker_id': 'one', 'prompt_path': str(prompt), 'target_path': str(target)}
    old = {'ground_truth_text': 'Target.', 'ground_truth_path': str(target)}
    resolved, evidence = historical_manifest_prompt(root, row, old)
    assert resolved == prompt
    assert evidence['source_row'] == 0 and evidence['source_manifest_sha256']
    path.write_text(json.dumps([record, record]))
    with pytest.raises(ValueError, match='one exact'):
        historical_manifest_prompt(root, row, old)
    path.write_text(json.dumps([record]))
    other = tmp_path / 'target.wav'
    other.write_bytes(b'other target')
    with pytest.raises(ValueError, match='target differs'):
        historical_manifest_prompt(root, row, {**old, 'ground_truth_path': str(other)})
