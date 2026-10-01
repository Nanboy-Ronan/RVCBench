import csv
import importlib.util
import json
from pathlib import Path
import wave

import pytest

spec = importlib.util.spec_from_file_location('freeze_historical_population',
    Path(__file__).resolve().parents[1] / 'scripts' / 'freeze_historical_population.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
freeze = module.freeze


@pytest.fixture
def population(tmp_path):
    source, audio = tmp_path / 'legacy', tmp_path / 'Libritts'
    (source / 'filelists').mkdir(parents=True)
    rows = []
    for sid in ('1', '2'):
        folder = audio / 'audios' / sid
        folder.mkdir(parents=True)
        for name in ('prompt', 'target'):
            with wave.open(str(folder / f'{name}.wav'), 'wb') as f:
                f.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                f.writeframes(b'\x01\x00' * 100)
        record = {'ori_pth': f'Libritts/audios/{sid}/prompt.wav', 'ori_spk': sid,
                  'ori_text': 'Prompt.', 'gt_pth': f'Libritts/audios/{sid}/target.wav',
                  'gt_spk': sid, 'gt_text': 'Target.'}
        (source / 'filelists' / f'{sid}.json').write_text(json.dumps([record]))
        # Alternate transcripts must not accidentally enter the frozen legacy population.
        (source / 'filelists' / f'{sid}_text.json').write_text(json.dumps([record]))
        rows.append({'speaker_id': sid, 'ground_truth_path': str(folder / 'target.wav'),
                     'ground_truth_text': 'Target.',
                     'generated_path': f'/old/{sid}/prompt_to_target_cloned.wav'})
    history = tmp_path / 'history.csv'
    def write_history(rows):
        with history.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_history(rows)
    return source, audio, history, rows, write_history


def test_entire_population_preserves_order_and_variant(population, tmp_path):
    source, audio, history, _, _ = population
    output = freeze(source, audio, history, tmp_path / 'frozen', 2)
    metadata = json.loads((output / 'metadata.json').read_text())
    selection = json.loads((output / 'selection.json').read_text())
    assert [r['source_index'] for r in metadata] == [0, 1]
    assert [r['source_manifest'] for r in metadata] == ['1.json', '2.json']
    assert {r['manifest_variant'] for r in metadata} == {'speaker'}
    assert selection['population'] == selection['requested'] == 2
    assert all(r['prompt_sha256'] and r['target_sha256'] for r in selection['samples'])


@pytest.mark.parametrize('mutation', ['order', 'transcript', 'pair', 'audio'])
def test_rejects_unmatched_history_before_writing(population, tmp_path, mutation):
    source, audio, history, rows, write_history = population
    if mutation == 'order':
        rows.reverse()
    elif mutation == 'transcript':
        rows[0]['ground_truth_text'] = 'Another target.'
    elif mutation == 'pair':
        rows[0]['generated_path'] = '/old/different_to_target_cloned.wav'
    else:
        changed = tmp_path / 'historical_audio'
        changed.mkdir()
        (changed / 'target.wav').write_bytes(b'different audio')
        rows[0]['ground_truth_path'] = str(changed / 'target.wav')
    write_history(rows)
    output = tmp_path / 'frozen'
    with pytest.raises(ValueError, match='mismatch'):
        freeze(source, audio, history, output, 2)
    assert not output.exists()
