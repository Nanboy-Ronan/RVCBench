import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.benchmark.artifacts import file_hash
from src.benchmark.noise_replay import replay_gr_noise


@pytest.fixture
def replay_fixture(tmp_path):
    root, history = tmp_path / 'dataset', tmp_path / 'history' / 'one'
    root.mkdir()
    history.mkdir(parents=True)
    rows = []
    noise = torch.zeros(2, 1, 2048)
    noise[0] = .125  # Long reference sorts first, despite its later manifest row.
    noise[1] = .25
    for index, frames in enumerate((1024, 2048)):
        name = f'{index}.wav'
        audio = np.full(frames, 1000, dtype=np.int16)
        sf.write(root / name, audio, 24000, subtype='PCM_16')
        sf.write(history / name, audio.astype(np.float32) / 32768 + (.25 if index == 0 else .125),
                 24000, subtype='PCM_16')
        rows.append({'pair_id': f'p{index}', 'speaker_id': 'one', 'dataset_name': 'LibriTTS', 'split': 'default',
                     'manifest_variant': 'speaker', 'prompt_file_name': name, 'target_file_name': name,
                     'prompt_text': 'hello', 'target_text': 'hello',
                     'prompt_phonemes': 'h e', 'target_phonemes': 'h e'})
    pd.DataFrame(rows).to_parquet(root / 'metadata.parquet')
    subset = tmp_path / 'subset.json'
    subset.write_text(json.dumps([rows[1]]))
    archive = tmp_path / 'gr.noise'
    torch.save({'one': [noise]}, archive)
    return root, subset, archive, history.parent, tmp_path / 'output'


def test_replay_uses_original_batch_slot_and_records_verified_stage(replay_fixture):
    root, subset, archive, history, output = replay_fixture
    before = file_hash(archive), file_hash(root / '1.wav')
    result = replay_gr_noise(*replay_fixture, batch_size=2)
    assert result['status'] == 'complete' and result['verified'] == result['requested'] == 1
    row = result['rows'][0]
    assert row['archive_batch'] == row['archive_slot'] == 0
    assert row['reference_sha256'] == file_hash(history / 'one' / '1.wav')
    assert before == (file_hash(archive), file_hash(root / '1.wav'))
    assert json.loads((output / 'stage_manifest.json').read_text()) == result
    with pytest.raises(FileExistsError):
        replay_gr_noise(*replay_fixture, batch_size=2)


@pytest.mark.parametrize('failure', ['missing_clean', 'archive_shape', 'changed_subset'])
def test_replay_rejects_invalid_batch_mapping_before_creating_outputs(replay_fixture, failure):
    root, subset, archive, _, output = replay_fixture
    if failure == 'missing_clean':
        (root / '0.wav').unlink()  # Unselected cohort row still determines batch positions.
    elif failure == 'archive_shape':
        torch.save({'one': [torch.zeros(1, 1, 2048)]}, archive)
    else:
        rows = json.loads(subset.read_text())
        rows[0]['prompt_text'] = 'different'
        subset.write_text(json.dumps(rows))
    with pytest.raises((FileNotFoundError, ValueError)):
        replay_gr_noise(*replay_fixture, batch_size=2)
    assert not output.exists()


def test_replay_mismatch_is_failed_and_never_reported_as_verified(replay_fixture):
    _, _, _, history, output = replay_fixture
    sf.write(history / 'one' / '1.wav', np.zeros(2048), 24000, subtype='PCM_16')
    with pytest.raises(ValueError, match='differs from historical'):
        replay_gr_noise(*replay_fixture, batch_size=2)
    result = json.loads((output / 'stage_manifest.json').read_text())
    assert result['status'] == 'failed' and result['verified'] == 0
    assert result['rows'][0]['reference_sha256'] != result['rows'][0]['historical_sha256']


def test_regenerated_rng_matches_full_batch_and_selected_historical_wav(replay_fixture):
    root, subset, archive, history, output = replay_fixture
    generator = torch.Generator().manual_seed(42)
    noise = torch.randn((2, 1, 2048), generator=generator) * .03137255
    torch.save({'one': [noise]}, archive)
    clean, rate = sf.read(root / '1.wav', dtype='int16')
    sf.write(history / 'one' / '1.wav',
             (torch.from_numpy(clean).float() / 32768 + noise[0, 0]).clamp(-1, 1).numpy(),
             rate, subtype='PCM_16')
    result = replay_gr_noise(*replay_fixture, batch_size=2, rng_seed=42)
    assert result['variant'] == 'gr_seeded_batch_rng_v1'
    assert result['rng_verification']['requested_batches'] == result['rng_verification']['verified_batches'] == 1
    assert result['verified'] == 1
    assert file_hash(output / 'protected_audio/one/1.wav') == file_hash(history / 'one/1.wav')


def test_regenerated_rng_mismatch_fails_before_any_reference_is_written(replay_fixture):
    with pytest.raises(ValueError, match='Regenerated RNG differs'):
        replay_gr_noise(*replay_fixture, batch_size=2, rng_seed=42)
    output = replay_fixture[-1]
    result = json.loads((output / 'stage_manifest.json').read_text())
    assert result['status'] == 'failed' and result['verified'] == 0
    assert result['rng_verification']['verified_batches'] == 0
    assert not list(output.rglob('*.wav'))
