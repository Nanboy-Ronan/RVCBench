"""Run-scoped protection artifacts must survive loading without source writes."""
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import soundfile as sf
import torch

from rvcbench.protection.base_protector import BaseProtector
from rvcbench.protection.enkidu import EnkiduProtector


def protector(tmp_path):
    obj = EnkiduProtector.__new__(EnkiduProtector)
    obj.output_dir = tmp_path / 'run'
    obj.config = SimpleNamespace()
    obj.dataset_config = SimpleNamespace(speaker_id=None, sampling_rate=24000)
    obj.logger = logging.getLogger('protection-test')
    obj.device = 'cpu'
    obj.protect_method = 'enkidu'
    obj.noises = {}
    for key, value in dict(n_fft=256, hop_length=64, win_length=256,
                           frame_length=4, noise_level=.1, random_offset=False,
                           mask_ratio=0, noise_smooth=False, sampling_rate=16000,
                           learning_rate=.01, decay=0, perturbation_epochs=2).items():
        setattr(obj, key, value)
    obj.window = torch.hann_window(obj.n_fft)
    batch = SimpleNamespace(wav=torch.zeros(1, 1, 1024), wav_len=[1024], path_out=['a.wav'])
    obj.speaker_data = SimpleNamespace(speakers_ids=['one'], speaker_dataloaders={'one': [batch]})
    return obj


def test_enkidu_generates_noise_only_in_run_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'enkidu.noise').write_bytes(b'unrelated existing archive')
    obj = protector(tmp_path)
    with patch.object(obj, '_protect_losses', side_effect=lambda clean, noisy: (noisy.square().mean(), {})):
        obj.generate_perturbations()
    saved = torch.load(obj.output_dir / 'enkidu.noise', weights_only=True)
    assert set(saved) == {'one'} and torch.isfinite(saved['one']['real']).all()
    assert (tmp_path / 'enkidu.noise').read_bytes() == b'unrelated existing archive'


@pytest.mark.parametrize('kind', ['enkidu', 'additive'])
def test_protection_reload_uses_run_root_and_preserves_waveform_rate(tmp_path, kind):
    obj = protector(tmp_path)
    obj.output_dir.mkdir()
    if kind == 'enkidu':
        noises = {'one': {'real': torch.zeros(1, 129, 4), 'imag': torch.zeros(1, 129, 4)}}
        method, expected_rate = obj.save_protected_audio_by_speaker, 16000
    else:
        obj.protect_method = 'gr'
        noises = {'one': [torch.zeros(1, 1, 1024)]}
        batch = obj.speaker_data.speaker_dataloaders['one'][0]
        batch.to = lambda device: batch
        method = lambda sid: BaseProtector.save_protected_audio_by_speaker(obj, sid)
        expected_rate = 24000
    torch.save(noises, obj.output_dir / f'{obj.protect_method}.noise')
    obj.noises = None
    method('one')
    info = sf.info(obj.output_dir / 'one/a.wav')
    assert info.frames == 1024 and info.samplerate == expected_rate


@pytest.mark.parametrize('config', [dict(batch_size=2, sampling_rate=16000),
                                    dict(batch_size=1, sampling_rate=24000)])
def test_enkidu_rejects_incompatible_batches_and_sample_rates_before_loading(config):
    with pytest.raises(ValueError):
        EnkiduProtector(None, config=SimpleNamespace(**config))
