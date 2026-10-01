import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.benchmark.model_assets import resolve_model_assets
from src.models.fireredtts2.generator import FireRedTTS2Generator, FireRedTTS2GeneratorConfig


def generator(audio):
    wrapper = FireRedTTS2Generator(FireRedTTS2GeneratorConfig('fixture'),
        torch.device('cpu'), logging.getLogger(__name__))
    wrapper._generator = SimpleNamespace(sample_rate=16000,
        generate_monologue=lambda **kwargs: audio)
    wrapper.ensure_model = lambda: None
    return wrapper


def test_codec_output_rate_is_independent_of_prompt_input_rate():
    wrapper = generator(np.ones((1, 24000), dtype=np.float32) * 1.1)
    audio, rate = wrapper.generate(text='Target.')
    assert rate == 24000 and audio.shape == (24000,)
    assert audio.max() > 1  # Do not silently clip the native result.
    wrapper.close()
    assert wrapper._generator is wrapper.model is wrapper._spliter_module is None
    assert not wrapper.is_model_ready()


@pytest.mark.parametrize('kwargs', [dict(text=''), dict(text='Target.', prompt_wav='ref.wav'),
    dict(text='Target.', prompt_text='Reference.')])
def test_missing_content_or_unpaired_reference_fails_before_loading(kwargs):
    wrapper = generator(np.ones(10))
    wrapper.ensure_model = lambda: pytest.fail('Invalid request must not load model')
    with pytest.raises(ValueError):
        wrapper.generate(**kwargs)


@pytest.mark.parametrize('audio', [np.array([]), np.array([np.nan]),
    np.array([np.inf]), np.ones((2, 10))])
def test_invalid_native_audio_is_rejected(audio):
    with pytest.raises(ValueError, match='waveform'):
        generator(audio).generate(text='Target.')


def test_pretrained_weights_and_tokenizer_change_asset_fingerprint(tmp_path):
    (tmp_path / 'codec.pt').write_bytes(b'codec')
    tokenizer = tmp_path / 'Qwen2.5-1.5B'
    tokenizer.mkdir()
    (tokenizer / 'tokenizer.json').write_text('{}')
    conf = OmegaConf.create({'vc': {'model': 'fireredtts2'},
        'adversary': {'pretrained_dir': str(tmp_path)}})
    _, reference, first = resolve_model_assets(conf)
    assert 'Qwen2.5-1.5B/tokenizer.json' in reference['assets']['pretrained_dir']['files']
    (tmp_path / 'codec.pt').write_bytes(b'changed')
    assert resolve_model_assets(conf)[2] != first
    second = resolve_model_assets(conf)[2]
    (tokenizer / 'tokenizer.json').write_text('{"changed": true}')
    assert resolve_model_assets(conf)[2] != second
