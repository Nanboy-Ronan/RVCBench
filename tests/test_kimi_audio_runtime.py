"""Kimi runtime state, auxiliary source and output contracts without downloads."""
from contextlib import contextmanager
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from rvcbench.models.kimi_audio import KimiAudioGenerator, KimiAudioGeneratorConfig


def prepared_generator(wave):
    generator = KimiAudioGenerator(KimiAudioGeneratorConfig('.', 'fixture', seed=42), 'cuda:0', logging.getLogger())
    generator._model = Mock()
    generator._model.generate.return_value = (wave, 'spoken text')
    generator._model_ready = True
    return generator


@pytest.mark.parametrize('fail', [False, True])
def test_source_index_seed_and_cuda_scope_restore(fail):
    generator = prepared_generator(torch.zeros(1, 24))
    state = {'device': 'caller'}
    @contextmanager
    def scope(device):
        previous = state['device']
        state['device'] = device
        try:
            yield
        finally:
            state['device'] = previous
    def generate(*args, **kwargs):
        assert state['device'] == torch.device('cuda:0')
        assert kwargs['output_type'] == 'both' and kwargs['max_new_tokens'] == -1
        if fail:
            raise RuntimeError('fixture')
        return torch.zeros(1, 24), 'spoken text'
    generator._model.generate.side_effect = generate
    with patch('torch.cuda.device', side_effect=scope), patch('rvcbench.models.kimi_audio.generator.configure_seeds') as seed:
        if fail:
            with pytest.raises(RuntimeError, match='fixture'):
                generator.generate([], sample_index=73)
        else:
            wave, rate = generator.generate([], sample_index=73)
            assert wave.shape == (24,) and rate == 24000
            assert generator.last_generated_text == 'spoken text'
        seed.assert_called_once_with(115)
    assert state['device'] == 'caller'
    generator.close()
    assert generator._model is generator.model is None and not generator._model_ready


@pytest.mark.parametrize('wave,message', [(None, 'no waveform'), (np.array([]), 'empty'),
    (np.array([np.nan]), 'nonfinite'), (np.zeros((2, 3)), 'one mono')])
def test_invalid_output_is_rejected(wave, message):
    with patch('torch.cuda.device'), patch('rvcbench.models.kimi_audio.generator.configure_seeds'):
        with pytest.raises(RuntimeError, match=message):
            prepared_generator(wave).generate([])


@pytest.mark.parametrize('fail', [False, True])
def test_auxiliary_tokenizer_override_is_scoped(tmp_path, fail):
    original = Mock(return_value='local tokenizer')
    module = SimpleNamespace(Glm4Tokenizer=original)
    generator = KimiAudioGenerator(KimiAudioGeneratorConfig('.', 'fixture', audio_tokenizer_path=str(tmp_path)), 'cuda:0', logging.getLogger())
    with patch('importlib.import_module', return_value=module):
        try:
            with generator._tokenizer_scope():
                assert module.Glm4Tokenizer('THUDM/glm-4-voice-tokenizer') == 'local tokenizer'
                if fail:
                    raise RuntimeError('fixture')
        except RuntimeError:
            assert fail
    assert module.Glm4Tokenizer is original
    original.assert_called_once_with(str(tmp_path))


def test_constructor_failure_restores_device_and_releases_partial_resources():
    generator = prepared_generator(torch.zeros(24))
    generator.close()
    generator._kimi_cls = Mock(side_effect=RuntimeError('constructor failed'))
    with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device') as scope:
        with pytest.raises(RuntimeError, match='constructor failed'):
            generator.ensure_model()
    assert scope.return_value.__exit__.call_count == 1
    assert generator.model is generator._model is None and not generator._model_ready


@pytest.mark.parametrize('options,message', [({'sample_rate':16000}, 'sample rate'),
    ({'max_new_tokens':100}, 'overrides max_new_tokens'), ({'load_detokenizer':False}, 'load_detokenizer')])
def test_reject_settings_the_official_audio_api_cannot_honor(options, message):
    generator = KimiAudioGenerator(KimiAudioGeneratorConfig('.', 'fixture', **options), 'cuda:0', logging.getLogger())
    with patch('torch.cuda.is_available', return_value=True):
        with pytest.raises(ValueError, match=message):
            generator.ensure_model()
