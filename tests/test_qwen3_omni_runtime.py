"""Omni output contracts independent of optional Transformers installation."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from src.models.qwen3_omni import Qwen3OmniGenerator, Qwen3OmniGeneratorConfig


class Inputs(dict):
    def to(self, value):
        return self


def prepared_generator(audio):
    generator = Qwen3OmniGenerator(Qwen3OmniGeneratorConfig('fixture', seed=42), 'cpu', logging.getLogger())
    generator._model = Mock(device=torch.device('cpu'), dtype=torch.float32)
    generator._model.generate.return_value = (torch.tensor([[1, 2, 3]]), audio)
    generator._processor = Mock(feature_extractor=SimpleNamespace(sampling_rate=16000))
    generator._processor.return_value = Inputs(input_ids=torch.tensor([[1, 2]]))
    generator._processor.batch_decode.return_value = ['spoken text']
    generator._process_mm_info = Mock(return_value=([], [], []))
    generator._model_ready = True
    return generator


@pytest.mark.parametrize('audio', [torch.zeros(240), [torch.zeros(240)]])
def test_output_rate_does_not_use_input_feature_extractor_rate_and_seed_preserves_index(audio):
    generator = prepared_generator(audio)
    with patch('src.utils.seeding.configure_seeds') as seed:
        waveform, rate = generator.generate([{'role': 'user', 'content': []}], sample_index=73)
    assert waveform.shape == (240,) and rate == 24000
    assert seed.call_args.args == (115,)
    assert generator.last_generated_text == 'spoken text'
    generator.close()
    assert generator._model is generator.model is generator._processor is None
    assert not generator._model_ready


@pytest.mark.parametrize('audio,message', [(None, 'no audio'), (torch.tensor([]), 'empty'),
    (torch.tensor([float('nan')]), 'nonfinite'), ([torch.zeros(2), torch.zeros(2)], 'one waveform')])
def test_invalid_waveform_is_rejected(audio, message):
    with pytest.raises(RuntimeError, match=message):
        prepared_generator(audio).generate([])


def test_processor_load_failure_releases_already_loaded_weights():
    generator = Qwen3OmniGenerator(Qwen3OmniGeneratorConfig('fixture', device_map=None), 'cpu', logging.getLogger())
    model = Mock()
    transformers = SimpleNamespace(Qwen3OmniMoeForConditionalGeneration=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        Qwen3OmniMoeProcessor=SimpleNamespace(from_pretrained=Mock(side_effect=RuntimeError('processor failed'))))
    with patch('importlib.import_module', side_effect=lambda name: transformers if name == 'transformers' else SimpleNamespace()):
        with pytest.raises(RuntimeError, match='processor failed'):
            generator.ensure_model()
    model.to.assert_called_once_with(torch.device('cpu'))
    assert generator._model is generator.model is None
    assert not generator._model_ready
