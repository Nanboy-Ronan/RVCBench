"""VibeVoice checkout/API compatibility, truthful errors and owned resources."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
import pytest
import torch

from src.models.vibevoice.generator import VibeVoiceGenerator, VibeVoiceGeneratorConfig, _check_generation_cache_api
from src.adversary.vibevoice_ots import VibeVoiceZeroShotAdversary
from test_benchmark import setup_run


@pytest.mark.parametrize('requires_device', [False, True])
def test_checkout_cache_call_checked_against_runtime_signature(tmp_path, requires_device):
    source = tmp_path / 'vibevoice/modular/modeling_vibevoice_inference.py'
    source.parent.mkdir(parents=True)
    source.write_text('class Model:\n def generate(self):\n'
                      '  self._prepare_cache_for_generation(config, kwargs, None, batch, length)\n')
    class Modern:
        def _prepare_cache_for_generation(self, config, kwargs, mode, batch, length):
            pass
    class Old:
        def _prepare_cache_for_generation(self, config, kwargs, mode, batch, length, device):
            pass
    modules = {'transformers.generation.utils': SimpleNamespace(GenerationMixin=Old if requires_device else Modern)}
    with patch.dict('sys.modules', modules):
        if requires_device:
            with pytest.raises(RuntimeError, match='incompatible.*device'):
                _check_generation_cache_api(tmp_path)
        else:
            _check_generation_cache_api(tmp_path)


def test_runner_managed_adapter_preserves_original_generation_error(setup_run, tmp_path):
    _, dataset, _, _ = setup_run
    conf = OmegaConf.create({'code_path': str(tmp_path), 'model_path': 'fixture'})
    adapter = VibeVoiceZeroShotAdversary(conf, {}, torch.device('cpu'), logging.getLogger())
    adapter._generator = SimpleNamespace(generate=Mock(side_effect=RuntimeError('cache API fixture')))
    adapter._managed_lifetime = True
    with pytest.raises(RuntimeError, match='cache API fixture'):
        adapter.attack(output_path=tmp_path / 'outputs', dataset=dataset)


def test_close_releases_processor_and_model():
    generator = object.__new__(VibeVoiceGenerator)
    generator.model = generator.processor = object()
    generator._model_ready = True
    generator.close()
    assert generator.model is generator.processor is None
    assert not generator._model_ready


@pytest.mark.parametrize('attention_failure', [False, True])
def test_attention_fallback_does_not_hide_checkpoint_errors(tmp_path, attention_failure):
    failure = RuntimeError('FlashAttention2 unavailable' if attention_failure else 'checkpoint weights mismatch')
    model = torch.nn.Linear(2, 2)
    loader = Mock(side_effect=[failure, model])
    processor = SimpleNamespace(audio_processor=SimpleNamespace(sampling_rate=24000))
    modules = {
        'vibevoice.modular.modeling_vibevoice_inference': SimpleNamespace(
            VibeVoiceForConditionalGenerationInference=SimpleNamespace(from_pretrained=loader)),
        'vibevoice.processor.vibevoice_processor': SimpleNamespace(
            VibeVoiceProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor))),
        'vibevoice.modular.lora_loading': None,
    }
    config = VibeVoiceGeneratorConfig(code_path=tmp_path, model_path='fixture', num_inference_steps=None)
    with patch.object(VibeVoiceGenerator, '_ensure_repo_on_path'), \
            patch('src.models.vibevoice.generator._check_generation_cache_api'), \
            patch.dict('sys.modules', modules):
        generator = VibeVoiceGenerator(config, 'cpu', logging.getLogger())
        if attention_failure:
            generator.load_model()
            assert loader.call_count == 2
            assert loader.call_args.kwargs['attn_implementation'] == 'sdpa'
            assert generator._attn_impl == 'sdpa'
        else:
            with pytest.raises(RuntimeError, match='weights mismatch'):
                generator.load_model()
            assert loader.call_count == 1
