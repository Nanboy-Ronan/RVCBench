import logging
from types import ModuleType

import pytest
import torch

from src.models.bark_voice_clone.generator import (
    BarkVoiceCloneGenerator, BarkVoiceCloneGeneratorConfig, CPULoadTorch, private_runtime)


def test_private_bark_runtime_keeps_model_caches_and_device_selection_separate():
    module = ModuleType('native_fixture')
    module.models, module.models_devices, module.torch = {}, {}, torch
    module.GPT = module.FineGPT = torch.nn.Linear
    module.BertTokenizer = type('Tokenizer', (), {'from_pretrained': staticmethod(lambda *a, **k: None)})
    exec('def remember(key, value):\n models[key] = value\n return _grab_best_device()\n', vars(module))
    original_load = torch.load
    first = private_runtime(module, torch.device('cpu'), '/tokens')
    second = private_runtime(module, torch.device('cuda:3'), '/tokens')
    assert first['remember']('owned', 1) == 'cpu'
    assert second['remember']('other', 2) == 'cuda:3'
    assert module.models == {} and first['models'] == {'owned': 1} and second['models'] == {'other': 2}
    assert torch.load is original_load
    model = first['GPT'](2, 2)
    state = model.state_dict()
    state['unexpected_trained_weight'] = torch.ones(1)
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        model.load_state_dict(state)


def test_bark_cpu_loader_rejects_discarding_lora_adapters(tmp_path):
    path = tmp_path / 'model.pt'
    torch.save({'model': {'layer.lora_right_weight': torch.ones(1)}}, path)
    with pytest.raises(ValueError, match='refusing to discard adapters'):
        CPULoadTorch().load(path, weights_only=True)


def test_bark_requires_the_learned_tokenizers_native_hubert_layer(tmp_path):
    cfg = BarkVoiceCloneGeneratorConfig(code_path=tmp_path, hubert_layer=12)
    with pytest.raises(ValueError, match='layer 9'):
        BarkVoiceCloneGenerator(cfg, 'cpu', logging.getLogger())
    model = BarkVoiceCloneGenerator(BarkVoiceCloneGeneratorConfig(code_path=tmp_path), 'cpu', logging.getLogger())
    with pytest.raises(FileNotFoundError, match='explicit local'):
        model.load_model()
    with pytest.raises(ValueError, match='target text'):
        model.generate(text='', prompt_audio=tmp_path/'audio.wav')


def test_bark_scope_restores_matmul_policy_after_failure(tmp_path):
    model = BarkVoiceCloneGenerator(BarkVoiceCloneGeneratorConfig(code_path=tmp_path), 'cpu', logging.getLogger())
    before = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('medium')
        with pytest.raises(RuntimeError):
            with model._scope():
                raise RuntimeError('inference failed')
        assert torch.get_float32_matmul_precision() == 'medium'
    finally:
        torch.set_float32_matmul_precision(before)
