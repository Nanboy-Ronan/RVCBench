import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rvcbench.models.valle import VallEGenerator, VallEGeneratorConfig


def wrapper(tmp_path, codes=None, audio=None):
    w = VallEGenerator(VallEGeneratorConfig(tmp_path, tmp_path / 'weights.pt'),
                      torch.device('cpu'), logging.getLogger(__name__))
    calls = []
    def inference(*args, **kwargs):
        calls.append((args, kwargs))
        return codes if codes is not None else torch.ones((1, 3, 8), dtype=torch.long)
    w._model = SimpleNamespace(inference=inference)
    w._audio_tokenizer = SimpleNamespace(decode=lambda frames:
        audio if audio is not None else torch.ones(1, 1, 100))
    w._text_tokenizer = object()
    w._text_collater = lambda texts: (torch.ones(1, len(texts[0]), dtype=torch.long),
                                     torch.tensor([len(texts[0])]))
    texts = []
    def tokenize(tokenizer, *, text):
        texts.append(text)
        return list(text)
    w._data = SimpleNamespace(tokenize_text=tokenize,
        tokenize_audio=lambda *args: [(torch.ones(1, 8, 4), None)])
    w.ensure_model = lambda: None
    prompt = tmp_path / 'prompt.wav'
    prompt.write_bytes(b'fixture')
    return w, prompt, calls, texts


def test_reference_and_target_remain_distinct_and_codec_shape_is_native(tmp_path):
    w, prompt, calls, texts = wrapper(tmp_path)
    audio, rate = w.generate(text='Target.', prompt_audio=prompt, prompt_text='Reference.')
    assert texts == ['Reference. Target.', 'Reference.']
    assert calls[0][0][2].shape == (1, 4, 8)
    assert calls[0][1]['enroll_x_lens'].item() == len('Reference.')
    assert audio.shape == (100,) and rate == 24000
    w.close()
    assert w._model is w.model is w._audio_tokenizer is None and not w.is_model_ready()


@pytest.mark.parametrize('codes', [torch.ones(1, 0, 8), torch.ones(2, 3, 8), torch.ones(3, 8)])
def test_empty_or_multiple_codec_outputs_are_not_padded_into_success(tmp_path, codes):
    w, prompt, _, _ = wrapper(tmp_path, codes=codes)
    with pytest.raises(ValueError, match='codec tokens'):
        w.generate(text='Target.', prompt_audio=prompt, prompt_text='Reference.')


@pytest.mark.parametrize('audio', [torch.full((1, 1, 10), float('nan')),
                                 torch.ones(2, 1, 10), torch.ones(1, 1, 0)])
def test_invalid_waveform_fails(tmp_path, audio):
    w, prompt, _, _ = wrapper(tmp_path, audio=audio)
    with pytest.raises(ValueError, match='waveform'):
        w.generate(text='Target.', prompt_audio=prompt, prompt_text='Reference.')


def test_missing_checkpoint_is_actionable_before_native_import(tmp_path):
    w = VallEGenerator(VallEGeneratorConfig(tmp_path, tmp_path / 'missing.pt'),
                      torch.device('cpu'), logging.getLogger(__name__))
    with pytest.raises(FileNotFoundError, match='not compatible replacements'):
        w.ensure_model()


def test_checkpoint_shape_mismatch_fails_strictly_and_cleans_state(tmp_path, monkeypatch):
    import sys
    import rvcbench.models.valle.generator as runtime
    weights, tokens = tmp_path / 'weights.pt', tmp_path / 'tokens.txt'
    weights.write_bytes(b'fixture')
    tokens.write_text('fixture')
    w = VallEGenerator(VallEGeneratorConfig(tmp_path, weights),
                      torch.device('cpu'), logging.getLogger(__name__))
    data = SimpleNamespace(__file__=str(tmp_path / 'data.py'))
    model = torch.nn.Linear(2, 2)
    models = SimpleNamespace(__file__=str(tmp_path / 'models.py'), get_model=lambda args: model)
    monkeypatch.setitem(sys.modules, 'icefall.utils',
                        SimpleNamespace(AttributeDict=lambda value: SimpleNamespace(**value)))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs:
        {'model': {'weight': torch.ones(3, 3)}, 'text_tokens': str(tokens)})
    monkeypatch.setattr(runtime.importlib, 'import_module',
        lambda name: data if name == 'valle.data' else models)
    with pytest.raises(RuntimeError, match='size mismatch'):
        w.ensure_model()
    assert w._model is w.model is w._audio_tokenizer is None
    assert not w.is_model_ready()
