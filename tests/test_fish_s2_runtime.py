import json
import queue
from types import SimpleNamespace

import pytest
import torch

from src.adversary.fishspeech_s2_ots import SerialSemanticQueue, load_shards, load_codec_state


def test_s2_shards_must_match_index_without_duplicate_or_escaping_keys(tmp_path):
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'a': '1.safetensors', 'b': '2.safetensors'}}))
    tensors = {'1.safetensors': {'a': 1}, '2.safetensors': {'b': 2}}
    loader = lambda path, **kwargs: tensors[path.rsplit('/', 1)[-1]]
    assert load_shards(tmp_path, loader) == {'a': 1, 'b': 2}
    tensors['2.safetensors']['a'] = 3
    with pytest.raises(ValueError, match='duplicate tensor'):
        load_shards(tmp_path, loader)
    tensors['2.safetensors'] = {'unexpected': 3}
    with pytest.raises(ValueError, match='differ from its shard index'):
        load_shards(tmp_path, loader)
    index.write_text(json.dumps({'weight_map': {'a': '../escaped.safetensors'}}))
    with pytest.raises(ValueError, match='escapes'):
        load_shards(tmp_path, loader)


def test_serial_s2_propagates_native_failure_without_waiting_for_worker():
    observed = []
    def generate(**kwargs):
        observed.append(kwargs)
        yield 'first'
        raise RuntimeError('native failure')
    native = SimpleNamespace(generate_long=generate, WrappedGenerateResponse=lambda **kwargs: kwargs)
    model, decode = object(), object()
    semantic = SerialSemanticQueue(model, decode, native)
    responses = queue.Queue()
    with pytest.raises(RuntimeError, match='native failure'):
        semantic.put(SimpleNamespace(request={'text': 'Actual target.'}, response_queue=responses))
    assert responses.get_nowait() == {'status': 'success', 'response': 'first'}
    assert observed == [{'model': model, 'decode_one_token': decode, 'text': 'Actual target.'}]
    semantic.put(None)


def test_codec_allows_only_exact_rebuilt_nonpersistent_buffers():
    codec = torch.nn.Linear(2, 2)
    codec.register_buffer('freqs_cis', torch.arange(8), persistent=False)
    state = dict(codec.state_dict(), freqs_cis=torch.arange(4))
    load_codec_state(codec, state)
    state['freqs_cis'] = torch.ones(4, dtype=torch.int64)
    with pytest.raises(ValueError, match='differs from native reconstruction'):
        load_codec_state(codec, state)
    state = dict(codec.state_dict(), extra_weight=torch.ones(1))
    with pytest.raises(ValueError, match='Unexpected S2 codec weight'):
        load_codec_state(codec, state)
    state = codec.state_dict()
    del state['weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        load_codec_state(codec, state)
