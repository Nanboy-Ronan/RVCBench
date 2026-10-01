"""MGM checkpoint dispatch and restoration of upstream global hooks."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from src.benchmark.model_assets import resolve_model_assets
from src.models.mgm_omni.generator import MGMOmniGenerator, MGMOmniGeneratorConfig


@pytest.mark.parametrize('fail', [False, True])
def test_upstream_hooks_restore_after_success_or_failure(tmp_path, fail):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'MGMTTS'}))
    generator = object.__new__(MGMOmniGenerator)
    generator.config = MGMOmniGeneratorConfig(repo_root=str(tmp_path), checkpoint_path=str(tmp_path))
    generator.device = 'cpu'
    original_initializer = lambda *a: 'original'
    upstream_initializer = lambda *a: 'upstream'
    mixin = type('Mixin', (), {'_maybe_initialize_input_ids_for_generation': original_initializer})
    original_name = lambda path: 'revision-hash'
    builder = SimpleNamespace(get_model_name_from_path=original_name,
                              initialize_input_ids_for_generation=upstream_initializer)
    linear, norm = torch.nn.Linear.reset_parameters, torch.nn.LayerNorm.reset_parameters
    def disable():
        torch.nn.Linear.reset_parameters = lambda self: None
        torch.nn.LayerNorm.reset_parameters = lambda self: None
    modules = {'transformers': SimpleNamespace(GenerationMixin=mixin),
               'mgm.model': SimpleNamespace(builder=builder),
               'mgm.utils': SimpleNamespace(disable_torch_init=disable)}
    with patch.dict('sys.modules', modules):
        try:
            with generator._upstream_runtime(preparing=True):
                assert mixin._maybe_initialize_input_ids_for_generation is upstream_initializer
                assert torch.nn.Linear.reset_parameters is not linear
                assert builder.get_model_name_from_path(str(tmp_path)) == 'MGM-Omni-TTS'
                assert builder.get_model_name_from_path('other') == 'revision-hash'
                if fail:
                    raise RuntimeError('fixture failure')
        except RuntimeError:
            assert fail
    assert mixin._maybe_initialize_input_ids_for_generation is original_initializer
    assert builder.get_model_name_from_path is original_name
    assert torch.nn.Linear.reset_parameters is linear
    assert torch.nn.LayerNorm.reset_parameters is norm
    # Subsequent upstream generation must re-enter its compatibility hook without
    # disabling initialization again after an already-imported builder.
    with patch.dict('sys.modules', modules), generator._upstream_runtime():
        assert mixin._maybe_initialize_input_ids_for_generation is upstream_initializer
        assert torch.nn.Linear.reset_parameters is linear
    assert mixin._maybe_initialize_input_ids_for_generation is original_initializer


def test_auxiliary_cosyvoice_weights_change_model_fingerprint(tmp_path):
    codec = tmp_path / 'cosyvoice'
    codec.mkdir()
    weights = codec / 'flow.pt'
    weights.write_bytes(b'first')
    config = OmegaConf.create({'vc': {'model': 'mgm_omni'}, 'adversary': {'cosyvoice_path': str(codec)}})
    first = resolve_model_assets(config)[2]
    weights.write_bytes(b'second')
    assert resolve_model_assets(config)[2] != first


def test_close_releases_main_model_and_fallback_asr():
    generator = object.__new__(MGMOmniGenerator)
    generator.model = generator._model = generator._tokenizer = generator._whisper_model = object()
    generator._model_ready = True
    generator.close()
    assert generator.model is generator._model is generator._tokenizer is generator._whisper_model is None
    assert not generator._model_ready
