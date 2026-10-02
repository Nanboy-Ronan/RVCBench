"""GLM-TTS preparation context and local dependency provenance."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest
import torch

from rvcbench.benchmark.model_assets import resolve_model_assets
from rvcbench.models.glmtts.synthesizer import GLMTTSSynthesizer, GLMTTSSynthesizerConfig


@pytest.mark.parametrize('fail', [False, True])
def test_preparation_uses_upstream_working_directory_and_restores_it(tmp_path, fail):
    generator = object.__new__(GLMTTSSynthesizer)
    generator.config = GLMTTSSynthesizerConfig(code_path=tmp_path)
    generator.device = 'cpu'
    generator._llm = generator._flow = None
    original = Path.cwd()
    def load():
        assert Path.cwd() == tmp_path
        if fail:
            raise RuntimeError('fixture failure')
    with patch.object(generator, '_load_model_in_code_path', side_effect=load):
        if fail:
            with pytest.raises(RuntimeError):
                generator.load_model()
        else:
            generator.load_model()
    assert Path.cwd() == original


def test_upstream_default_cuda_calls_use_configured_device_context(tmp_path):
    generator = object.__new__(GLMTTSSynthesizer)
    generator.config = GLMTTSSynthesizerConfig(code_path=tmp_path)
    generator.device = 'cuda:2'
    entered = []
    @contextmanager
    def device_context(device):
        entered.append(str(device))
        try:
            yield
        finally:
            entered.append('restored')
    with patch('torch.cuda.device', side_effect=device_context):
        with generator._in_code_path():
            assert entered == ['cuda:2']
    assert entered == ['cuda:2', 'restored']


def test_glm_fingerprint_tracks_checkpoint_frontend_and_text_rules(tmp_path):
    for name in ('ckpt', 'frontend', 'configs'):
        (tmp_path / name).mkdir()
    checkpoint = tmp_path / 'ckpt' / 'model.safetensors'
    checkpoint.write_bytes(b'weights')
    frontend = tmp_path / 'frontend' / 'campplus.onnx'
    frontend.write_bytes(b'speaker-encoder')
    rules = tmp_path / 'configs' / 'custom_replace.jsonl'
    rules.write_text('{"text":"first"}\n')
    conf = OmegaConf.create({'vc': {'model': 'glm_tts'}, 'adversary': {
        'code_path': str(tmp_path), 'ckpt_dir': str(tmp_path / 'ckpt'),
        'frontend_dir': str(tmp_path / 'frontend')}})
    _, reference, first = resolve_model_assets(conf)
    assert set(reference['assets']) == {'ckpt_dir', 'frontend_dir', 'glmtts.configs'}
    assert not reference['unresolved_references']
    for path in (checkpoint, frontend, rules):
        before = resolve_model_assets(conf)[2]
        path.write_bytes(path.read_bytes() + b'changed')
        assert resolve_model_assets(conf)[2] != before
