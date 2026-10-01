"""dots.tts device/global-state scopes and package provenance."""
from contextlib import contextmanager
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.models.dots_tts.generator import DotsTTSGenerator, DotsTTSGeneratorConfig
from src.benchmark.model_assets import resolve_model_assets
from src.benchmark.fingerprints import generation_runtime


@pytest.mark.parametrize('fail', [False, True])
def test_cuda_scope_and_runtime_precision_restore_after_load_and_generation(fail):
    generator = DotsTTSGenerator(DotsTTSGeneratorConfig(), 'cuda:2', logging.getLogger())
    entered = []
    @contextmanager
    def device_context(device):
        entered.append(str(device))
        try:
            yield
        finally:
            entered.pop()
    precision = torch.get_float32_matmul_precision()
    def load(*args, **kwargs):
        assert entered == ['cuda:2']
        torch.set_float32_matmul_precision('high')
        if fail:
            raise RuntimeError('load fixture')
        return SimpleNamespace(model=None, generate=generate)
    def generate(**kwargs):
        assert entered == ['cuda:2']
        assert torch.get_float32_matmul_precision() == 'high'
        return {'audio': np.zeros(10), 'sample_rate': 48000}
    module = SimpleNamespace(DotsTtsRuntime=SimpleNamespace(from_pretrained=load))
    with patch('torch.cuda.device', side_effect=device_context), \
            patch('importlib.import_module', return_value=module):
        if fail:
            with pytest.raises(RuntimeError, match='load fixture'):
                generator.ensure_model()
        else:
            generator.ensure_model()
            assert torch.get_float32_matmul_precision() == precision
            assert generator.generate('text', 'reference.wav', 'reference', 1)[1] == 48000
    assert entered == []
    assert torch.get_float32_matmul_precision() == precision
    generator.close()
    assert generator._runtime is None and not generator._model_ready


def test_explicit_cpu_request_rejected_if_runtime_would_select_cuda():
    generator = DotsTTSGenerator(DotsTTSGeneratorConfig(), 'cpu', logging.getLogger())
    with patch('torch.cuda.is_available', return_value=True):
        with pytest.raises(ValueError, match='CPU request'):
            with generator._runtime_context():
                raise AssertionError('Must not silently load on CUDA')


def test_configured_source_rejects_cached_foreign_runtime(tmp_path):
    generator = DotsTTSGenerator(DotsTTSGeneratorConfig(code_path=tmp_path), 'cuda:0', logging.getLogger())
    loader = Mock()
    module = SimpleNamespace(__file__='/foreign/dots_tts/runtime.py', DotsTtsRuntime=SimpleNamespace(from_pretrained=loader))
    with patch('importlib.import_module', return_value=module):
        with pytest.raises(RuntimeError, match='outside'):
            generator.load_model()
    loader.assert_not_called()


def test_package_source_and_chat_template_changes_are_fingerprinted(tmp_path):
    source = tmp_path / 'dots_tts'
    source.mkdir()
    runtime = source / 'runtime.py'
    runtime.write_text('VERSION = 1\n')
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    template = checkpoint / 'chat_template.jinja'
    template.write_text('first')
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'dots_tts'},
        'adversary': {'code_path': str(source), 'checkpoint': str(checkpoint)}})
    first = generation_runtime(tmp_path, conf, {})['source_sha256']
    runtime.write_text('VERSION = 2\n')
    assert generation_runtime(tmp_path, conf, {})['source_sha256'] != first
    _, reference, fingerprint = resolve_model_assets(conf)
    assert 'chat_template.jinja' in reference['assets']['checkpoint']['files']
    template.write_text('second')
    assert resolve_model_assets(conf)[2] != fingerprint
