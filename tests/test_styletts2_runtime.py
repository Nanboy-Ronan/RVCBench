"""StyleTTS2 checkpoint loading scope and implicit local asset coverage."""
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
import pytest
import torch

from src.benchmark.model_assets import resolve_model_assets
from src.models.styletts2.synthesizer import legacy_checkpoint_loading


@pytest.mark.parametrize('fail', [False, True])
def test_legacy_loading_restores_torch_loader_on_success_and_failure(fail):
    loader = Mock(return_value={})
    with patch('torch.load', loader):
        try:
            with legacy_checkpoint_loading():
                torch.load('trusted-upstream.pth')
                torch.load('explicit-safe.pth', weights_only=True)
                if fail:
                    raise RuntimeError('fixture loading failed')
        except RuntimeError:
            assert fail
        assert torch.load is loader
        assert loader.call_args_list[0].kwargs['weights_only'] is False
        assert loader.call_args_list[1].kwargs['weights_only'] is True


def test_style_model_fingerprint_tracks_config_referenced_weights(tmp_path):
    config = tmp_path / 'config.yml'
    plbert = tmp_path / 'Utils' / 'PLBERT'
    plbert.mkdir(parents=True)
    (tmp_path / 'asr.yml').write_text('hidden: 1\n')
    (tmp_path / 'asr.pth').write_bytes(b'asr')
    (tmp_path / 'f0.t7').write_bytes(b'f0')
    weight = plbert / 'step_1000000.t7'
    weight.write_bytes(b'plbert-v1')
    config.write_text('ASR_config: asr.yml\nASR_path: asr.pth\nF0_path: f0.t7\nPLBERT_dir: Utils/PLBERT\n')
    conf = OmegaConf.create({'vc': {'model': 'styletts2'}, 'adversary': {
        'code_path': str(tmp_path), 'config_path': str(config)}})
    _, reference, first = resolve_model_assets(conf)
    assert not reference['unresolved_references']
    assert 'step_1000000.t7' in reference['assets']['styletts2.PLBERT_dir']['files']
    weight.write_bytes(b'plbert-v2')
    assert first != resolve_model_assets(conf)[2]
    weight.unlink()
    assert 'styletts2.PLBERT_dir' in resolve_model_assets(conf)[1]['unresolved_references']
