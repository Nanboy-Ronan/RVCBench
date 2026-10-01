"""PlayDiffusion preset selection, one-time setup and immutable asset provenance."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
import pytest

from src.models.playdiffusion.generator import (
    PlayDiffusionGenerator, PlayDiffusionGeneratorConfig, PRESET_FILES,
)
from src.benchmark.model_assets import resolve_model_assets


def make_preset(root):
    for field in PRESET_FILES:
        (root / getattr(PlayDiffusionGeneratorConfig, field)).write_bytes(field.encode())


def test_local_preset_is_used_for_the_only_manager_initialization(tmp_path):
    make_preset(tmp_path)
    configurations = []
    class Engine:
        def __init__(self, device):
            self.device = device
            self.preset = self.load_preset()
            configurations.append(self.preset)
            self.preset['speech_tokenizer'].pop('sample_rate')  # upstream mutation
            self.mm = object()
        def load_preset(self):
            raise AssertionError('Local preset must avoid upstream default downloads')
    config = PlayDiffusionGeneratorConfig(code_path=tmp_path, preset_dir=tmp_path)
    generator = PlayDiffusionGenerator(config, 'cpu', logging.getLogger())
    modules = {'playdiffusion.inference': SimpleNamespace(PlayDiffusion=Engine),
               'playdiffusion.pydantic_models.models': SimpleNamespace(TTSInput=object)}
    with patch.dict('sys.modules', modules), patch.object(generator, '_ensure_imports'), \
            patch.dict('sys.modules', {'huggingface_hub': None}):
        generator.load_model()
    assert len(configurations) == 1
    assert configurations[0]['vocoder']['checkpoint'] == str(tmp_path / config.vocoder_checkpoint)
    assert generator._build_local_preset()['speech_tokenizer']['sample_rate'] == 16000
    generator.close()
    assert generator._engine is None and not generator._model_ready


@pytest.mark.parametrize('field', PRESET_FILES)
def test_every_preset_asset_changes_model_fingerprint(tmp_path, field):
    make_preset(tmp_path)
    conf = OmegaConf.create({'vc': {'model': 'playdiffusion'}, 'adversary': {'preset_dir': str(tmp_path)}})
    _, reference, first = resolve_model_assets(conf)
    assert len(reference['assets']) == 6
    file = tmp_path / getattr(PlayDiffusionGeneratorConfig, field)
    file.write_bytes(b'changed')
    assert resolve_model_assets(conf)[2] != first


def test_hub_preset_is_pinned_before_snapshot_download(tmp_path):
    make_preset(tmp_path)
    api = Mock()
    api.model_info.return_value.sha = 'immutable-revision'
    download = Mock(return_value=str(tmp_path))
    hub = SimpleNamespace(HfApi=Mock(return_value=api), snapshot_download=download,
                          constants=SimpleNamespace(HF_HUB_OFFLINE=False))
    conf = OmegaConf.create({'vc': {'model': 'playdiffusion'}, 'adversary': {'cache_dir': 'chosen-cache'}})
    with patch.dict('sys.modules', {'huggingface_hub': hub}):
        resolved, reference, _ = resolve_model_assets(conf)
    assert resolved.adversary.preset_dir == str(tmp_path)
    assert resolved.adversary.hf_revision == 'immutable-revision'
    assert download.call_args.kwargs['revision'] == 'immutable-revision'
    assert download.call_args.kwargs['cache_dir'] == 'chosen-cache'
    assert len(download.call_args.kwargs['allow_patterns']) == 6
    assert reference['assets']['playdiffusion.hub']['revision'] == 'immutable-revision'
