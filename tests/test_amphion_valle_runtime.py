import json
from omegaconf import OmegaConf
import pytest

from rvcbench.models.valle.amphion import read_config
from rvcbench.benchmark.model_assets import resolve_model_assets


def merge(base, override):
    return OmegaConf.to_container(OmegaConf.merge(base, override))


def test_base_configs_resolve_without_changing_cwd_or_environment(tmp_path, monkeypatch):
    import os
    root = tmp_path / 'runtime'
    (root / 'config').mkdir(parents=True)
    (root / 'config/base.json').write_text('{"model": {"dim": 1024, "heads": 16}}')
    config = tmp_path / 'release.json'
    config.write_text('{"base_config": "config/base.json", "model": {"heads": 12}}')
    monkeypatch.setenv('WORK_DIR', '/unrelated/runtime')
    before = os.getcwd()
    result = read_config(config, root, merge)
    assert result['model'] == {'dim': 1024, 'heads': 12}
    assert os.getcwd() == before and os.environ['WORK_DIR'] == '/unrelated/runtime'


def test_cyclic_config_is_rejected(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{"base_config": "config.json"}')
    with pytest.raises(ValueError, match='Cyclic'):
        read_config(config, tmp_path, merge)


def test_base_configuration_and_symbols_affect_assets(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path))
    (tmp_path / 'config').mkdir()
    base = tmp_path / 'config/base.json'
    base.write_text('{}')
    symbols = tmp_path / 'symbols.dict'
    symbols.write_text('a 1')
    conf = OmegaConf.create({'vc': {'model': 'vall_e'}, 'adversary': {
        'implementation': 'amphion', 'code_path': str(tmp_path), 'text_tokens_path': str(symbols)}})
    before = resolve_model_assets(conf)[2]
    base.write_text('{"changed": true}')
    assert resolve_model_assets(conf)[2] != before
    before = resolve_model_assets(conf)[2]
    symbols.write_text('a 2')
    assert resolve_model_assets(conf)[2] != before
    cache = tmp_path / 'checkpoints'
    cache.mkdir()
    codec = cache / 'encodec_24khz-d7cc33bc.th'
    codec.write_bytes(b'codec')
    before = resolve_model_assets(conf)[2]
    codec.write_bytes(b'changed codec')
    assert resolve_model_assets(conf)[2] != before
