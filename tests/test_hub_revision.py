from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from src.benchmark.model_assets import resolve_hub_revision, resolve_model_assets


PIN = 'fd4b254389122332181a7c3db7f27e918eec64e3'


def test_pinned_qwen_assets_resolve_offline_without_metadata_request(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    conf = OmegaConf.create({'vc': {'model': 'qwen3_tts'},
        'adversary': {'checkpoint_path': 'Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'revision': PIN}})
    with patch('huggingface_hub.constants.HF_HUB_OFFLINE', True), \
            patch('huggingface_hub.HfApi', side_effect=AssertionError('network forbidden')), \
            patch('huggingface_hub.snapshot_download', return_value=str(tmp_path)) as snapshot:
        resolved, reference, _ = resolve_model_assets(conf)
    assert resolved.adversary.revision == PIN
    assert resolved.adversary.checkpoint_path == str(tmp_path)
    snapshot.assert_called_once()
    assert snapshot.call_args.kwargs['revision'] == PIN
    assert reference['assets']['qwen3_tts.hub'] == {
        'repo_id': conf.adversary.checkpoint_path, 'revision': PIN}
    assert set(reference['assets']['checkpoint_path']['files']) == {'config.json', 'model.safetensors'}
    (tmp_path / 'model.safetensors').write_bytes(b'changed')
    with patch('huggingface_hub.snapshot_download', return_value=str(tmp_path)):
        assert reference != resolve_model_assets(conf)[1]


@pytest.mark.parametrize('revision', [None, 'main', 'v1.0', PIN[:7], 'g' * 40])
def test_offline_mutable_or_incomplete_revision_requires_pin(revision):
    with patch('huggingface_hub.constants.HF_HUB_OFFLINE', True), \
            patch('huggingface_hub.HfApi', side_effect=AssertionError('network forbidden')):
        with pytest.raises(ValueError, match='full 40-character commit revision'):
            resolve_hub_revision('example/model', revision)


def test_online_mutable_revision_resolves_to_server_commit():
    with patch('huggingface_hub.constants.HF_HUB_OFFLINE', False), \
            patch('huggingface_hub.HfApi') as api:
        api.return_value.model_info.return_value.sha = PIN
        assert resolve_hub_revision('example/model', 'release') == PIN
        api.return_value.model_info.assert_called_once_with('example/model', revision='release')
