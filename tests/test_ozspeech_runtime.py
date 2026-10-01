"""OZSpeech load isolation, actual device placement and codec asset provenance."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf, DictConfig
import pytest
import torch

from src.models.ozspeech.synthesizer import OzSpeechSynthesizer
from src.benchmark.model_assets import resolve_model_assets


@pytest.mark.parametrize('fail', [False, True])
def test_checkpoint_allowlist_restores_caller_state_and_places_resources(tmp_path, fail):
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'fixture')
    generator = OzSpeechSynthesizer(config, checkpoint, 'cpu', .01, logging.getLogger())
    resources = [torch.nn.Linear(2, 2) for _ in range(3)]
    for resource in resources:
        resource.to = Mock(wraps=resource.to)
    def load(**kwargs):
        assert DictConfig in torch.serialization.get_safe_globals()
        if fail:
            raise RuntimeError('fixture failure')
        return resources[0]
    generator._zact_class = SimpleNamespace(from_pretrained=load)
    original = torch.serialization.get_safe_globals()
    torch.serialization.add_safe_globals([bool])
    try:
        before = set(torch.serialization.get_safe_globals())
        with patch.object(generator, '_ensure_dependencies'), \
                patch.object(generator, '_load_codec_models', return_value=resources[1:]):
            if fail:
                with pytest.raises(RuntimeError, match='fixture failure'):
                    generator.load_model()
            else:
                generator.load_model()
                for resource in resources:
                    resource.to.assert_called_once_with('cpu')
                    assert not resource.training
        assert set(torch.serialization.get_safe_globals()) == before
        generator.close()
        assert generator._model is generator._codec_encoder is generator._codec_decoder is None
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(original)


def test_codec_and_lexicon_assets_change_model_fingerprint(tmp_path):
    lexicon = tmp_path / 'zact/lexicon/librispeech-lexicon.txt'
    lexicon.parent.mkdir(parents=True)
    lexicon.write_text('hello HH AH L OW\n')
    encoder, decoder = tmp_path / 'encoder.bin', tmp_path / 'decoder.bin'
    encoder.write_bytes(b'encoder'); decoder.write_bytes(b'decoder')
    conf = OmegaConf.create({'vc': {'model': 'ozspeech'}, 'adversary': {
        'code_path': str(tmp_path), 'codec_encoder_path': str(encoder), 'codec_decoder_path': str(decoder)}})
    _, reference, first = resolve_model_assets(conf)
    assert set(reference['assets']) == {'codec_encoder_path', 'codec_decoder_path', 'ozspeech.lexicon'}
    for path in (encoder, decoder, lexicon):
        original = path.read_bytes()
        path.write_bytes(b'changed')
        assert resolve_model_assets(conf)[2] != first
        path.write_bytes(original)


def test_default_codec_downloads_share_a_pinned_revision(tmp_path):
    def download(**kwargs):
        path = tmp_path / kwargs['filename']
        path.write_bytes(kwargs['filename'].encode())
        return str(path)
    api = Mock(); api.model_info.return_value.sha = 'immutable-revision'
    fetch = Mock(side_effect=download)
    hub = SimpleNamespace(HfApi=Mock(return_value=api), hf_hub_download=fetch,
                          constants=SimpleNamespace(HF_HUB_OFFLINE=False))
    conf = OmegaConf.create({'vc': {'model': 'ozspeech'}, 'adversary': {}})
    with patch.dict('sys.modules', {'huggingface_hub': hub}):
        resolved, reference, _ = resolve_model_assets(conf)
    assert fetch.call_count == 2
    assert all(call.kwargs['revision'] == 'immutable-revision' for call in fetch.call_args_list)
    assert resolved.adversary.codec_encoder_path.endswith('ns3_facodec_encoder.bin')
    assert reference['assets']['ozspeech.codec_hub']['revision'] == 'immutable-revision'


def test_codec_alias_is_used_when_primary_yaml_field_is_null(tmp_path):
    from src.adversary.ozspeech_ots import OzSpeechZeroShotAdversary
    config = OmegaConf.create({'code_path': str(tmp_path), 'checkpoint_path': str(tmp_path / 'model.pt'),
        'config_path': str(tmp_path / 'config.yaml'), 'codec_encoder_path': None,
        'facodec_encoder_path': str(tmp_path / 'encoder.bin'), 'codec_decoder_path': None,
        'facodec_decoder_path': str(tmp_path / 'decoder.bin')})
    adapter = OzSpeechZeroShotAdversary(config, {}, 'cpu', logging.getLogger())
    assert adapter.codec_encoder_path == tmp_path / 'encoder.bin'
    assert adapter.codec_decoder_path == tmp_path / 'decoder.bin'
