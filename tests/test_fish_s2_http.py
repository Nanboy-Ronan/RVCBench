import logging
from types import SimpleNamespace

import ormsgpack
from omegaconf import OmegaConf
import pytest
import torch

from src.adversary.fish_audio_s2_server_ots import FishAudioS2ServerZeroShotAdversary


def adapter(**kwargs):
    config = OmegaConf.create({'endpoint_url': 'http://127.0.0.1:18011/v1/tts',
                              'seed': 42, 'request_attempts': 3, 'retry_delay_sec': 0, **kwargs})
    return FishAudioS2ServerZeroShotAdversary(config, OmegaConf.create({}),
                                            torch.device('cpu'), logging.getLogger(__name__))


def test_http_s2_readiness_failure_prevents_generation(monkeypatch):
    observed = []
    def get(url, **kwargs):
        observed.append(url)
        return SimpleNamespace(status_code=404)
    monkeypatch.setattr('src.adversary.fish_audio_s2_server_ots.requests.get', get)
    with pytest.raises(RuntimeError, match='readiness check failed'):
        adapter().prepare()
    assert observed == ['http://127.0.0.1:18011/v1/health']


def test_http_s2_sends_original_index_seed_and_exact_reference_bytes(tmp_path):
    path = tmp_path / 'reference.wav'
    path.write_bytes(b'actual reference bytes')
    model = adapter()
    payload = ormsgpack.unpackb(model._build_payload('Target.', path, 'Reference.', 89))
    assert payload['seed'] == 131
    assert payload['text'] == 'Target.'
    assert payload['references'] == [{'audio': path.read_bytes(), 'text': 'Reference.'}]
    for text, reference in [('', 'Reference.'), ('Target.', '')]:
        with pytest.raises(ValueError, match='actual target and reference'):
            model._build_payload(text, path, reference, 89)


@pytest.mark.parametrize('status,content,message', [
    (400, b'invalid', 'request rejected'),
    (200, b'<html>wrong service</html>', 'non-WAV response'),
])
def test_http_s2_permanent_errors_do_not_retry(monkeypatch, status, content, message):
    observed = []
    def post(*args, **kwargs):
        observed.append(kwargs['data'])
        return SimpleNamespace(status_code=status, content=content, text='invalid request')
    monkeypatch.setattr('src.adversary.fish_audio_s2_server_ots.requests.post', post)
    with pytest.raises(RuntimeError, match=message):
        adapter()._post_generation(b'exact payload')
    assert observed == [b'exact payload']


def test_http_s2_transient_retry_keeps_identical_payload(monkeypatch):
    observed = []
    wav = b'RIFF' + b'1234' + b'WAVE' + b'fixture data'
    responses = iter([SimpleNamespace(status_code=503, text='temporary'),
                      SimpleNamespace(status_code=200, content=wav)])
    def post(*args, **kwargs):
        observed.append(kwargs['data'])
        return next(responses)
    monkeypatch.setattr('src.adversary.fish_audio_s2_server_ots.requests.post', post)
    assert adapter()._post_generation(b'exact payload') == wav
    assert observed == [b'exact payload', b'exact payload']
