"""Auxiliary models must expose failures and fingerprint the weights they use."""
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from src.evaluation.scorers import ScoreInput
from src.evaluation.scorers.auxiliary import AuxiliaryScorer


def test_speechmos_asset_scope_and_loaded_weights(tmp_path, monkeypatch):
    hub = tmp_path / 'hub'
    root = hub / 'tarepan_SpeechMOS_main'
    root.mkdir(parents=True)
    source = root / 'model.py'
    source.write_text('# source fixture')
    (hub / 'checkpoints').mkdir()
    weights = hub / 'checkpoints/utmos22_strong_step7459_v1.pt'
    torch.save({'weight': torch.tensor(2.)}, weights)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))

    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(hub))
    monkeypatch.setattr('src.evaluation.scorers.auxiliary.importlib.import_module',
                        lambda _: SimpleNamespace(__file__=str(source), UTMOS22Strong=Model))
    scorer = AuxiliaryScorer('speechmos', 'cpu', logging.getLogger())
    scorer.prepare()
    baseline = scorer.model_provenance
    assert scorer.model.weight.item() == 2.
    scorer.close()
    (hub / 'checkpoints/unrelated.pt').write_bytes(b'unrelated model')
    scorer.prepare()
    assert scorer.model_provenance == baseline
    scorer.close()
    torch.save({'weight': torch.tensor(3.)}, weights)
    scorer.prepare()
    assert scorer.model.weight.item() == 3.
    assert scorer.model_provenance['weights'] != baseline['weights']
    scorer.close()


@pytest.mark.parametrize('metric', ['speechmos', 'dnsmos'])
def test_auxiliary_native_error_is_not_swallowed_or_retried(tmp_path, metric):
    wav = tmp_path / 'audio.wav'
    sf.write(wav, np.ones(1600) * .01, 24000)
    calls = []
    failure = RuntimeError('CUDA error: illegal memory access')
    def model(*args):
        calls.append(args)
        raise failure
    scorer = AuxiliaryScorer(metric, 'cpu', logging.getLogger())
    scorer.model = model
    with pytest.raises(RuntimeError) as raised:
        scorer.score(ScoreInput(wav, wav, 'Hello.', 'EN'))
    assert raised.value is failure and len(calls) == 1
