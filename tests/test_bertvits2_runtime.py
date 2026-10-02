import json
from dataclasses import replace
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from rvcbench.adversary.bertvit2_ots import BertVits2ZeroShotAdversary
from rvcbench.datasets.zero_shot import ZeroShotSample


def adapter(tmp_path, speakers=850):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'data': {'n_speakers': speakers}}))
    return BertVits2ZeroShotAdversary(OmegaConf.create({
        'code_path': str(tmp_path), 'config_path': str(config),
        'generator_checkpoint': str(tmp_path / 'weights.pth')}),
        OmegaConf.create({}), torch.device('cpu'), logging.getLogger(__name__))


def test_closed_set_bert_fails_before_native_import_or_global_mutation(tmp_path, monkeypatch):
    model = adapter(tmp_path)
    before = (Path.cwd(), sys.argv[:], sys.path[:])
    monkeypatch.setattr('rvcbench.adversary.bertvit2_ots.importlib.import_module',
                        lambda *args: pytest.fail('native import must not occur'))
    with pytest.raises(ValueError, match='closed-set TTS'):
        model.prepare()
    assert (Path.cwd(), sys.argv, sys.path) == before
    assert model._net_g is None


def test_bert_generator_checkpoint_changes_asset_fingerprint(tmp_path):
    from rvcbench.benchmark.model_assets import resolve_model_assets
    path = tmp_path / 'G_0.pth'
    path.write_bytes(b'first checkpoint')
    conf = OmegaConf.create({'vc': {'model': 'bertvits2'},
                            'adversary': {'generator_checkpoint': str(path)}})
    _, reference, first = resolve_model_assets(conf)
    assert 'generator_checkpoint' in reference['assets']
    path.write_bytes(b'second checkpoint')
    assert resolve_model_assets(conf)[2] != first


def test_bert_synthesizes_target_text_and_refuses_reference_fallback(tmp_path):
    model = adapter(tmp_path)
    model._imports_loaded = True
    model._hps = SimpleNamespace(data=SimpleNamespace(sampling_rate=24000))
    model._load_reference_audio = lambda path: np.zeros(240)
    observed = []
    model._infer_fn = lambda text, **kwargs: observed.append(text) or np.zeros(240)
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'fixture')
    sample = ZeroShotSample(index=0, speaker_id='121', prompt_path=reference,
                           target_path=None, prompt_text='Reference sentence.', target_text='Target sentence.',
                           prompt_language='EN', target_language='EN', extra={})
    dataset = SimpleNamespace(get_zero_shot_samples=lambda **kwargs: [sample])
    model.attack(output_path=tmp_path / 'output', dataset=dataset)
    assert observed == ['Target sentence.']
    sample = replace(sample, target_text='')
    with pytest.raises(ValueError, match='nonempty target text'):
        model.attack(output_path=tmp_path / 'empty', dataset=dataset)
    assert observed == ['Target sentence.']
