"""Regressions for sample identity, worker lifetime and seed propagation."""
import atexit
from dataclasses import replace
import json
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest

from test_benchmark import setup_run, mocked_evaluator, fake_evaluate
from src.benchmark.artifacts import coverage, input_records, input_fingerprint
from src.datasets.zero_shot import ZeroShotDataset


def test_input_fingerprint_includes_prompt_language_and_seed_index(setup_run):
    _, dataset, _, _ = setup_run
    sample = dataset.get_zero_shot_samples()[0]
    fingerprint = lambda s: input_fingerprint(input_records([s]))
    assert fingerprint(sample) != fingerprint(replace(sample, prompt_language='ZH'))
    assert fingerprint(sample) != fingerprint(replace(sample, index=51))


def test_speaker_filter_preserves_source_index(setup_run):
    conf, _, _, _ = setup_run
    conf.dataset.speaker_id = 'two'
    dataset = ZeroShotDataset(conf, conf.dataset, logging.getLogger())
    assert dataset.get_zero_shot_samples()[0].index == 2


def test_pending_samples_are_not_failures():
    rows = [{'status': s} for s in ('pending', 'generating', 'input_failed', 'generation_failed')]
    result = coverage(rows, ['sim'], evaluated=False)
    assert {k: result[k] for k in ('pending', 'generating', 'input_failed', 'generation_failed')} == {
        'pending': 1, 'generating': 1, 'input_failed': 1, 'generation_failed': 1}


@pytest.mark.parametrize('module_name,class_name', [
    ('index_tts', 'IndexTTSGenerator'), ('maskgct', 'MaskGCTGenerator')])
def test_real_worker_cleanup_reaps_owned_process(module_name, class_name):
    import importlib
    cls = getattr(importlib.import_module(f'src.models.{module_name}.generator'), class_name)
    generator = cls.__new__(cls)
    worker = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.readline()'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    generator._process, generator._stderr_handle = worker, None
    atexit.register(generator.close)
    try:
        generator.close()
        assert worker.poll() is not None
        generator.close()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        atexit.unregister(generator.close)


@pytest.mark.parametrize('interrupt', [False, True])
def test_runner_closes_resources_on_success_and_interrupt(setup_run, interrupt):
    from src.adversary.smoke import SmokeAdversary
    conf, _, run, _ = setup_run
    state = {'closed': False}
    class Owned(SmokeAdversary):
        def attack(self, **kwargs):
            if interrupt:
                raise KeyboardInterrupt()
            super().attack(**kwargs)
        def close(self):
            state['closed'] = True
    def evaluate(*args, **kwargs):
        assert state['closed'], 'Generation resources must close before scoring'
        return fake_evaluate(*args, **kwargs)
    conf.vc.generate_only = False
    with patch('src.benchmark.backends.select_adversary', return_value=Owned(conf, conf.dataset, 'cpu', logging.getLogger())), mocked_evaluator(evaluate):
        if interrupt:
            with pytest.raises(KeyboardInterrupt):
                run('interrupted-lifetime')
        else:
            run('closed-before-score')
    assert state['closed']


def test_moss_adapter_propagates_original_index(setup_run, tmp_path):
    from src.adversary.moss_ttsd_ots import MossTTSDZeroShotAdversary
    from src.benchmark.backends import SampleView
    conf, dataset, _, _ = setup_run
    config = OmegaConf.create({'code_path': '.', 'spt_config_path': '.', 'spt_checkpoint_path': '.', 'seed': 42})
    adapter = MossTTSDZeroShotAdversary(config, conf.dataset, 'cpu', logging.getLogger())
    indices = []
    def generate(**kwargs):
        indices.append(kwargs['sample_index'])
        return np.zeros(160), 16000
    adapter._generator = SimpleNamespace(generate=generate)
    sample = replace(dataset.get_zero_shot_samples()[0], index=73)
    adapter.attack(output_path=str(tmp_path / 'moss'), dataset=SampleView(dataset, sample))
    assert indices == [73]


def test_generation_fingerprint_excludes_evaluator_and_includes_worker(tmp_path):
    from src.benchmark.fingerprints import generation_runtime
    root = tmp_path
    (root / 'src/adversary').mkdir(parents=True)
    (root / 'src/evaluation').mkdir()
    (root / 'src/benchmark').mkdir()
    adapter = root / 'src/adversary/fixture.py'
    adapter.write_text('import numpy\n')
    evaluator = root / 'src/evaluation/pipeline.py'
    evaluator.write_text('VERSION = 1\n')
    (root / 'src/benchmark/runner.py').write_text('from src.evaluation.pipeline import evaluate_run\n')
    worker = root / 'worker.py'
    worker.write_text('VERSION = 1\n')
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'fixture'},
                            'adversary': {'worker_script_path': str(worker)}})
    with patch.dict('src.benchmark.fingerprints._ADVERSARY_REGISTRY', {'ots': {'fixture': 'src.adversary.fixture:Adapter'}}), patch('importlib.metadata.packages_distributions', return_value={'numpy': ['numpy']}), patch('importlib.metadata.requires', return_value=[]):
        first = generation_runtime(root, conf, {'numpy': '1.26.4', 'whisper': 'unused'})
        evaluator.write_text('VERSION = 2\n')
        assert generation_runtime(root, conf, {'numpy': '1.26.4', 'whisper': 'changed'}) == first
        worker.write_text('VERSION = 2\n')
        assert generation_runtime(root, conf, {'numpy': '1.26.4'})['source_sha256'] != first['source_sha256']


def test_worker_environment_probe_uses_actual_interpreter():
    from src.benchmark.fingerprints import worker_environment
    result = worker_environment(sys.executable)
    assert result['python'] and result['executable']
    assert result['packages']['pytest']
    assert set(result) == {'python', 'executable', 'packages'}


def test_model_fingerprint_tracks_converter_config_and_speaker_embedding(tmp_path):
    from src.benchmark.model_assets import resolve_model_assets
    config_path = tmp_path / 'converter.json'
    config_path.write_text('{"tau": 0.3}')
    speakers = tmp_path / 'speakers'
    speakers.mkdir()
    embedding = speakers / 'en-au.pth'
    embedding.write_bytes(b'first embedding')
    conf = OmegaConf.create({'vc': {'model': 'openvoice'}, 'adversary': {
        'converter_config_path': str(config_path), 'base_speaker_dir': str(speakers)}})
    _, reference, first = resolve_model_assets(conf)
    assert not reference['unresolved_references']
    embedding.write_bytes(b'changed embedding')
    _, _, second = resolve_model_assets(conf)
    assert second != first
    config_path.write_text('{"tau": 0.4}')
    assert resolve_model_assets(conf)[2] != second
    embedding.unlink()
    assert 'base_speaker_dir' in resolve_model_assets(conf)[1]['unresolved_references']


def test_generation_fingerprint_tracks_auxiliary_upstream_source(tmp_path):
    from src.benchmark.fingerprints import generation_runtime
    source = tmp_path / 'melo'
    source.mkdir()
    module = source / 'infer.py'
    module.write_text('TEMPERATURE = 1\n')
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'openvoice'},
                            'adversary': {'melo_code_path': str(source)}})
    first = generation_runtime(tmp_path, conf, {})
    module.write_text('TEMPERATURE = 2\n')
    assert generation_runtime(tmp_path, conf, {})['source_sha256'] != first['source_sha256']


def test_cosyvoice_explicit_matcha_source_rejects_cached_other_checkout(tmp_path):
    from src.models.cosyvoice.generator import CosyVoiceGenerator
    root = tmp_path / 'Matcha-TTS'
    (root / 'matcha').mkdir(parents=True)
    generator = CosyVoiceGenerator.__new__(CosyVoiceGenerator)
    generator.config = SimpleNamespace(matcha_code_path=root)
    generator.logger = logging.getLogger()
    with patch.object(sys, 'path', list(sys.path)), patch.dict(sys.modules,
            {'matcha': SimpleNamespace(__file__=str(root / 'matcha/__init__.py'))}):
        generator._ensure_matcha_dependency()
        assert sys.path[0] == str(root)
    with patch.object(sys, 'path', list(sys.path)), patch.dict(sys.modules,
            {'matcha': SimpleNamespace(__file__=str(tmp_path / 'other/matcha/__init__.py'))}):
        with pytest.raises(RuntimeError, match='different source'):
            generator._ensure_matcha_dependency()
    generator.config.matcha_code_path = tmp_path / 'missing'
    with pytest.raises(FileNotFoundError, match='Configured Matcha-TTS source'):
        generator._ensure_matcha_dependency()


def test_xtts_managed_lifetime_retains_generator_between_samples(setup_run, tmp_path):
    from src.adversary.xtts_ots import XttsZeroShotAdversary
    from src.benchmark.backends import SampleView
    conf, dataset, _, _ = setup_run
    adapter = XttsZeroShotAdversary(OmegaConf.create({'seed': 42}), conf.dataset,
                                   'cpu', logging.getLogger())
    calls = []
    generator = SimpleNamespace(generate=lambda **kwargs: (calls.append(kwargs) or np.zeros(160), 16000))
    adapter._generator = generator
    adapter.prepare()
    for sample in dataset.get_zero_shot_samples()[:2]:
        adapter.attack(output_path=str(tmp_path / 'xtts'), dataset=SampleView(dataset, sample))
        assert adapter._generator is generator
    assert len(calls) == 2
    adapter.close()
    assert adapter._generator is None
    assert not adapter._managed_lifetime


def test_resume_rejects_worker_environment_drift(setup_run):
    from src.benchmark import fingerprints
    conf, _, run, _ = setup_run
    original = fingerprints.generation_runtime
    def runtime(*args):
        result = original(*args)
        result['worker_environment'] = {'packages': {'fixture': version[0]}}
        return result
    version = ['1.0']
    with patch('src.benchmark.fingerprints.generation_runtime', side_effect=runtime):
        first, _, _ = run('worker-v1')
        conf.vc.resume_from = str(first)
        version[0] = '2.0'
        with pytest.raises(ValueError, match='worker Python environment changed'):
            run('worker-v2')
