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
