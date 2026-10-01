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


@pytest.mark.parametrize('suffix', ['.cu', '.cuh', '.cpp', '.h', '.pyx', '.pxd', '.cmake'])
def test_upstream_native_source_changes_generation_fingerprint(tmp_path, suffix):
    from src.benchmark.fingerprints import generation_runtime
    adapter = tmp_path / 'src/adversary/fixture.py'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('')
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    source = upstream / ('kernel' + suffix)
    source.write_text('source version one')
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'fixture'},
                            'adversary': {'code_path': str(upstream)}})
    with patch.dict('src.benchmark.fingerprints._ADVERSARY_REGISTRY',
                    {'ots': {'fixture': 'src.adversary.fixture:Adapter'}}), \
            patch('importlib.metadata.packages_distributions', return_value={}):
        before = generation_runtime(tmp_path, conf, {})
        assert str(source.relative_to(tmp_path)) in before['source_files']
        source.write_text('source version two')
        after = generation_runtime(tmp_path, conf, {})
    assert before['source_sha256'] != after['source_sha256']


@pytest.mark.parametrize('source', [
    b'\xef\xbb\xbfimport numpy\n',
    b'# coding: latin-1\n# caf\xe9\nimport numpy\n',
])
def test_python_encoding_rules_preserve_upstream_hash_and_imports(tmp_path, source):
    from src.benchmark.fingerprints import generation_runtime
    from src.benchmark.artifacts import file_hash
    adapter = tmp_path / 'src/adversary/fixture.py'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('')
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    path = upstream / 'native.py'
    path.write_bytes(source)
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'fixture'},
                            'adversary': {'code_path': str(upstream)}})
    with patch.dict('src.benchmark.fingerprints._ADVERSARY_REGISTRY',
                    {'ots': {'fixture': 'src.adversary.fixture:Adapter'}}), \
            patch('importlib.metadata.packages_distributions', return_value={'numpy': ['numpy']}), \
            patch('importlib.metadata.requires', return_value=[]):
        result = generation_runtime(tmp_path, conf, {'numpy': '1.26.4'})
    assert result['source_files']['upstream/native.py'] == file_hash(path)
    assert result['packages']['numpy'] == '1.26.4'


def test_authored_build_definition_changes_resume_but_generated_outputs_do_not(tmp_path):
    from src.benchmark.fingerprints import generation_runtime
    adapter = tmp_path / 'src/adversary/fixture.py'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('')
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    definition = upstream / 'CMakeLists.txt'
    definition.write_text('set(FLAG 1)')
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'fixture'},
                            'adversary': {'code_path': str(upstream)}})
    with patch.dict('src.benchmark.fingerprints._ADVERSARY_REGISTRY',
                    {'ots': {'fixture': 'src.adversary.fixture:Adapter'}}), \
            patch('importlib.metadata.packages_distributions', return_value={}):
        before = generation_runtime(tmp_path, conf, {})
        for dirname in ('build', 'dist', '.venv', '__pycache__', '.git'):
            directory = upstream / dirname
            directory.mkdir()
            (directory / 'generated.cpp').write_text('generated output')
            (directory / 'invalid.py').write_text('not valid Python!')
        (upstream / 'kernel.so').write_bytes(b'compiled output')
        assert generation_runtime(tmp_path, conf, {}) == before
        definition.write_text('set(FLAG 2)')
        assert generation_runtime(tmp_path, conf, {})['source_sha256'] != before['source_sha256']


def test_runner_rejects_resume_after_native_source_change(setup_run):
    conf, _, run, tmp_path = setup_run
    upstream = tmp_path / 'native-upstream'
    upstream.mkdir()
    kernel = upstream / 'kernel.cu'
    kernel.write_text('kernel version one')
    conf.adversary.code_path = str(upstream)
    first, _, _ = run('native-first')
    conf.vc.resume_from = str(first)
    build = upstream / 'build'
    build.mkdir()
    (build / 'temporary.cpp').write_text('generated compiler output')
    with patch('src.benchmark.backends.select_adversary', side_effect=AssertionError('must reuse')):
        _, resumed, _ = run('native-build-reuse')
    assert all(row['reused'] for row in resumed['samples'])
    kernel.write_text('kernel version two')
    with pytest.raises(ValueError, match='runtime source changed'):
        run('native-source-rejected')


@pytest.mark.parametrize('settings,imports', [
    ({'attn_implementation': 'flash_attention_2'}, ''),
    ({'use_flash_attn': True}, ''),
    ({'use_flash_attn2': True}, ''),
    ({}, 'import flash_attn\n'),
    ({}, 'import transformers\n'),
])
def test_optional_attention_absence_and_installation_change_runtime_fingerprint(tmp_path, settings, imports):
    from src.benchmark.fingerprints import generation_runtime
    adapter = tmp_path / 'src/adversary/fixture.py'
    adapter.parent.mkdir(parents=True)
    adapter.write_text(imports)
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'fixture'}, 'adversary': settings})
    with patch.dict('src.benchmark.fingerprints._ADVERSARY_REGISTRY',
                    {'ots': {'fixture': 'src.adversary.fixture:Adapter'}}), \
            patch('importlib.metadata.packages_distributions', return_value={'transformers': ['transformers']}), \
            patch('importlib.metadata.requires', return_value=[]):
        absent = generation_runtime(tmp_path, conf, {'transformers': 'fixture'})
        installed = generation_runtime(tmp_path, conf, {'transformers': 'fixture', 'flash_attn': '2.7.4'})
        upgraded = generation_runtime(tmp_path, conf, {'transformers': 'fixture', 'flash-attn': '2.8.0'})
        assert absent['packages']['flash-attn'] is None
        assert installed['packages']['flash-attn'] == '2.7.4'
        assert absent != installed != upgraded


def test_transitive_transformers_attention_dependency_closure(tmp_path):
    from src.benchmark.fingerprints import generation_runtime
    adapter = tmp_path / 'src/adversary/fixture.py'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('import qwen_tts\n')
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'fixture'}, 'adversary': {}})
    requirements = {'qwen-tts': ['transformers'], 'transformers': [], 'flash-attn': ['einops'], 'einops': []}
    with patch.dict('src.benchmark.fingerprints._ADVERSARY_REGISTRY',
                    {'ots': {'fixture': 'src.adversary.fixture:Adapter'}}), \
            patch('importlib.metadata.packages_distributions', return_value={'qwen_tts': ['qwen-tts']}), \
            patch('importlib.metadata.requires', side_effect=requirements.__getitem__):
        first = generation_runtime(tmp_path, conf, {'qwen-tts': 'fixture', 'transformers': 'fixture',
                                                  'flash-attn': 'fixture', 'einops': '0.7'})
        second = generation_runtime(tmp_path, conf, {'qwen-tts': 'fixture', 'transformers': 'fixture',
                                                   'flash-attn': 'fixture', 'einops': '0.8'})
    assert first['packages']['einops'] == '0.7'
    assert first != second


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


def test_model_fingerprint_tracks_checkpoint_code_and_auxiliary_codec(tmp_path):
    from src.benchmark.model_assets import resolve_model_assets
    checkpoint, codec = tmp_path / 'tts', tmp_path / 'codec'
    checkpoint.mkdir()
    codec.mkdir()
    implementation = checkpoint / 'modeling.py'
    implementation.write_text('VERSION = 1\n')
    weights = codec / 'model.safetensors'
    weights.write_bytes(b'codec-v1')
    conf = OmegaConf.create({'vc': {'model': 'moss_tts'}, 'adversary': {
        'checkpoint': str(checkpoint), 'codec_path': str(codec)}})
    _, reference, first = resolve_model_assets(conf)
    assert 'modeling.py' in reference['assets']['checkpoint']['files']
    assert 'model.safetensors' in reference['assets']['codec_path']['files']
    implementation.write_text('VERSION = 2\n')
    second = resolve_model_assets(conf)[2]
    assert second != first
    weights.write_bytes(b'codec-v2')
    assert resolve_model_assets(conf)[2] != second


def test_model_fingerprint_tracks_higgs_audio_tokenizer_and_scene_prompt(tmp_path):
    from src.benchmark.model_assets import resolve_model_assets
    tokenizer = tmp_path / 'tokenizer'
    tokenizer.mkdir()
    weights = tokenizer / 'model.pth'
    weights.write_bytes(b'tokenizer-v1')
    prompt = tmp_path / 'scene.txt'
    prompt.write_text('Quiet recording.')
    conf = OmegaConf.create({'vc': {'model': 'higgs_audio'}, 'adversary': {
        'audio_tokenizer_path': str(tokenizer), 'scene_prompt_path': str(prompt)}})
    _, reference, first = resolve_model_assets(conf)
    assert not reference['unresolved_references']
    assert set(reference['assets']) == {'audio_tokenizer_path', 'scene_prompt_path'}
    weights.write_bytes(b'tokenizer-v2')
    second = resolve_model_assets(conf)[2]
    assert first != second
    prompt.write_text('Noisy recording.')
    assert second != resolve_model_assets(conf)[2]


def test_moss_processor_receives_explicit_codec_path(tmp_path):
    from src.models.moss_tts.generator import MossTTSGenerator, MossTTSGeneratorConfig
    from unittest.mock import Mock
    import torch
    processor = SimpleNamespace(audio_tokenizer=SimpleNamespace(to=lambda device: object()),
                                model_config=SimpleNamespace(sampling_rate=24000))
    generator = MossTTSGenerator(MossTTSGeneratorConfig(checkpoint=str(tmp_path), codec_path='frozen-codec'),
                                 torch.device('cpu'), logging.getLogger())
    load_processor = Mock(return_value=processor)
    transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
                                   AutoModel=SimpleNamespace(from_pretrained=Mock(return_value=torch.nn.Linear(1, 1))))
    with patch.dict(sys.modules, {'transformers': transformers}):
        generator.load_model()
    assert load_processor.call_args.kwargs['codec_path'] == 'frozen-codec'
    assert generator._sample_rate == 24000


@pytest.mark.parametrize('model,path_field', [('openvoice', 'melo_code_path'), ('mgm_omni', 'repo_root')])
def test_generation_fingerprint_tracks_auxiliary_upstream_source(tmp_path, model, path_field):
    from src.benchmark.fingerprints import generation_runtime
    source = tmp_path / 'melo'
    source.mkdir()
    module = source / 'infer.py'
    module.write_text('TEMPERATURE = 1\n')
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': model},
                            'adversary': {path_field: str(source)}})
    first = generation_runtime(tmp_path, conf, {})
    module.write_text('TEMPERATURE = 2\n')
    assert generation_runtime(tmp_path, conf, {})['source_sha256'] != first['source_sha256']
    module.write_text('import transformers\n')
    with patch('importlib.metadata.packages_distributions',
               return_value={'transformers': ['transformers']}), \
            patch('importlib.metadata.requires', return_value=[]):
        first = generation_runtime(tmp_path, conf, {'transformers': '4.51.3'})
        second = generation_runtime(tmp_path, conf, {'transformers': '4.57.3'})
    assert first['packages']['transformers'] == '4.51.3'
    assert second['packages'] != first['packages']
    requirements = source / 'requirements.txt'
    requirements.write_text('transformers==4.51.3\n')
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


def test_cosyvoice_rejects_transformers_drift_before_model_loading(tmp_path):
    from src.models.cosyvoice.generator import CosyVoiceGenerator
    generator = CosyVoiceGenerator.__new__(CosyVoiceGenerator)
    generator.config = SimpleNamespace(code_path=tmp_path)
    (tmp_path / 'requirements.txt').write_text('transformers==4.51.3\n')
    with patch('importlib.metadata.version', return_value='4.57.3'):
        with pytest.raises(RuntimeError, match='source requires transformers==4.51.3'):
            generator._validate_transformers_version()
    with patch('importlib.metadata.version', return_value='4.51.3'):
        generator._validate_transformers_version()


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
