import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import soundfile as sf

import rvcbench
from rvcbench.benchmark.registry import adapter_target
from test_benchmark import setup_run


ADAPTER = '''
from rvcbench import VoiceCloningAdapter
from .tones import tone


class ToneAdapter(VoiceCloningAdapter):
    def load(self):
        self.loaded = True

    def clone(self, *, text, reference_audio, reference_text, language):
        assert self.loaded and reference_audio.is_file() and reference_text == "hello"
        if self.config.get("fail"):
            raise RuntimeError("model exploded")
        return tone(len(text), invalid=self.config.get("invalid", False)), 16000
'''
TONES = '''
import numpy as np


def tone(extra, invalid=False):
    wave = 0.1 * np.sin(np.arange(1600 + extra) * 0.1)
    return wave * np.nan if invalid else wave
'''
TARGET = 'my_models.adapter:ToneAdapter'


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    package = tmp_path / 'plugins' / 'my_models'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'adapter.py').write_text(ADAPTER)
    (package / 'tones.py').write_text(TONES)
    monkeypatch.syspath_prepend(str(package.parent))
    yield package
    for name in [name for name in sys.modules if name == 'my_models' or name.startswith('my_models.')]:
        del sys.modules[name]


def external_config(conf, **adversary):
    return OmegaConf.merge(conf, {'vc': {'model': 'tone', 'adapter': TARGET}, 'adversary': adversary})


def test_external_adapter_generates_and_records_its_source(setup_run, plugins):
    conf, _, run, _ = setup_run
    out, manifest, _ = run('external', config=external_config(conf))
    assert manifest['status'] == 'generated'
    assert manifest['coverage']['generated'] == manifest['coverage']['requested'] == 2
    for row in manifest['samples']:
        audio, rate = sf.read(row['generated_path'])
        assert rate == 16000 and audio.size == 1600 + len('hello')
        assert row['timing_scope'] == 'external_adapter_clone_call_excluding_output_write'
    sources = manifest['generation_provenance']['source_files']
    for name in ('my_models/adapter.py', 'my_models/tones.py'):
        assert any(Path(source).as_posix().endswith(name) for source in sources), name
    assert any(name.endswith('rvcbench/adapter.py') for name in sources)

    (plugins / 'tones.py').write_text(TONES + '\nVERSION = 2\n')
    _, changed, _ = run('changed', config=external_config(conf))
    assert (changed['generation_provenance']['source_sha256']
            != manifest['generation_provenance']['source_sha256'])


@pytest.mark.parametrize('adversary,message', [
    ({'fail': True}, 'model exploded'),
    ({'invalid': True}, 'nonempty finite mono audio'),
])
def test_external_adapter_failures_are_recorded_per_sample(setup_run, plugins, adversary, message):
    conf, _, run, _ = setup_run
    _, manifest, _ = run('failing', config=external_config(conf, **adversary))
    assert manifest['status'] != 'generated'
    assert manifest['coverage']['generated'] == 0
    assert all(message in json.dumps(row) for row in manifest['samples'])


def test_external_adapter_must_have_a_source_file(setup_run):
    conf, _, run, _ = setup_run
    with pytest.raises(ImportError, match='Cannot locate the source file of vc.adapter'):
        run('missing', config=OmegaConf.merge(conf, {'vc': {'model': 'tone', 'adapter': 'not_installed.adapter:Adapter'}}))


@pytest.mark.parametrize('value', ['my_models.adapter', 'my_models.adapter:', ':ToneAdapter', 'my models:Tone', 'pkg/mod.py:Tone'])
def test_malformed_adapter_targets_are_rejected(value):
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'tone', 'adapter': value}})
    with pytest.raises(ValueError, match="vc.adapter must be 'package.module:ClassName'"):
        adapter_target(conf)


@pytest.mark.parametrize('model', ['xtts', 'qwen3_tts', 'cozyvoice2', 'smoke', 'XTTS'])
def test_external_adapters_cannot_reuse_a_builtin_model_name(model):
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': model, 'adapter': TARGET}})
    with pytest.raises(ValueError, match='names a built-in integration'):
        adapter_target(conf)


def test_external_adapters_use_the_generic_backend():
    from rvcbench.benchmark.backends import LegacyAdversaryBackend, create_backend
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'tone', 'adapter': TARGET}})
    assert type(create_backend(conf, None, 'cpu', None)) is LegacyAdversaryBackend


def test_builtin_models_do_not_need_an_adapter_target():
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'smoke'}})
    assert adapter_target(conf) == 'rvcbench.adversary.smoke:SmokeAdversary'
    with pytest.raises(ValueError, match='Set vc.adapter='):
        adapter_target(OmegaConf.create({'vc': {'mode': 'ots', 'model': 'unknown_model'}}))


def cli(arguments, plugins, cwd):
    paths = [str(plugins.parent), str(Path(rvcbench.__file__).resolve().parents[1]), os.environ.get('PYTHONPATH')]
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, paths))}
    return subprocess.run([sys.executable, '-m', 'rvcbench.benchmark.cli', 'run', '--config-name',
                           'ots_vc/clean/libritts/custom_ots', *arguments], cwd=cwd, env=environment,
                          capture_output=True, text=True)


def test_run_command_drives_an_external_adapter_from_the_template_config(setup_run, plugins):
    conf, _, _, root = setup_run
    workdir = root / 'elsewhere'
    workdir.mkdir()
    result = cli([f'vc.adapter={TARGET}', 'vc.model=tone', 'run_name=tone_on_fixture', '+vc.generate_only=true',
                  'device=cpu', f'base_dir={root}', f'dataset.root_path={conf.dataset.root_path}',
                  'dataset.use_hf_dataset=false', 'dataset.speaker_id=one', 'dataset.manifest_variant=null',
                  '+dataset.manifest_filename=metadata.json', 'adversary.max_samples=1',
                  f'hydra.run.dir={root}/hydra'], plugins, workdir)
    assert result.returncode == 0, result.stderr
    manifest, = root.glob('results/tone_on_fixture/*/run_manifest.json')
    recorded = json.loads(manifest.read_text())
    assert recorded['status'] == 'generated' and recorded['coverage']['requested'] == 1
    assert recorded['config']['vc']['adapter'] == TARGET


def test_template_config_requires_an_adapter_before_touching_the_dataset(setup_run, plugins):
    _, _, _, root = setup_run
    result = cli(['device=cpu', f'base_dir={root}', f'dataset.root_path={root}/no-such-dataset',
                  'dataset.use_hf_dataset=false', f'hydra.run.dir={root}/hydra'], plugins, root)
    assert result.returncode != 0
    assert 'Set vc.adapter=package.module:ClassName' in result.stderr


def test_documented_example_adapter_runs(setup_run):
    conf, _, _, root = setup_run
    examples = Path(__file__).resolve().parents[1] / 'examples'
    result = cli(['vc.adapter=echo_adapter:EchoAdapter', 'vc.model=echo', 'run_name=echo_on_fixture',
                  '+vc.generate_only=true', '+adversary.gain=0.5', 'device=cpu', f'base_dir={root}',
                  f'dataset.root_path={conf.dataset.root_path}', 'dataset.use_hf_dataset=false',
                  'dataset.speaker_id=one', 'dataset.manifest_variant=null',
                  '+dataset.manifest_filename=metadata.json', f'hydra.run.dir={root}/hydra'],
                 examples / 'echo_adapter.py', root)
    assert result.returncode == 0, result.stderr
    manifest, = root.glob('results/echo_on_fixture/*/run_manifest.json')
    recorded = json.loads(manifest.read_text())
    assert recorded['status'] == 'generated' and recorded['coverage']['generated'] == 2
    reference, _ = sf.read(Path(conf.dataset.root_path) / 'a.wav')
    generated, _ = sf.read(recorded['samples'][0]['generated_path'])
    assert abs(float(abs(generated).max()) - 0.5 * float(abs(reference).max())) < 1e-3
