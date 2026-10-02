import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import rvcbench
from rvcbench.benchmark import artifacts
from rvcbench.benchmark.cli import HYDRA_COMMANDS
from rvcbench.benchmark.fingerprints import generation_runtime


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(rvcbench.__file__).resolve().parent


def run_cli(*arguments, cwd):
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(
        filter(None, [str(PACKAGE.parent), os.environ.get('PYTHONPATH')]))}
    return subprocess.run([sys.executable, '-m', 'rvcbench.benchmark.cli', *arguments], cwd=cwd,
                          env=environment, capture_output=True, text=True)


def test_package_exposes_a_version():
    assert rvcbench.__version__ and rvcbench.__version__ != '0+unknown'


@pytest.mark.parametrize('command,config', [
    ('run', 'ots_vc/clean/libritts/qwen3_tts_ots'),
    ('run-protected', 'ots_vc/clean/libritts/qwen3_tts_ots'),
])
def test_run_commands_compose_packaged_configs_from_any_directory(tmp_path, command, config):
    result = run_cli(command, '--config-name', config, '--cfg', 'job', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    composed = OmegaConf.create(result.stdout)
    assert composed.vc.model == 'qwen3_tts' and composed.dataset.name


def test_every_run_command_has_an_importable_entrypoint_and_wrapper():
    wrappers = {'run': 'run_vc.py', 'run-protected': 'run_vc_protect.py', 'protect': 'run_protect.py',
                'denoise': 'run_denoiser.py'}
    assert set(HYDRA_COMMANDS) == set(wrappers)
    for command, (module, _) in HYDRA_COMMANDS.items():
        source = PACKAGE.parent.joinpath(*module.split('.')).with_suffix('.py')
        assert 'config_path="../configs"' in source.read_text()
        assert f'from {module} import main' in (ROOT / wrappers[command]).read_text()


def test_package_data_patterns_cover_every_tracked_data_file():
    listed = subprocess.run(['git', 'ls-files', '-z', 'src/rvcbench'], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode or not listed.stdout:
        pytest.skip('not a Git checkout')
    patterns = tomllib.loads((ROOT / 'pyproject.toml').read_text())['tool']['setuptools']['package-data']['rvcbench']
    covered = {path for pattern in patterns for path in PACKAGE.glob(pattern)}
    data_files = [ROOT / name for name in filter(None, listed.stdout.split('\0')) if not name.endswith('.py')]
    assert [str(path.relative_to(PACKAGE)) for path in data_files if path not in covered] == []


def test_checkout_layout_is_detected():
    assert artifacts.runtime_root() == ROOT
    assert artifacts.package_parent(ROOT) == ROOT / 'src'
    if (ROOT / '.git').exists():
        assert artifacts.provenance(ROOT)['git_commit']


def test_installed_layout_records_package_relative_sources(tmp_path):
    site = tmp_path / 'site-packages'
    (site / 'rvcbench/adversary').mkdir(parents=True)
    (site / 'rvcbench/benchmark').mkdir()
    (site / 'rvcbench/adversary/fixture.py').write_text('import numpy\n')
    (site / 'rvcbench/benchmark/runner.py').write_text('VERSION = 1\n')
    assert artifacts.package_parent(site) == site
    with patch.object(artifacts, 'PACKAGE_DIR', site / 'rvcbench'):
        assert artifacts.runtime_root() == site
    conf = OmegaConf.create({'vc': {'mode': 'ots', 'model': 'fixture'}, 'adversary': {}})
    with patch.dict('rvcbench.benchmark.fingerprints._ADVERSARY_REGISTRY',
                    {'ots': {'fixture': 'rvcbench.adversary.fixture:Adapter'}}), \
            patch('importlib.metadata.packages_distributions', return_value={'numpy': ['numpy']}), \
            patch('importlib.metadata.requires', return_value=[]):
        runtime = generation_runtime(site, conf, {'numpy': '1.26.4'})
    assert sorted(runtime['source_files']) == ['rvcbench/adversary/fixture.py', 'rvcbench/benchmark/runner.py']
    recorded = artifacts.provenance(site)
    assert recorded['git_commit'] is None and recorded['git_dirty'] is None
    hashed = recorded['source_sha256']
    (site / 'rvcbench/benchmark/runner.py').write_text('VERSION = 2\n')
    assert artifacts.provenance(site)['source_sha256'] != hashed
