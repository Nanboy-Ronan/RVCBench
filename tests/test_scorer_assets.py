import sys
from types import ModuleType, SimpleNamespace

import pytest

from rvcbench.evaluation.assets import asset_dir
from rvcbench.evaluation.scorers.speaker import SpeakerScorer


SPEAKER_FILES = ('hyperparams.yaml', 'embedding_model.ckpt', 'mean_var_norm_emb.ckpt', 'classifier.ckpt')


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('RVCBENCH_ASSET_DIR', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    return tmp_path


def test_environment_variable_takes_precedence(workdir, monkeypatch):
    (workdir / 'checkpoints' / 'dnsmos').mkdir(parents=True)
    monkeypatch.setenv('RVCBENCH_ASSET_DIR', str(workdir / 'assets'))
    assert asset_dir('dnsmos') == workdir / 'assets' / 'dnsmos'


def test_existing_checkout_folder_is_used(workdir):
    (workdir / 'checkpoints' / 'dnsmos').mkdir(parents=True)
    assert asset_dir('dnsmos') == workdir / 'checkpoints' / 'dnsmos'


def test_user_cache_is_the_default_and_nothing_is_created(workdir):
    assert asset_dir('dnsmos') == workdir / 'cache' / 'rvcbench' / 'dnsmos'
    assert sorted(p.name for p in workdir.iterdir()) == []


@pytest.fixture
def speechbrain(monkeypatch):
    calls = []

    class SpeakerRecognition:
        @staticmethod
        def from_hparams(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace()

    strategy = SimpleNamespace(COPY='copy', SYMLINK='symlink')
    modules = {'speechbrain': ModuleType('speechbrain'), 'speechbrain.inference': ModuleType('speechbrain.inference'),
               'speechbrain.inference.speaker': ModuleType('speechbrain.inference.speaker'),
               'speechbrain.utils': ModuleType('speechbrain.utils'),
               'speechbrain.utils.fetching': ModuleType('speechbrain.utils.fetching')}
    modules['speechbrain.inference.speaker'].SpeakerRecognition = SpeakerRecognition
    modules['speechbrain.utils.fetching'].LocalStrategy = strategy
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_speaker_model_fetched_from_the_hub_is_copied(workdir, speechbrain):
    SpeakerScorer('cpu', None).prepare()
    call, = speechbrain
    assert call['source'] == 'speechbrain/spkrec-ecapa-voxceleb' and call['local_strategy'] == 'copy'
    assert call['savedir'] == str(workdir / 'cache' / 'rvcbench' / 'spkrec-ecapa-voxceleb')


def test_complete_local_speaker_model_is_loaded_in_place(workdir, speechbrain):
    root = workdir / 'checkpoints' / 'spkrec-ecapa-voxceleb'
    root.mkdir(parents=True)
    for name in SPEAKER_FILES:
        (root / name).write_text(name)
    scorer = SpeakerScorer('cpu', None)
    scorer.prepare()
    call, = speechbrain
    assert call['source'] == str(root) and 'local_strategy' not in call
    assert set(scorer.model_provenance['files']) == set(SPEAKER_FILES)
