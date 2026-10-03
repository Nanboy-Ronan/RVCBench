import sys
from types import ModuleType, SimpleNamespace

import pytest

from rvcbench.benchmark.artifacts import file_hash
from rvcbench.evaluation import setup
from rvcbench.evaluation.scorers import emotion


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('RVCBENCH_ASSET_DIR', str(tmp_path / 'assets'))
    return tmp_path / 'assets'


@pytest.fixture
def hub(monkeypatch):
    """A fake Hub that serves fixed bytes per file name and records downloads."""
    calls = []

    def download(repo, name, revision, local_dir):
        calls.append((repo, name, revision))
        path = __import__('pathlib').Path(local_dir) / name
        path.write_text(f'{repo}/{name}')
        return str(path)

    module = ModuleType('huggingface_hub')
    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', module)
    return calls


def pin_assets(monkeypatch, tmp_path):
    names = ['custom_interface.py', 'model.ckpt']
    sample = tmp_path / 'sample'
    hashes = {}
    for name in names:
        sample.write_text(f'speechbrain/emotion-recognition-wav2vec2-IEMOCAP/{name}')
        hashes[name] = file_hash(sample)
    monkeypatch.setattr(emotion, 'ASSETS', hashes)
    return hashes


def test_emotion_files_are_fetched_into_the_asset_directory_at_pinned_revisions(assets, hub, monkeypatch, tmp_path):
    hashes = pin_assets(monkeypatch, tmp_path)
    report = setup.setup_emotion()
    assert (assets / 'emotion-native' / 'model.ckpt').is_file()
    assert (assets / 'wav2vec2-base' / emotion.BASE_REVISION / 'pytorch_model.bin').is_file()
    assert {(repo, revision) for repo, _, revision in hub} == {
        ('speechbrain/emotion-recognition-wav2vec2-IEMOCAP', emotion.REVISION), ('facebook/wav2vec2-base', emotion.BASE_REVISION)}
    assert {k: v for k, v in report['files'].items() if k in hashes} == hashes
    hub.clear()
    setup.setup_emotion()
    assert hub == []  # present files are verified, not downloaded again


def test_changed_pinned_file_is_refused(assets, hub, monkeypatch, tmp_path):
    pin_assets(monkeypatch, tmp_path)
    (assets / 'emotion-native').mkdir(parents=True)
    (assets / 'emotion-native' / 'model.ckpt').write_text('something else')
    with pytest.raises(ValueError, match='differs from the pinned'):
        setup.setup_emotion()


def test_check_only_downloads_nothing(assets, hub, monkeypatch, tmp_path):
    pin_assets(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        setup.setup_emotion(check_only=True)
    assert hub == [] and not assets.exists()


def test_legacy_wav2vec2_location_is_used_in_source_checkouts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('RVCBENCH_ASSET_DIR', raising=False)
    legacy = tmp_path / 'wav2vec2_checkpoints' / 'models--facebook--wav2vec2-base' / 'snapshots' / emotion.BASE_REVISION
    legacy.mkdir(parents=True)
    assert emotion.base_initializer_dir() == legacy
    monkeypatch.setenv('RVCBENCH_ASSET_DIR', str(tmp_path / 'assets'))
    assert emotion.base_initializer_dir() == tmp_path / 'assets' / 'wav2vec2-base' / emotion.BASE_REVISION


def test_setup_scorers_reports_each_metric_and_loads_the_scorers(assets, monkeypatch):
    loaded = []

    class Scorer:
        def __init__(self, name):
            self.name = name

        def prepare(self):
            loaded.append(self.name)

        def close(self):
            pass

    monkeypatch.setattr('rvcbench.evaluation.scorers.create_scorer', lambda name, device, logger: Scorer(name))
    monkeypatch.setitem(setup.SETUP, 'wer', lambda check_only: {'location': 'whisper', 'files': {}})
    report = setup.setup_scorers(['wer', 'mcd', 'stoi', 'dnsmos'])
    assert report['mcd']['status'] == report['stoi']['status'] == 'no model files needed'
    assert report['dnsmos']['status'] == 'manual setup'
    assert report['wer']['status'] == 'ready (loaded)' and loaded == ['wer']


def test_setup_scorers_command(monkeypatch, capsys):
    from rvcbench.benchmark import cli
    seen = {}
    monkeypatch.setattr(sys, 'argv', ['rvcbench', 'setup-scorers', '--metrics', 'mcd', '--check-only'])
    monkeypatch.setattr('rvcbench.evaluation.setup.setup_scorers',
                        lambda metrics, check_only: seen.update(metrics=metrics, check_only=check_only) or {'mcd': {}})
    cli.main()
    assert seen == {'metrics': ['mcd'], 'check_only': True}
    assert '"mcd"' in capsys.readouterr().out


def test_missing_speechmos_points_to_the_setup_command(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path))
    from rvcbench.evaluation.scorers.auxiliary import AuxiliaryScorer
    with pytest.raises(FileNotFoundError, match='rvcbench setup-scorers'):
        AuxiliaryScorer('speechmos', 'cpu', SimpleNamespace(info=lambda *a: None)).prepare()
