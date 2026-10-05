"""Download and verify the model files the scorers need (`rvcbench setup-scorers`)."""
from pathlib import Path

from rvcbench.benchmark.artifacts import file_hash
from .assets import asset_dir
from .locked_assets import scorer_lock, speechmos_paths, fetch_url

BASE_FILES = ('config.json', 'preprocessor_config.json', 'pytorch_model.bin')


def _fetch(repo, revision, destination, files, check_only):
    """Place pinned Hub files as regular files in ``destination`` and check their hashes."""
    from huggingface_hub import hf_hub_download
    destination = Path(destination)
    recorded = {}
    for name, expected in files.items():
        path = destination / name
        if not path.is_file():
            if check_only:
                raise FileNotFoundError(path)
            destination.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo, name, revision=revision, local_dir=str(destination))
        actual = file_hash(path)
        if expected and actual != expected:
            raise ValueError(f'{path} differs from the pinned {repo} file; remove it and rerun')
        recorded[name] = actual
    return {'location': str(destination), 'files': recorded}


def setup_emotion(check_only=False):
    from .scorers.emotion import ASSETS, BASE_REVISION, REVISION, base_initializer_dir
    model = _fetch('speechbrain/emotion-recognition-wav2vec2-IEMOCAP', REVISION, asset_dir('emotion-native'),
                   ASSETS, check_only)
    base = _fetch('facebook/wav2vec2-base', BASE_REVISION, base_initializer_dir(), scorer_lock('emotion_base')['files'], check_only)
    return {'location': model['location'], 'files': {**model['files'], **{f'wav2vec2-base/{k}': v for k, v in base['files'].items()}}}


def setup_speechmos(check_only=False):
    import torch
    hub = Path(torch.hub.get_dir())
    spec = scorer_lock('speechmos')
    source, weights = speechmos_paths()
    for name, sha in spec['files'].items():
        fetch_url(f'https://raw.githubusercontent.com/{spec["repo"]}/{spec["revision"]}/{name}',
                  source / name, sha, cached=hub / 'tarepan_SpeechMOS_main' / name, check_only=check_only)
    fetch_url(spec['weights_url'], weights, spec['weights_sha256'],
              cached=hub / 'checkpoints' / weights.name, check_only=check_only)
    return {'location': str(source.parent), 'revision': spec['revision'],
            'files': {**spec['files'], weights.name: spec['weights_sha256']}}


def setup_whisper(check_only=False):
    from .locked_assets import whisper_path
    spec, path = scorer_lock('wer'), whisper_path()
    fetch_url(spec['weights_url'], path, spec['weights_sha256'], check_only=check_only)
    return {'location': str(path.parent), 'files': {path.name: spec['weights_sha256']}}


def setup_speaker(check_only=False):
    root = asset_dir('spkrec-ecapa-voxceleb')
    spec = scorer_lock('sim')
    return {**_fetch(spec['repo'], spec['revision'], root, spec['files'], check_only),
            'revision': spec['revision']}


SETUP = {'sim': setup_speaker, 'speechmos': setup_speechmos, 'wer': setup_whisper, 'emotion': setup_emotion}
NO_ASSETS = ('mcd', 'stoi')


def setup_scorers(metrics, *, check_only=False, verify=True):
    """Fetch (or only check) each metric's model files, then load each scorer once on CPU."""
    from .scorers import create_scorer
    groups = sorted({'sim' if metric == 'sva' else metric for metric in metrics})
    report = {}
    for group in groups:
        if group in NO_ASSETS:
            report[group] = {'status': 'no model files needed'}
            continue
        if group not in SETUP:
            report[group] = {'status': 'manual setup', 'location': str(asset_dir(group))}
            continue
        entry = {'status': 'ready', **SETUP[group](check_only)}
        if verify:
            scorer = create_scorer(group, 'cpu', None)
            try:
                scorer.prepare()
            finally:
                scorer.close()
            entry['status'] = 'ready (loaded)'
        if group == 'sim':
            root = Path(entry['location'])
            entry['files'] = {p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()}
        report[group] = entry
    return report
