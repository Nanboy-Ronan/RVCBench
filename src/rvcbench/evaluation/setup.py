"""Download and verify the model files the scorers need (`rvcbench setup-scorers`)."""
import os
from pathlib import Path

from rvcbench.benchmark.artifacts import file_hash
from .assets import asset_dir

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
    base = _fetch('facebook/wav2vec2-base', BASE_REVISION, base_initializer_dir(), dict.fromkeys(BASE_FILES), check_only)
    return {'location': model['location'], 'files': {**model['files'], **{f'wav2vec2-base/{k}': v for k, v in base['files'].items()}}}


def setup_speechmos(check_only=False):
    import torch
    hub = Path(torch.hub.get_dir())
    source, weights = hub / 'tarepan_SpeechMOS_main', hub / 'checkpoints' / 'utmos22_strong_step7459_v1.pt'
    if not (source.is_dir() and weights.is_file()):
        if check_only:
            raise FileNotFoundError(weights)
        torch.hub.load('tarepan/SpeechMOS', 'utmos22_strong', trust_repo=True)
    return {'location': str(hub), 'files': {weights.name: file_hash(weights)}}


def setup_whisper(check_only=False):
    import whisper
    root = Path(os.getenv('XDG_CACHE_HOME') or Path.home() / '.cache') / 'whisper'
    path = root / 'medium.pt'
    if not path.is_file():
        if check_only:
            raise FileNotFoundError(path)
        whisper._download(whisper._MODELS['medium'], str(root), False)  # verifies the published checksum
    return {'location': str(root), 'files': {path.name: 'checksum verified by whisper'}}


def setup_speaker(check_only=False):
    root = asset_dir('spkrec-ecapa-voxceleb')
    if check_only:
        missing = [n for n in ('hyperparams.yaml', 'embedding_model.ckpt', 'mean_var_norm_emb.ckpt', 'classifier.ckpt')
                   if not (root / n).is_file()]
        if missing:
            raise FileNotFoundError(root / missing[0])
    return {'location': str(root), 'files': {}}  # the scorer fetches and copies the model when preparing


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
