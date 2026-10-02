#!/usr/bin/env python3
"""Fetch pinned native emotion assets, preserving and checking existing files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rvcbench.benchmark.artifacts import file_hash
from rvcbench.evaluation.scorers.emotion import ASSETS, BASE_REVISION, REVISION


def setup(check_only=False):
    from huggingface_hub import hf_hub_download
    cache = ROOT / 'checkpoints/hf-cache'
    root = ROOT / 'checkpoints/emotion-native'
    base = ROOT / 'wav2vec2_checkpoints/models--facebook--wav2vec2-base/snapshots' / BASE_REVISION
    hashes = {}
    for repo, revision, destination, files in (
        ('speechbrain/emotion-recognition-wav2vec2-IEMOCAP', REVISION, root, ASSETS),
        ('facebook/wav2vec2-base', BASE_REVISION, base,
         dict.fromkeys(('config.json', 'preprocessor_config.json', 'pytorch_model.bin'))),
    ):
        for name, expected in files.items():
            path = destination / name
            if path.exists() or path.is_symlink():
                if not path.is_file():
                    raise ValueError(f'Existing asset is not a readable file: {path}')
            else:
                if check_only:
                    raise FileNotFoundError(path)
                source = Path(hf_hub_download(repo, name, revision=revision, cache_dir=str(cache)))
                if expected and file_hash(source) != expected:
                    raise ValueError(f'Downloaded pinned asset differs: {name}')
                destination.mkdir(parents=True, exist_ok=True)
                path.symlink_to(source.resolve())
            actual = file_hash(path)
            if expected and actual != expected:
                raise ValueError(f'Existing pinned asset differs: {path}; refusing to overwrite')
            hashes[str(path.relative_to(ROOT))] = actual
    return hashes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify local assets without downloading or modifying files')
    args = parser.parse_args()
    print(json.dumps(setup(args.check_only), indent=2))
