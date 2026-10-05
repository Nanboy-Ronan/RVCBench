"""Release-pinned scorer assets, verified before any upstream code or weights load."""
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from rvcbench.benchmark.artifacts import file_hash
from .assets import asset_dir


def scorer_lock(name):
    return json.loads(Path(__file__).with_name('scorer_lock.json').read_text())[name]


def verify_files(root, files):
    root = Path(root)
    for name, expected in files.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Pinned scorer file missing: {path}; run `rvcbench setup-scorers`')
        if file_hash(path) != expected:
            raise ValueError(f'{path} differs from the pinned scorer asset; move it aside and rerun `rvcbench setup-scorers`')
    return dict(files)


def fetch_url(url, path, sha256, *, cached=None, check_only=False):
    """Download or copy a matching cache entry atomically; never replace mismatched files."""
    path = Path(path)
    if path.is_file():
        verify_files(path.parent, {path.name: sha256})
        return
    if check_only:
        raise FileNotFoundError(f'{path}; run `rvcbench setup-scorers`')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.download-')
    try:
        with os.fdopen(fd, 'wb') as output:
            if cached and Path(cached).is_file() and file_hash(cached) == sha256:
                with Path(cached).open('rb') as source:
                    shutil.copyfileobj(source, output)
            else:
                with urlopen(url, timeout=120) as source:
                    shutil.copyfileobj(source, output)
        verify_files(Path(temporary).parent, {Path(temporary).name: sha256})
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def speechmos_paths():
    spec = scorer_lock('speechmos')
    root = asset_dir('speechmos') / spec['revision']
    return root / 'source', root / 'utmos22_strong_step7459_v1.pt'


def whisper_path():
    return Path(os.getenv('XDG_CACHE_HOME') or Path.home() / '.cache') / 'whisper' / 'medium.pt'


def verify_speechmos():
    spec = scorer_lock('speechmos')
    source, weights = speechmos_paths()
    verify_files(source, spec['files'])
    verify_files(weights.parent, {weights.name: spec['weights_sha256']})
    return source, weights, spec
