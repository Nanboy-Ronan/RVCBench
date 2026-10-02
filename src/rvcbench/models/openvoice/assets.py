"""Explicit local MeloTTS asset checks for OpenVoice."""
from pathlib import Path


def melo_files(entry):
    files = {}
    for key in ('config_path', 'checkpoint_path'):
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f'OpenVoice Melo model requires a resolved local {key}')
        path = Path(value).expanduser().resolve()
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f'OpenVoice Melo asset is missing or empty: {path}')
        files[key] = path
    return files
