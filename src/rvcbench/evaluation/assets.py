"""Location of scorer model files."""
import os
from pathlib import Path


def asset_dir(name):
    """Directory holding the model files of one scorer.

    ``$RVCBENCH_ASSET_DIR/<name>`` when that variable is set; otherwise ``./checkpoints/<name>``
    when it exists (source checkouts); otherwise a per-user cache, so scoring from any
    directory never creates files there.
    """
    configured = os.environ.get('RVCBENCH_ASSET_DIR')
    if configured:
        return Path(configured).expanduser().absolute() / name
    local = Path.cwd() / 'checkpoints' / name
    if local.exists():
        return local
    cache = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    return cache / 'rvcbench' / name
