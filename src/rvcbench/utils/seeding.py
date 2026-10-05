import os
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch


@contextmanager
def isolated_seed(seed, *, device='cpu'):
    """Seed a scoring call, restoring the caller's RNGs and cuDNN flags on every exit.

    Global RNGs are temporarily changed: callers must serialize concurrent scoring
    and training in the same process, or use separate processes.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    hash_seed = os.environ.get('PYTHONHASHSEED')
    deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    cudnn_tf32, matmul_tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    target = torch.device(device)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                # torch.manual_seed also changes every CUDA generator, including devices
                # unrelated to this evaluator. Seed only CPU and the requested device.
                torch.random.default_generator.manual_seed(seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = cudnn_tf32, matmul_tf32
        if hash_seed is None:
            os.environ.pop('PYTHONHASHSEED', None)
        else:
            os.environ['PYTHONHASHSEED'] = hash_seed


def configure_seeds(
    seed: Optional[int],
    *,
    deterministic: bool = True,
    disable_benchmark: bool = True,
    logger=None,
) -> Optional[int]:
    """Configure global RNG seeds for reproducible runs.

    Parameters
    ----------
    seed: Optional[int]
        The desired random seed. If ``None`` or not convertible to ``int`` no changes are made.
    deterministic: bool, default=True
        Whether to force deterministic CUDA convolutions when available.
    disable_benchmark: bool, default=True
        Whether to disable the cuDNN benchmark mode to avoid nondeterministic kernel selection.
    logger: logging.Logger, optional
        Logger used to report the applied seed.

    Returns
    -------
    Optional[int]
        The integer seed that was applied, or ``None`` if no changes were made.
    """
    if seed is None:
        if logger is not None:
            logger.warning("No seed provided; skipping deterministic setup.")
        return None

    try:
        seed_value = int(seed)
    except (TypeError, ValueError):
        if logger is not None:
            logger.warning("Invalid seed '%s'; skipping deterministic setup.", seed)
        return None

    random.seed(seed_value)
    np.random.seed(seed_value % (2 ** 32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    if hasattr(torch.backends, "cudnn"):
        if deterministic:
            torch.backends.cudnn.deterministic = True
        if disable_benchmark:
            torch.backends.cudnn.benchmark = False

    if logger is not None:
        logger.info("Set global seed to %d", seed_value)

    return seed_value
