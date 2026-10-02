import os
import random
from typing import Optional

import numpy as np
import torch


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
