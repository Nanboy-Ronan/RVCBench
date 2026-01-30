"""Environment helper utilities."""
from __future__ import annotations

import os
from typing import Iterable, Optional

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _normalize_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    token = value.strip().lower()
    if not token:
        return None
    if token in _TRUE_VALUES:
        return True
    if token in _FALSE_VALUES:
        return False
    return None


def configure_offline_env(
    *,
    default: bool = False,
    override_var: str = "AUDIOBENCH_FORCE_OFFLINE",
    env_vars: Iterable[str] = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"),
) -> bool:
    """Ensure offline-related env vars are set consistently.

    If ``override_var`` is present it wins, otherwise ``default`` is used.
    Returns the resolved offline flag.
    """

    override = _normalize_bool(os.environ.get(override_var)) if override_var else None
    offline = override if override is not None else bool(default)
    value = "1" if offline else "0"
    for name in env_vars:
        if name:
            os.environ.setdefault(name, value)
    return offline
