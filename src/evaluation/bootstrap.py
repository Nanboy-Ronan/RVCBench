"""Utility helpers for metric bootstrapping during evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

try:  # Optional dependency; we only use it when available.
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover - OmegaConf might be absent at runtime
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore


@dataclass(frozen=True)
class BootstrapParams:
    enabled: bool
    num_resamples: int
    confidence_level: float
    seed: Optional[int]


class MetricBootstrapper:
    def __init__(self, params: BootstrapParams):
        self.params = params
        # When no explicit seed is provided we rely on the global NumPy RNG,
        # which is already seeded via ``configure_seeds`` for the entire run.
        if params.seed is not None:
            self._choice = np.random.default_rng(params.seed).choice
        else:
            self._choice = np.random.choice

    @staticmethod
    def _clean(values: Sequence[Optional[float]]) -> np.ndarray:
        arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
        if arr.size == 0:
            return arr
        return arr[np.isfinite(arr)]

    def compute_interval(self, values: Sequence[Optional[float]]):
        arr = self._clean(values)
        if arr.size <= 1:
            return None

        mean_value = float(arr.mean())
        samples = np.empty(self.params.num_resamples, dtype=np.float64)
        for i in range(self.params.num_resamples):
            draw = self._choice(arr, size=arr.size, replace=True)
            samples[i] = float(draw.mean())

        tail = 0.5 * (1.0 - self.params.confidence_level)
        lower = float(np.quantile(samples, tail))
        upper = float(np.quantile(samples, 1.0 - tail))
        std = float(samples.std(ddof=1)) if self.params.num_resamples > 1 else 0.0
        return {
            "lower": lower,
            "upper": upper,
            "confidence_level": self.params.confidence_level,
            "num_resamples": self.params.num_resamples,
            "std": std,
            "avg": mean_value,
        }

    def maybe_add_interval(
        self,
        result: MutableMapping[str, Any],
        metric_name: str,
        values: Sequence[Optional[float]],
    ):
        interval = self.compute_interval(values)
        if interval is not None:
            result[f"{metric_name}_ci"] = interval


def _to_plain_mapping(config: Any):
    if config is None:
        return None
    if DictConfig is not None and isinstance(config, DictConfig):  # type: ignore[arg-type]
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[union-attr]
    if isinstance(config, Mapping):
        return config
    return None


def create_bootstrapper(config: Any) -> Optional[MetricBootstrapper]:
    mapping = _to_plain_mapping(config)
    if not mapping:
        return None

    enabled = bool(mapping.get("enabled", False))
    if not enabled:
        return None

    num_resamples = int(mapping.get("num_samples", mapping.get("num_resamples", 1000)))
    if num_resamples <= 0:
        raise ValueError("evaluation.bootstrap.num_samples must be > 0 when bootstrapping is enabled.")

    confidence_level = float(mapping.get("confidence_level", 0.95))
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("evaluation.bootstrap.confidence_level must be between 0 and 1.")

    raw_seed = mapping.get("seed")
    seed = int(raw_seed) if raw_seed is not None else None

    params = BootstrapParams(
        enabled=True,
        num_resamples=num_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return MetricBootstrapper(params)
