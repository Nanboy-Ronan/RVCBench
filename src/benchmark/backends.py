"""Execution contract and an explicit bridge for the existing model adapters.

The bridge is serial: a batch is a scheduling unit, not a claim of vectorized
inference. New backends can implement the same contract without a dataset API.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
import time

from .artifacts import output_path, sample_id
from .registry import select_adversary
from src.datasets.zero_shot import ZeroShotSample


@dataclass(frozen=True)
class GenerationRequest:
    sample: ZeroShotSample
    seed: int
    output_dir: Path
    protected_audio_dir: Path | None = None


@dataclass(frozen=True)
class GenerationResult:
    sample_id: str
    path: Path
    elapsed_sec: float
    timing_scope: str
    native_seed: int | None = None
    native_seed_policy: str | None = None
    native_requested_seed: int | None = None


class Backend(Protocol):
    def prepare(self) -> None: ...
    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...
    def close(self) -> None: ...


class SampleView:
    def __init__(self, dataset, sample):
        self.dataset, self.sample = dataset, sample

    def get_zero_shot_samples(self, **kwargs):
        return [self.sample]

    def iter_zero_shot_samples(self, **kwargs):
        return iter([self.sample])

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class LegacyAdversaryBackend:
    def __init__(self, conf, dataset, device, logger):
        self.conf, self.dataset, self.device, self.logger = conf, dataset, device, logger
        self.adapter = None

    def prepare(self):
        if self.adapter is None:
            self.adapter = select_adversary(self.conf, self.conf.dataset, self.device, self.logger)
            self.adapter.prepare()

    def generate_batch(self, requests):
        from src.utils.seeding import configure_seeds
        if self.adapter is None:
            raise RuntimeError('Backend.prepare() must precede generation')
        results = []
        for request in requests:
            configure_seeds(request.seed, logger=None)
            started = time.perf_counter()
            self.adapter.attack(output_path=str(request.output_dir),
                dataset=SampleView(self.dataset, request.sample),
                protected_audio_path=str(request.protected_audio_dir) if request.protected_audio_dir else None)
            elapsed = time.perf_counter() - started
            scope = 'adapter_call_including_lazy_initialization'
            dest = output_path(request.output_dir, request.sample)
            for timing in getattr(self.adapter, '_synthesis_timing_records', []):
                if Path(timing['generated_path']).resolve() == dest.resolve():
                    elapsed = timing['synthesis_time_sec']
                    scope = 'adapter_reported_synthesis'
            generator = getattr(self.adapter, '_generator', None)
            results.append(GenerationResult(sample_id(request.sample), dest, elapsed, scope,
                native_seed=getattr(generator, 'last_native_seed', None),
                native_seed_policy=getattr(getattr(generator, 'config', None), 'native_seed_policy', None),
                native_requested_seed=getattr(generator, 'last_native_requested_seed', None)))
        return results

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None


class Qwen3Backend(LegacyAdversaryBackend):
    """Qwen3 inference consumes samples directly without a dataset facade."""

    def generate_batch(self, requests):
        from src.utils.seeding import configure_seeds
        if self.adapter is None:
            raise RuntimeError('Backend.prepare() must precede generation')
        results = []
        for request in requests:
            native_seed = self.adapter.seed
            if native_seed is not None and int(native_seed) + request.sample.index != request.seed:
                raise ValueError('Qwen3 native seed differs from the generation request')
            configure_seeds(request.seed, logger=None)
            path, elapsed = self.adapter.generate_sample(request.sample, output_dir=request.output_dir)
            results.append(GenerationResult(sample_id(request.sample), path, elapsed,
                'qwen3_generate_excluding_prompt_encoding_and_io_v1',
                native_seed=request.seed, native_seed_policy='source_index', native_requested_seed=request.seed))
        return results


def create_backend(conf, dataset, device, logger):
    backend = Qwen3Backend if conf.vc.model == 'qwen3_tts' else LegacyAdversaryBackend
    return backend(conf, dataset, device, logger)
