"""Execution contract and an explicit bridge for the existing model adapters.

The bridge is serial: a batch is a scheduling unit, not a claim of vectorized
inference. New backends can implement the same contract without a dataset API.
"""
from dataclasses import dataclass
from pathlib import Path
from numbers import Integral
from typing import Protocol, Sequence
import time

from .artifacts import output_path, sample_id
from .registry import select_adversary
from rvcbench.datasets.zero_shot import ZeroShotSample


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
    adapter_call_time_sec: float | None = None
    conditioning_variant: str | None = None
    native_initialization: dict | None = None


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
        from rvcbench.utils.seeding import configure_seeds
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
            adapter_call_elapsed = elapsed
            scope = 'adapter_call_including_lazy_initialization'
            dest = output_path(request.output_dir, request.sample)
            for timing in getattr(self.adapter, '_synthesis_timing_records', []):
                if Path(timing['generated_path']).resolve() == dest.resolve():
                    elapsed = timing['synthesis_time_sec']
                    scope = timing.get('timing_scope') or 'adapter_reported_synthesis'
            generator = getattr(self.adapter, '_generator', None)
            results.append(GenerationResult(sample_id(request.sample), dest, elapsed, scope,
                native_seed=getattr(generator, 'last_native_seed', None),
                native_seed_policy=getattr(getattr(generator, 'config', None), 'native_seed_policy', None),
                native_requested_seed=getattr(generator, 'last_native_requested_seed', None),
                adapter_call_time_sec=adapter_call_elapsed))
        return results

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None


class SingleSampleBackend(LegacyAdversaryBackend):
    """Inference consumes explicit samples without a dataset facade."""

    timing_scope: str

    def native_seed_metadata(self, request):
        return dict(native_seed=int(request.seed), native_seed_policy='source_index',
                    native_requested_seed=int(request.seed))

    def validate_request(self, request):
        if isinstance(request.seed, bool) or not isinstance(request.seed, Integral):
            raise ValueError('Generation request seed must be an integer')
        if not -(2**63) <= request.seed < 2**64:
            raise ValueError('Generation request seed is outside the supported Torch range')
        index = request.sample.index
        if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
            raise ValueError('Generation request requires a nonnegative original source index')
        native_seed = self.adapter.seed
        if native_seed is None:
            raise ValueError('Native run seed is required for explicit sample requests')
        if isinstance(native_seed, bool) or not isinstance(native_seed, Integral):
            raise ValueError('Native run seed must be an integer')
        if int(native_seed) + int(index) != request.seed:
            raise ValueError('Native seed differs from the generation request')
        adapter_config = getattr(self.adapter, 'config', None)
        if adapter_config is not None and hasattr(adapter_config, 'get'):
            configured = adapter_config.get('seed')
            if configured is None or isinstance(configured, bool) or not isinstance(configured, Integral) or int(configured) + int(index) != request.seed:
                raise ValueError('Adapter configured seed differs from the generation request')
        sample_seed = getattr(self.adapter, '_sample_seed', None)
        if sample_seed is not None and sample_seed(request.sample) != request.seed:
            raise ValueError('Adapter configured seed differs from the generation request')
        generator = getattr(self.adapter, '_generator', None)
        config = getattr(generator, 'config', None)
        if config is not None and getattr(config, 'native_seed_policy', 'source_index') != 'source_index':
            raise ValueError('Direct native seed policy must be source_index')
        if config is not None and hasattr(config, 'seed'):
            seed = config.seed
            if seed is None or isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) + int(index) != request.seed:
                raise ValueError('Generator configured seed differs from the generation request')

    def generate_batch(self, requests):
        from rvcbench.utils.seeding import configure_seeds
        if self.adapter is None:
            raise RuntimeError('Backend.prepare() must precede generation')
        results = []
        for request in requests:
            self.validate_request(request)
            configure_seeds(request.seed, logger=None)
            path, elapsed = self.adapter.generate_sample(request.sample, output_dir=request.output_dir)
            results.append(GenerationResult(sample_id(request.sample), path, elapsed,
                self.timing_scope,
                **self.native_seed_metadata(request)))
        return results


class Qwen3Backend(SingleSampleBackend):
    timing_scope = 'qwen3_generate_excluding_prompt_encoding_and_io_v1'


class F5Backend(SingleSampleBackend):
    timing_scope = 'f5_sequential_inference_excluding_transcription_and_output_write_v1'


class FireRedTTS2Backend(SingleSampleBackend):
    timing_scope = 'fireredtts2_prompted_monologue_including_token_retries_and_cleanup_excluding_output_write_v1'


class BarkBackend(SingleSampleBackend):
    timing_scope = 'bark_reference_encoding_semantic_coarse_fine_and_codec_excluding_output_write_v1'

    def native_seed_metadata(self, request):
        return dict(**super().native_seed_metadata(request),
                    native_initialization=self.adapter._generator.initialization_metadata)


class StyleTTS2Backend(SingleSampleBackend):
    timing_scope = 'styletts2_synthesis_including_reference_style_and_diffusion_excluding_output_write_v1'

    def native_seed_metadata(self, request):
        return dict(**super().native_seed_metadata(request),
                    native_initialization=getattr(self.adapter._synthesizer, 'initialization_metadata', None))


class OpenVoiceBackend(SingleSampleBackend):
    timing_scope = 'openvoice_generation_including_conditioning_lazy_language_initialization_and_temporary_io_excluding_output_write_v1'


    def native_seed_metadata(self, request):
        return dict(**super().native_seed_metadata(request),
                    native_initialization=getattr(self.adapter._generator, 'initialization_metadata', None))


class CosyVoiceBackend(SingleSampleBackend):
    timing_scope = 'cosyvoice_native_inference_and_stream_collection_excluding_reference_loading_and_output_write_v1'


class SparkTTSBackend(SingleSampleBackend):
    timing_scope = 'sparktts_inference_including_enabled_transcript_retry_excluding_output_write_v1'

    def native_seed_metadata(self, request):
        variant = self.adapter._generator.last_conditioning_variant
        if variant not in ('prompt_audio_and_text', 'prompt_audio_only',
                           'prompt_audio_only_after_transcript_failure'):
            raise ValueError('SparkTTS did not record its conditioning variant')
        return dict(**super().native_seed_metadata(request), conditioning_variant=variant,
                    native_initialization=self.adapter._generator.initialization_metadata)


class ZipVoiceBackend(SingleSampleBackend):
    @property
    def timing_scope(self):
        mode = self.adapter._generator.execution_backend
        if mode not in ('native', 'cli'):
            raise ValueError('Unknown ZipVoice execution backend')
        if mode == 'cli':
            return 'zipvoice_cli_request_including_process_startup_model_loading_and_temporary_wav_io_v1'
        return 'zipvoice_native_generation_including_temporary_wav_io_excluding_output_write_v1'

    def native_seed_metadata(self, request):
        if self.adapter._generator.execution_backend == 'cli':
            return dict(native_seed=None, native_seed_policy='source_index_cli_argument',
                        native_requested_seed=int(request.seed))
        return super().native_seed_metadata(request)


class XttsBackend(SingleSampleBackend):
    timing_scope = 'xtts_synthesize_including_conditioning_excluding_output_write_v1'


class VoxCPMBackend(SingleSampleBackend):
    timing_scope = 'voxcpm_generate_including_native_retries_excluding_output_write_v1'

    def native_seed_metadata(self, request):
        generator = self.adapter._generator
        requested = generator.last_native_requested_seed
        if requested != request.seed:
            raise ValueError('VoxCPM native requested seed differs from the generation request')
        return dict(native_seed=generator.last_native_seed,
                    native_seed_policy=generator.config.native_seed_policy,
                    native_requested_seed=requested)


class WorkerSampleBackend(SingleSampleBackend):
    def native_seed_metadata(self, request):
        generator = self.adapter._generator
        seed = self.adapter._sample_seed(request.sample)
        if seed != request.seed or generator.last_native_requested_seed != seed or generator.last_native_seed != seed:
            raise ValueError(f'{self.adapter.MODEL_NAME} worker did not confirm the native sample seed')
        return dict(native_seed=seed, native_seed_policy='source_index', native_requested_seed=seed)


class IndexTTSBackend(WorkerSampleBackend):
    timing_scope = 'indextts_worker_request_including_ipc_inference_and_output_validation_v1'


class MaskGCTBackend(WorkerSampleBackend):
    timing_scope = 'maskgct_worker_request_including_ipc_inference_and_output_validation_v1'


def create_backend(conf, dataset, device, logger):
    backend = {
        'qwen3_tts': Qwen3Backend,
        'f5_tts': F5Backend,
        'voxcpm': VoxCPMBackend,
        'index_tts': IndexTTSBackend,
        'maskgct': MaskGCTBackend,
        'xtts': XttsBackend,
        'zipvoice': ZipVoiceBackend,
        'sparktts': SparkTTSBackend,
        'cosyvoice': CosyVoiceBackend,
        'openvoice': OpenVoiceBackend,
        'styletts2': StyleTTS2Backend,
        'fireredtts2': FireRedTTS2Backend,
        'bark_voice_clone': BarkBackend,
    }.get(conf.vc.model, LegacyAdversaryBackend)
    return backend(conf, dataset, device, logger)
