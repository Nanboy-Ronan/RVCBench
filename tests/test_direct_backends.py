from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from src.benchmark import backends
from src.benchmark.artifacts import sample_id
from src.benchmark.backends import GenerationRequest, LegacyAdversaryBackend, create_backend
from src.datasets.zero_shot import ZeroShotSample


DIRECT = {
    'voxcpm': backends.VoxCPMBackend,
    'index_tts': backends.IndexTTSBackend,
    'maskgct': backends.MaskGCTBackend,
    'xtts': backends.XttsBackend,
    'zipvoice': backends.ZipVoiceBackend,
    'sparktts': backends.SparkTTSBackend,
    'cosyvoice': backends.CosyVoiceBackend,
    'openvoice': backends.OpenVoiceBackend,
    'styletts2': backends.StyleTTS2Backend,
    'fireredtts2': backends.FireRedTTS2Backend,
    'bark_voice_clone': backends.BarkBackend,
}
RUN_SEED, INDEX = 100, 3
SEED = RUN_SEED + INDEX


def make_sample(index=INDEX):
    return ZeroShotSample(speaker_id='speaker', index=index, prompt_path=Path('prompt.wav'), prompt_text='prompt',
                          prompt_language='en', target_path=Path('target.wav'), target_text='target',
                          target_language='en', extra={'pair_id': 'pair'})


def make_request(seed=SEED, index=INDEX):
    return GenerationRequest(make_sample(index), seed, Path('out'))


class Adapter:
    MODEL_NAME = 'Fixture'

    def __init__(self, seed=RUN_SEED, generator=None, failure=None):
        self.seed, self.config, self.failure, self.calls = seed, {'seed': seed}, failure, []
        self._generator = self._synthesizer = generator

    def generate_sample(self, sample, *, output_dir):
        self.calls.append(sample)
        if self.failure:
            raise self.failure
        return Path(output_dir) / 'cloned.wav', 1.5


def make_backend(cls=backends.XttsBackend, **adapter):
    backend = cls(None, None, 'cpu', None)
    backend.adapter = Adapter(**adapter)
    return backend


def native_generator(**fields):
    defaults = dict(config=SimpleNamespace(seed=RUN_SEED, native_seed_policy='source_index'),
                    initialization_metadata={'complete': True}, last_conditioning_variant='prompt_audio_and_text',
                    execution_backend='native', last_native_seed=SEED, last_native_requested_seed=SEED)
    return SimpleNamespace(**{**defaults, **fields})


@pytest.mark.parametrize('model,cls', sorted(DIRECT.items()))
def test_migrated_models_use_their_direct_backend(model, cls):
    conf = OmegaConf.create({'vc': {'model': model}})
    assert type(create_backend(conf, None, 'cpu', None)) is cls
    assert issubclass(cls, backends.SingleSampleBackend)


def test_unmigrated_models_keep_the_legacy_bridge():
    conf = OmegaConf.create({'vc': {'model': 'not_migrated'}})
    assert type(create_backend(conf, None, 'cpu', None)) is LegacyAdversaryBackend


def test_generation_requires_prepare():
    with pytest.raises(RuntimeError, match='prepare'):
        backends.XttsBackend(None, None, 'cpu', None).generate_batch([make_request()])


def test_valid_request_records_source_index_seed():
    backend = make_backend()
    request = make_request()
    result, = backend.generate_batch([request])
    assert backend.adapter.calls == [request.sample]
    assert (result.sample_id, result.path, result.elapsed_sec) == (sample_id(request.sample), Path('out/cloned.wav'), 1.5)
    assert result.timing_scope == backends.XttsBackend.timing_scope
    assert (result.native_seed, result.native_seed_policy, result.native_requested_seed) == (SEED, 'source_index', SEED)


@pytest.mark.parametrize('request_fields,adapter_fields,message', [
    (dict(seed=True), {}, 'seed must be an integer'),
    (dict(seed=float(SEED)), {}, 'seed must be an integer'),
    (dict(seed=2 ** 64), {}, 'outside the supported Torch range'),
    (dict(index=-1, seed=RUN_SEED - 1), {}, 'nonnegative original source index'),
    (dict(index=True), {}, 'nonnegative original source index'),
    ({}, dict(seed=None), 'Native run seed is required'),
    ({}, dict(seed=True), 'Native run seed must be an integer'),
    ({}, dict(seed=RUN_SEED + 1), 'Native seed differs'),
    (dict(seed=SEED + 1), {}, 'Native seed differs'),
])
def test_invalid_seed_requests_are_rejected_before_generation(request_fields, adapter_fields, message):
    backend = make_backend(**adapter_fields)
    with pytest.raises(ValueError, match=message):
        backend.generate_batch([make_request(**request_fields)])
    assert backend.adapter.calls == []


def test_adapter_configuration_seed_must_match_the_request():
    backend = make_backend()
    backend.adapter.config = {'seed': RUN_SEED + 1}
    with pytest.raises(ValueError, match='Adapter configured seed differs'):
        backend.generate_batch([make_request()])
    backend.adapter.config = {}
    with pytest.raises(ValueError, match='Adapter configured seed differs'):
        backend.generate_batch([make_request()])
    assert backend.adapter.calls == []


@pytest.mark.parametrize('config,message', [
    (SimpleNamespace(seed=RUN_SEED, native_seed_policy='fixed'), 'policy must be source_index'),
    (SimpleNamespace(seed=RUN_SEED + 1), 'Generator configured seed differs'),
    (SimpleNamespace(seed=None), 'Generator configured seed differs'),
])
def test_generator_configuration_must_match_the_request(config, message):
    backend = make_backend(generator=native_generator(config=config))
    with pytest.raises(ValueError, match=message):
        backend.generate_batch([make_request()])
    assert backend.adapter.calls == []


@pytest.mark.parametrize('model,cls', sorted(DIRECT.items()))
def test_generation_failure_propagates_without_a_result(model, cls):
    backend = make_backend(cls, generator=native_generator(), failure=RuntimeError('native generation failed'))
    backend.adapter._sample_seed = lambda sample: RUN_SEED + sample.index
    with pytest.raises(RuntimeError, match='native generation failed'):
        backend.generate_batch([make_request()])
    assert len(backend.adapter.calls) == 1


@pytest.mark.parametrize('cls', [backends.IndexTTSBackend, backends.MaskGCTBackend])
@pytest.mark.parametrize('fields', [dict(last_native_seed=None), dict(last_native_requested_seed=None),
                                    dict(last_native_seed=SEED + 1, last_native_requested_seed=SEED + 1)])
def test_worker_backends_require_seed_confirmation(cls, fields):
    backend = make_backend(cls, generator=native_generator(**fields))
    backend.adapter._sample_seed = lambda sample: RUN_SEED + sample.index
    with pytest.raises(ValueError, match='did not confirm the native sample seed'):
        backend.generate_batch([make_request()])


@pytest.mark.parametrize('cls', [backends.IndexTTSBackend, backends.MaskGCTBackend])
def test_worker_backends_record_confirmed_seed(cls):
    backend = make_backend(cls, generator=native_generator())
    backend.adapter._sample_seed = lambda sample: RUN_SEED + sample.index
    result, = backend.generate_batch([make_request()])
    assert (result.native_seed, result.native_seed_policy, result.native_requested_seed) == (SEED, 'source_index', SEED)
    backend.adapter._sample_seed = lambda sample: RUN_SEED
    with pytest.raises(ValueError, match='Adapter configured seed differs'):
        backend.generate_batch([make_request()])


@pytest.mark.parametrize('variant', ['prompt_audio_and_text', 'prompt_audio_only',
                                     'prompt_audio_only_after_transcript_failure'])
def test_sparktts_records_conditioning_variant(variant):
    backend = make_backend(backends.SparkTTSBackend, generator=native_generator(last_conditioning_variant=variant))
    result, = backend.generate_batch([make_request()])
    assert result.conditioning_variant == variant
    assert result.native_initialization == {'complete': True}


@pytest.mark.parametrize('variant', [None, 'text_only'])
def test_sparktts_requires_a_known_conditioning_variant(variant):
    backend = make_backend(backends.SparkTTSBackend, generator=native_generator(last_conditioning_variant=variant))
    with pytest.raises(ValueError, match='did not record its conditioning variant'):
        backend.generate_batch([make_request()])


@pytest.mark.parametrize('cls', [backends.BarkBackend, backends.StyleTTS2Backend, backends.OpenVoiceBackend])
def test_checkpoint_initialization_is_recorded(cls):
    backend = make_backend(cls, generator=native_generator(initialization_metadata={'missing_keys': 0}))
    result, = backend.generate_batch([make_request()])
    assert result.native_initialization == {'missing_keys': 0}


def test_zipvoice_scope_and_seed_policy_follow_the_execution_backend():
    native, = make_backend(backends.ZipVoiceBackend, generator=native_generator()).generate_batch([make_request()])
    cli, = make_backend(backends.ZipVoiceBackend,
                        generator=native_generator(execution_backend='cli')).generate_batch([make_request()])
    assert native.timing_scope != cli.timing_scope
    assert 'process_startup' in cli.timing_scope and 'process_startup' not in native.timing_scope
    assert (native.native_seed, native.native_seed_policy) == (SEED, 'source_index')
    assert (cli.native_seed, cli.native_seed_policy, cli.native_requested_seed) == (None, 'source_index_cli_argument', SEED)
    with pytest.raises(ValueError, match='Unknown ZipVoice execution backend'):
        make_backend(backends.ZipVoiceBackend,
                     generator=native_generator(execution_backend='server')).generate_batch([make_request()])


def test_voxcpm_reports_effective_seed_and_rejects_a_different_request():
    backend = make_backend(backends.VoxCPMBackend, generator=native_generator(last_native_seed=SEED + 9))
    result, = backend.generate_batch([make_request()])
    assert (result.native_seed, result.native_requested_seed) == (SEED + 9, SEED)
    backend = make_backend(backends.VoxCPMBackend, generator=native_generator(last_native_requested_seed=SEED + 1))
    with pytest.raises(ValueError, match='native requested seed differs'):
        backend.generate_batch([make_request()])
