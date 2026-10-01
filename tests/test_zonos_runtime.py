"""Single-rank engine ownership, sample seeding and CUDA state restoration."""
from contextlib import contextmanager
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.models.zonos2.generator import Zonos2Generator, Zonos2GeneratorConfig


@pytest.mark.parametrize('fail', [False, True])
def test_sample_seed_and_stream_restore_on_success_and_failure(fail):
    generator = Zonos2Generator(Zonos2GeneratorConfig(seed=42), 'cuda:0', logging.getLogger())
    state = {'stream': 'caller'}
    @contextmanager
    def scoped_stream(stream):
        previous = state['stream']
        state['stream'] = stream
        try:
            yield
        finally:
            state['stream'] = previous
    seeds = []
    def generate(text, sampling, **kwargs):
        assert state['stream'] == 'engine'
        seeds.append(sampling.seed)
        if fail:
            raise RuntimeError('generation fixture')
        return {'audio': np.zeros(10, dtype=np.float32).tobytes(), 'sample_rate': 44100}
    shutdown = Mock()
    generator._tts = SimpleNamespace(stream='engine', embed_speaker_file=Mock(return_value=[]),
                                    generate_one=generate, shutdown=shutdown)
    generator._vocoder_module = SimpleNamespace(_dac_model=None)
    generator._sampling_params_cls = SimpleNamespace
    generator._model_ready = True
    with patch('torch.cuda.device', side_effect=lambda _: scoped_stream(state['stream'])), \
            patch('torch.cuda.stream', side_effect=scoped_stream), \
            patch('torch.distributed.is_initialized', return_value=False):
        if fail:
            with pytest.raises(RuntimeError, match='generation fixture'):
                generator.generate('text', 'ref.wav', 'reference', 73)
        else:
            assert generator.generate('text', 'ref.wav', 'reference', 73)[1] == 44100
        assert state['stream'] == 'caller'
        generator.close()
        generator.close()
    assert seeds == [115] and shutdown.call_count == 1
    assert state['stream'] == 'caller' and generator._tts is None


@pytest.mark.parametrize('device', ['cpu', 'cuda:2'])
def test_reject_unsupported_device_before_engine_import(device):
    generator = Zonos2Generator(Zonos2GeneratorConfig(), device, logging.getLogger())
    with pytest.raises(ValueError, match='logical cuda:0'):
        generator.load_model()


def test_foreign_group_is_preserved_and_owned_partial_group_is_closed():
    generator = Zonos2Generator(Zonos2GeneratorConfig(), 'cuda:0', logging.getLogger())
    with patch('torch.distributed.is_initialized', return_value=True), \
            patch('torch.distributed.destroy_process_group') as destroy:
        with pytest.raises(RuntimeError, match='own process group'):
            generator.load_model()
        generator.close()
        destroy.assert_not_called()
        generator._owns_group = True  # A constructor failed after creating the group.
        generator.close()
        destroy.assert_called_once()


@pytest.mark.parametrize('fail', [False, True])
def test_scoped_scheduler_port_restores_upstream_class(fail):
    class Scheduler:
        distributed_addr = 'tcp://127.0.0.1:23333'
    module = SimpleNamespace(SchedulerConfig=Scheduler)
    generator = Zonos2Generator(Zonos2GeneratorConfig(distributed_port=29333), 'cuda:0', logging.getLogger())
    try:
        with generator._scheduler_config(module):
            assert module.SchedulerConfig().distributed_addr == 'tcp://127.0.0.1:29333'
            if fail:
                raise RuntimeError('fixture')
    except RuntimeError:
        assert fail
    assert module.SchedulerConfig is Scheduler
