"""Actual conditioning, seed and waveform contracts without optional runtimes."""
from contextlib import contextmanager
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from rvcbench.models.voxcpm import VoxCPMGenerator, VoxCPMGeneratorConfig
from test_benchmark import setup_run


def prepared(wave, v2=True, effective_seed=115):
    generator = VoxCPMGenerator(VoxCPMGeneratorConfig(seed=42), 'cuda:0', logging.getLogger())
    generator._pipeline = SimpleNamespace(generate=Mock(return_value=wave),
        tts_model=SimpleNamespace(sample_rate=48000, last_successful_seed=effective_seed))
    generator._supports_reference_audio = v2
    generator._model_ready = True
    return generator


@pytest.mark.parametrize('fail', [False, True])
def test_request_seed_original_index_and_effective_retry_seed_are_distinct(fail):
    generator = prepared(np.zeros(24), effective_seed=116)
    state = {'device': 'caller'}
    @contextmanager
    def device_scope(device):
        previous = state['device']
        state['device'] = device
        try:
            yield
        finally:
            state['device'] = previous
    def generate(**kwargs):
        assert state['device'] == torch.device('cuda:0')
        assert kwargs['seed'] == 115
        assert kwargs['reference_wav_path'] == 'reference.wav'
        assert kwargs['prompt_wav_path'] == 'prompt.wav' and kwargs['prompt_text'] == 'reference text'
        if fail:
            raise RuntimeError('generation fixture')
        return np.zeros(24)
    generator._pipeline.generate.side_effect = generate
    with patch('torch.cuda.device', side_effect=device_scope):
        if fail:
            with pytest.raises(RuntimeError, match='generation fixture'):
                generator.generate(text='target', reference_wav_path='reference.wav',
                    prompt_wav_path='prompt.wav', prompt_text='reference text', sample_index=73)
        else:
            wave, rate = generator.generate(text='target', reference_wav_path='reference.wav',
                prompt_wav_path='prompt.wav', prompt_text='reference text', sample_index=73)
            assert wave.shape == (24,) and rate == 48000
            assert generator.last_native_seed == 116 and generator.last_native_requested_seed == 115
    assert state['device'] == 'caller'
    generator.close()
    assert generator._pipeline is generator.model is None and not generator._model_ready


@pytest.mark.parametrize('wave', [np.array([]), np.array([np.nan]), np.array([np.inf]), np.zeros((2, 3))])
def test_invalid_waveform_is_rejected_without_silent_repair(wave):
    with patch('torch.cuda.device'), pytest.raises(RuntimeError):
        prepared(wave).generate(text='target')


def test_unsupported_reference_audio_and_incomplete_prompt_are_rejected():
    generator = prepared(np.zeros(24), v2=False)
    with pytest.raises(ValueError, match='does not support'):
        generator.generate(text='target', reference_wav_path='reference.wav')
    with pytest.raises(ValueError, match='supplied together'):
        generator.generate(text='target', prompt_wav_path='prompt.wav')
    generator._pipeline.generate.assert_not_called()


def test_load_failure_restores_device_and_closes_partial_state():
    generator = prepared(np.zeros(24))
    generator.close()
    module = SimpleNamespace(VoxCPM=SimpleNamespace(from_pretrained=Mock(side_effect=RuntimeError('load fixture'))))
    with patch('torch.cuda.device') as scope, patch('importlib.import_module', return_value=module):
        with pytest.raises(RuntimeError, match='load fixture'):
            generator.ensure_model()
    assert scope.return_value.__exit__.call_count == 1
    assert generator._pipeline is generator.model is None and not generator._model_ready


def test_adapter_propagates_seed_index_and_does_not_swallow_cuda_errors(setup_run, tmp_path):
    from dataclasses import replace
    from rvcbench.adversary.voxcpm_ots import VoxCPMZeroShotAdversary
    from rvcbench.benchmark.backends import SampleView
    conf, dataset, _, _ = setup_run
    adapter = VoxCPMZeroShotAdversary(conf, conf.dataset, 'cpu', logging.getLogger())
    adapter._generator = SimpleNamespace(generate=Mock(side_effect=RuntimeError('CUDA error: illegal memory access')))
    sample = replace(dataset.get_zero_shot_samples()[0], index=73)
    with pytest.raises(RuntimeError, match='CUDA error'):
        adapter.attack(output_path=tmp_path / 'voxcpm', dataset=SampleView(dataset, sample))
    assert adapter._generator.generate.call_args.kwargs['sample_index'] == 73
