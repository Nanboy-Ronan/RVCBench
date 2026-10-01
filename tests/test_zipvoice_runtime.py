"""ZipVoice model retention, CLI isolation and sample seed regressions."""
import logging
import atexit
import importlib
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

from src.models.zipvoice.generator import ZipVoiceGenerator, ZipVoiceGeneratorConfig


def test_unknown_tokenizer_fails_before_loading_weights(tmp_path):
    with pytest.raises(ValueError, match='Unsupported ZipVoice tokenizer'):
        ZipVoiceGenerator(ZipVoiceGeneratorConfig(code_path=tmp_path, tokenizer='emila'),
                          torch.device('cpu'), logging.getLogger())


def test_native_inference_retains_models_and_resets_sample_seed(tmp_path):
    generator = ZipVoiceGenerator(ZipVoiceGeneratorConfig(code_path=tmp_path, seed=42),
                                 torch.device('cpu'), logging.getLogger())
    generator._model_ready = True
    model, vocoder = object(), object()
    generator.model, generator.vocoder = model, vocoder
    generator._tokenizer = generator._feature_extractor = object()
    calls = []
    def inference(**kwargs):
        calls.append(kwargs)
        sf.write(kwargs['save_path'], np.random.uniform(-.1, .1, 160), 24000)
    generator._generate_sentence = inference
    prompt = tmp_path / 'prompt.wav'
    sf.write(prompt, np.zeros(160), 24000)
    with patch('subprocess.run', side_effect=AssertionError('Native inference must not reload in a CLI')):
        first, _ = generator.generate(text='hello', prompt_wav=prompt, prompt_text='voice', sample_index=37)
        second, _ = generator.generate(text='hello', prompt_wav=prompt, prompt_text='voice', sample_index=37)
    assert np.array_equal(first, second)
    assert all(c['model'] is model and c['vocoder'] is vocoder for c in calls)
    assert calls[0]['num_step'] == 16 and calls[0]['guidance_scale'] == 1.0
    assert all(not Path(c['save_path']).exists() for c in calls)
    threads = torch.get_num_threads()
    generator._previous_num_threads = threads
    torch.set_num_threads(1)
    try:
        generator.close()
        assert torch.get_num_threads() == threads
        assert generator.model is None and generator.vocoder is None
    finally:
        torch.set_num_threads(threads)


def test_cli_preflight_does_not_load_parent_models(tmp_path):
    generator = ZipVoiceGenerator(ZipVoiceGeneratorConfig(code_path=tmp_path,
                                 runtime_python=Path(sys.executable)),
                                 torch.device('cpu'), logging.getLogger())
    assert generator.execution_backend == 'cli'
    with patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stderr='')) as command, \
            patch('importlib.import_module', side_effect=AssertionError('No parent model import')):
        generator.ensure_model()
    assert command.call_args.args[0][0] == sys.executable
    assert generator.model is None and generator.vocoder is None


def test_native_backend_rejects_foreign_python(tmp_path):
    foreign = tmp_path / 'python'
    foreign.touch()
    with pytest.raises(ValueError, match='current interpreter'):
        ZipVoiceGenerator(ZipVoiceGeneratorConfig(code_path=tmp_path, execution_backend='native',
                          runtime_python=foreign), torch.device('cpu'), logging.getLogger())


@pytest.mark.parametrize('module,name', [('zipvoice', 'ZipVoice'), ('maskgct', 'MaskGCT'), ('index_tts', 'IndexTTS')])
def test_runtime_interpreter_preserves_real_venv_prefix(tmp_path, module, name):
    env = tmp_path / 'isolated'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(env)], check=True)
    interpreter = env / 'bin/python'
    assert interpreter.is_symlink()
    worker = tmp_path / 'worker.py'
    worker.write_text('')
    config_file = tmp_path / 'config.json'
    config_file.write_text('{}')
    source = importlib.import_module(f'src.models.{module}.generator')
    config_class = getattr(source, name + 'GeneratorConfig')
    kwargs = {'code_path': tmp_path, 'runtime_python': interpreter}
    if module != 'zipvoice':
        kwargs.update(config_path=config_file, worker_script_path=worker)
    if module == 'index_tts':
        kwargs['model_dir'] = tmp_path
    generator = getattr(source, name + 'Generator')(config_class(**kwargs), torch.device('cpu'), logging.getLogger())
    try:
        result = subprocess.run([str(generator.config.runtime_python), '-c', 'import sys; print(sys.prefix)'],
                                check=True, capture_output=True, text=True)
        assert result.stdout.strip() == str(env)
        from src.benchmark.fingerprints import worker_environment
        assert worker_environment(generator.config.runtime_python)['executable'] == str(interpreter)
    finally:
        generator.close()
        atexit.unregister(generator.close)
