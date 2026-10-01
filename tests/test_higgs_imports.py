"""Configured Higgs entrypoints must survive unrelated examples packages."""
import logging
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import torch
import pytest

from src.models.higgs_audio.generator import HiggsAudioGenerator, HiggsAudioGeneratorConfig


def test_incompatible_transformers_fails_before_upstream_import(tmp_path):
    (tmp_path / 'requirements.txt').write_text('transformers>=4.45.1,<4.47.0\n')
    generator = HiggsAudioGenerator(HiggsAudioGeneratorConfig(code_path=tmp_path, model_path='fixture'),
                                   torch.device('cpu'), logging.getLogger())
    with patch('src.models.higgs_audio.generator.version', return_value='4.57.3'):
        with pytest.raises(RuntimeError, match='requires transformers'):
            generator._ensure_imports()
    assert generator._generation_mod is None


def test_higgs_loads_configured_file_despite_examples_package_collision(tmp_path):
    source = tmp_path / 'examples'
    source.mkdir()
    entrypoint = source / 'generation.py'
    entrypoint.write_text('''from dataclasses import dataclass
@dataclass
class Message:
    role: str
AudioContent = object
prepare_chunk_text = object
load_higgs_audio_tokenizer = object
HiggsAudioModelClient = object
''')
    generator = HiggsAudioGenerator(HiggsAudioGeneratorConfig(code_path=tmp_path, model_path='fixture'),
                                   torch.device('cpu'), logging.getLogger())
    original_path = list(sys.path)
    try:
        with patch.dict(os.environ), patch.dict(sys.modules, {'examples': ModuleType('examples')}):
            # Prevent the optional upstream availability workaround from mutating
            # an installed Transformers module in this unit test.
            with patch.dict(sys.modules, {'transformers.utils.import_utils': None}):
                generator._ensure_imports()
            assert Path(generator._generation_mod.__file__) == entrypoint
            assert generator._Message('user').role == 'user'
            assert sys.modules['examples'].__name__ == 'examples'
    finally:
        sys.path[:] = original_path
        if generator._generation_mod is not None:
            sys.modules.pop(generator._generation_mod.__name__, None)
