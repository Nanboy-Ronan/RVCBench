"""MOSS-TTSD transcript selection and checkpoint loading integrity."""
import logging
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.models.moss_ttsd.generator import (
    MossTTSDGenerator, MossTTSDGeneratorConfig, _FullSequenceModel, _validate_loaded_parameters, _restore_declared_ties,
)


@pytest.mark.parametrize('manifest_text', [False, True])
def test_reference_transcript_source_is_explicit(tmp_path, manifest_text):
    config = MossTTSDGeneratorConfig(code_path=tmp_path, model_path=str(tmp_path),
                                    spt_config_path=tmp_path, spt_checkpoint_path=tmp_path,
                                    use_prompt_transcript=manifest_text)
    with patch.dict('sys.modules', {'whisper': None}):
        generator = MossTTSDGenerator(config, torch.device('cpu'), logging.getLogger())
    assert generator.whisper_model is None
    generator._model_ready = True
    generator.whisper_model = Mock(device=torch.device('cpu'))
    generator.whisper_model.transcribe.return_value = {'text': 'ASR reference'}
    process = Mock(return_value=(['target'], [{'audio_data': np.zeros(80), 'sample_rate': 24000}]))
    generator._generation_utils = SimpleNamespace(process_batch=process)
    generator.generate('target', tmp_path / 'reference.wav', 'Manifest reference', 42)
    item = process.call_args.kwargs['batch_items'][0]
    assert item['prompt_text_speaker1'] == ('Manifest reference' if manifest_text else 'ASR reference')
    assert generator.whisper_model.transcribe.call_count == (0 if manifest_text else 1)
    generator.close()
    assert generator.whisper_model is None and not generator._model_ready


def test_native_sequence_keeps_reference_prefix_for_legacy_batch_slicing():
    model = Mock()
    model.generate.return_value = torch.zeros(1, 20, 8)
    _FullSequenceModel(model).generate(input_ids=torch.zeros(1, 10, 8))
    assert model.generate.call_args.kwargs['output_only'] is False


def test_missing_output_head_is_rejected_unless_it_is_actually_tied():
    model = torch.nn.Module()
    model.source = torch.nn.Linear(2, 2, bias=False)
    model.head = torch.nn.Linear(2, 2, bias=False)
    model._tied_weights_keys = {'head.weight': 'source.weight'}
    loading = {'missing_keys': ['head.weight']}
    with pytest.raises(RuntimeError, match='parameters were not loaded'):
        _validate_loaded_parameters(model, loading)
    _restore_declared_ties(model, loading)
    assert model.head.weight is model.source.weight
    _validate_loaded_parameters(model, loading)


def test_native_checkpoint_dispatches_declared_class_and_restores_pad(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'auto_map': {'AutoModel': 'native.Model'}}))
    config = MossTTSDGeneratorConfig(code_path=tmp_path, model_path=str(tmp_path),
        spt_config_path=tmp_path, spt_checkpoint_path=tmp_path, use_prompt_transcript=True)
    generator = MossTTSDGenerator(config, torch.device('cpu'), logging.getLogger())
    legacy_loader = Mock(side_effect=AssertionError('Native checkpoint must not use Asteroid loader'))
    generator._generation_utils = SimpleNamespace(load_model=legacy_loader)
    native = torch.nn.Linear(2, 2)
    codec = torch.nn.Linear(2, 2)
    loaded_config = SimpleNamespace()  # modern config discards generation fields
    xy = SimpleNamespace(XY_Tokenizer=SimpleNamespace(load_from_checkpoint=Mock(return_value=codec)))
    with patch('src.models.moss_ttsd.generator._check_native_runtime'), \
            patch('transformers.AutoTokenizer.from_pretrained', return_value=SimpleNamespace(pad_token_id=152694)), \
            patch('transformers.AutoConfig.from_pretrained', return_value=loaded_config), \
            patch('transformers.AutoModel.from_pretrained', return_value=(native, {})) as loader, \
            patch.dict('sys.modules', {'XY_Tokenizer.xy_tokenizer.model': xy, 'whisper': None}):
        generator.load_model()
    assert generator._model is native
    assert loader.call_args.kwargs['config'].pad_token_id == 152694
    assert loader.call_args.kwargs['output_loading_info'] is True
    assert generator.whisper_model is None
    legacy_loader.assert_not_called()


def test_unused_reference_asr_does_not_change_manifest_text_asset_identity(tmp_path):
    from omegaconf import OmegaConf
    from src.benchmark.model_assets import resolve_model_assets
    asr = tmp_path / 'asr.pt'
    asr.write_bytes(b'first')
    config = OmegaConf.create({'vc': {'model': 'moss_ttsd'}, 'adversary': {
        'use_prompt_transcript': True, 'reference_asr_model': str(asr)}})
    _, reference, first = resolve_model_assets(config)
    assert 'reference_asr_model' not in reference['assets']
    asr.write_bytes(b'second')
    assert resolve_model_assets(config)[2] == first
    config.adversary.use_prompt_transcript = False
    _, reference, fingerprint = resolve_model_assets(config)
    assert 'reference_asr_model' in reference['assets']
    asr.write_bytes(b'third')
    assert resolve_model_assets(config)[2] != fingerprint
