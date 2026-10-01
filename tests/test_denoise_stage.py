import json
from unittest.mock import patch

import pytest
import torch

from test_benchmark import setup_run
from test_reference_stages import stage_directory
from src.benchmark.artifacts import file_hash
from src.benchmark.denoise_stage import denoise_dns64


@pytest.fixture
def stage_inputs(setup_run):
    conf, dataset, _, tmp_path = setup_run
    root, clean, staged = stage_directory(dataset, tmp_path)
    rows = json.loads((dataset._dataset_root / 'metadata.json').read_text())[:2]
    for row in rows:
        row['dataset_name'] = 'LibriTTS'
        row['manifest_variant'] = 'speaker'
    subset = tmp_path / 'subset.json'
    subset.write_text(json.dumps(rows))
    weights = tmp_path / 'weights.th'
    weights.write_bytes(b'test-weight-placeholder')
    return dataset._dataset_root, subset, root, weights, tmp_path / 'enhanced', clean, staged


class Echo(torch.nn.Module):
    sample_rate = 16000

    def __init__(self, invalid=False):
        super().__init__()
        self.calls = 0
        self.invalid = invalid

    def forward(self, audio):
        self.calls += 1
        return audio * float('nan') if self.invalid else audio


def test_stage_preserves_pair_rows_while_enhancing_shared_reference_once(stage_inputs):
    *args, clean, staged = stage_inputs
    model = Echo()
    with patch('src.benchmark.denoise_stage._load_dns64', return_value=(model, {})):
        result = denoise_dns64(*args)
    assert result['status'] == 'complete' and result['verified'] == 2
    assert model.calls == 1
    assert len({r['sample_id'] for r in result['rows']}) == 2
    assert len({r['reference_path'] for r in result['rows']}) == 1
    assert all(r['clean_prompt_sha256'] == file_hash(clean) for r in result['rows'])
    assert all(r['input_reference_sha256'] == file_hash(staged) for r in result['rows'])
    assert json.loads((args[-1] / 'stage_manifest.json').read_text()) == result


def test_missing_reference_fails_before_model_loading(stage_inputs):
    *args, _, staged = stage_inputs
    staged.unlink()
    with patch('src.benchmark.denoise_stage._load_dns64', side_effect=AssertionError('must not load')):
        with pytest.raises(FileNotFoundError):
            denoise_dns64(*args)
    assert not args[-1].exists()


def test_invalid_model_output_leaves_failed_stage(stage_inputs):
    *args, _, _ = stage_inputs
    with patch('src.benchmark.denoise_stage._load_dns64', return_value=(Echo(invalid=True), {})):
        with pytest.raises(ValueError, match='Invalid DNS64 output'):
            denoise_dns64(*args)
    result = json.loads((args[-1] / 'stage_manifest.json').read_text())
    assert result['status'] == 'failed' and result['verified'] == 0


@pytest.mark.parametrize('dry', [-1, 2, float('nan')])
def test_invalid_mixing_protocol_rejected_before_output_creation(stage_inputs, dry):
    *args, _, _ = stage_inputs
    with pytest.raises(ValueError, match='dry'):
        denoise_dns64(*args, dry=dry)
    assert not args[-1].exists()
