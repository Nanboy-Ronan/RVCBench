import json
from pathlib import Path
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from test_benchmark import setup_run
from test_reference_stages import stage_directory
from rvcbench.benchmark.artifacts import file_hash
from rvcbench.benchmark.denoise_stage import denoise_dns64


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
    with patch('rvcbench.benchmark.denoise_stage._load_dns64', return_value=(model, {})):
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
    with patch('rvcbench.benchmark.denoise_stage._load_dns64', side_effect=AssertionError('must not load')):
        with pytest.raises(FileNotFoundError):
            denoise_dns64(*args)
    assert not args[-1].exists()


def test_invalid_model_output_leaves_failed_stage(stage_inputs):
    *args, _, _ = stage_inputs
    with patch('rvcbench.benchmark.denoise_stage._load_dns64', return_value=(Echo(invalid=True), {})):
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


@pytest.mark.parametrize('corruption', [None, 'content', 'identity', 'source', 'environment'])
def test_worker_result_validated_against_request_and_actual_files(stage_inputs, corruption):
    import soundfile as sf
    *args, _, _ = stage_inputs
    real_run = subprocess.run
    def fake_run(command, **kwargs):
        if '--request' not in command:
            return real_run(command, **kwargs)
        request_path = Path(command[command.index('--request') + 1])
        result_path = Path(command[command.index('--result') + 1])
        request = json.loads(request_path.read_text())
        rows = []
        for item in request['items']:
            audio, rate = sf.read(item['input_path'])
            destination = Path(item['output_path'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, audio, rate)
            rows.append({'sample_id': item['sample_id'], 'output_path': str(destination),
                         'sha256': file_hash(destination), 'frames': len(audio), 'rate': rate})
        worker = Path(command[1])
        result = {'schema_version': 1, 'status': 'complete', 'rows': rows,
                  'request_sha256': file_hash(request_path), 'weights_sha256': request['weights_sha256'],
                  'runtime': {'worker_sha256': file_hash(worker),
                      'kernel_sha256': file_hash(worker.parent.parent / 'src/rvcbench/models/dns64_kernel.py'),
                      'packages': {'test': '1'}, 'executable': command[0]}}
        if corruption == 'content':
            rows[0]['sha256'] = 'wrong'
        elif corruption == 'identity':
            rows[0]['sample_id'] = 'wrong'
        elif corruption == 'source':
            result['runtime']['kernel_sha256'] = 'wrong'
        elif corruption == 'environment':
            alias = args[-1].parent / 'other-environment' / 'bin' / 'python'
            alias.parent.mkdir(parents=True)
            alias.symlink_to(command[0])
            assert alias.resolve() == Path(command[0]).resolve()
            result['runtime']['executable'] = str(alias)
        result_path.write_text(json.dumps(result))
    with patch('rvcbench.benchmark.denoise_stage.subprocess.run', side_effect=fake_run):
        if corruption:
            with pytest.raises(ValueError, match='worker'):
                denoise_dns64(*args, runtime_python=sys.executable)
        else:
            result = denoise_dns64(*args, runtime_python=sys.executable)
            assert result['verified'] == 2 and result['status'] == 'complete'
            assert len(result['worker_result']['rows']) == 1  # Shared reference, distinct pairs.
    if corruption:
        assert json.loads((args[-1] / 'stage_manifest.json').read_text())['status'] == 'failed'


def test_worker_timeout_stops_process_and_marks_stage_failed(stage_inputs):
    *args, _, _ = stage_inputs
    stub = args[-1].parent / 'sleep-worker'
    pid_file = stub.with_suffix('.pid')
    stub.write_text('#!' + sys.executable + '\nimport os,time\n'
                    + f'open({str(pid_file)!r}, "w").write(str(os.getpid()))\ntime.sleep(30)\n')
    stub.chmod(0o755)
    with pytest.raises(subprocess.TimeoutExpired):
        denoise_dns64(*args, runtime_python=stub, timeout_seconds=1)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    result = json.loads((args[-1] / 'stage_manifest.json').read_text())
    assert result['status'] == 'failed' and result['verified'] == 0
