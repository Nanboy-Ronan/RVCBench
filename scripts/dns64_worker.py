#!/usr/bin/env python3
"""Finite model-only worker; accepts one JSON request and writes a durable result."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rvcbench.models.dns64_kernel import enhance_reference, load_dns64, sha256


def write_result(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if (request.get('schema_version') != 1 or not request.get('items') or
            not math.isfinite(request['dry']) or not 0 <= request['dry'] <= 1 or
            request['dataset_rate'] <= 0):
        raise ValueError('Invalid DNS64 worker request')
    result = {'schema_version': 1, 'status': 'running', 'rows': [],
              'request_sha256': sha256(args.request), 'weights_sha256': sha256(request['weights'])}
    write_result(args.result, result)
    try:
        if result['weights_sha256'] != request['weights_sha256']:
            raise ValueError('DNS64 weights changed before worker loading')
        import torch
        import torchaudio
        model, sources = load_dns64(request['weights'], request['device'])
        result['runtime'] = {'python': platform.python_version(), 'executable': sys.executable,
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                         if d.metadata['Name']}, 'torch': torch.__version__,
            'torchaudio': torchaudio.__version__, 'cuda': torch.version.cuda,
            'cpu_threads': torch.get_num_threads(), 'audio_backends': torchaudio.list_audio_backends(),
            'worker_sha256': sha256(__file__),
            'kernel_sha256': sha256(Path(__file__).parents[1] / 'src/rvcbench/models/dns64_kernel.py'),
            'denoiser_source_files': sources}
        for item in request['items']:
            if sha256(item['input_path']) != item['input_sha256']:
                raise ValueError('DNS64 worker input changed before inference')
            frames = enhance_reference(model, item['input_path'], item['output_path'],
                request['device'], request['dry'], request['dataset_rate'])
            result['rows'].append({'sample_id': item['sample_id'], 'output_path': item['output_path'],
                'sha256': sha256(item['output_path']), 'frames': frames, 'rate': request['dataset_rate']})
            write_result(args.result, result)
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_result(args.result, result)


if __name__ == '__main__':
    main()
