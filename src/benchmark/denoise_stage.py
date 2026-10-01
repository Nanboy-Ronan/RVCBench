"""Finite DNS64 reference enhancement with explicit weights and input lineage."""
import logging
import math
import json
import subprocess
import os
from pathlib import Path

from omegaconf import OmegaConf

from .artifacts import atomic_json, file_hash, provenance, sample_id
from .reference_stages import bind_reference_stage
from src.datasets.zero_shot import ZeroShotDataset


def _load_dns64(weights, device):
    from src.models.dns64_kernel import load_dns64
    return load_dns64(weights, device)


def _run_worker(runtime_python, output, weights, weight_sha, samples, bindings,
                destinations, device, dry, dataset_rate, timeout_seconds):
    import soundfile as sf
    import numpy as np
    runtime_python = os.path.abspath(os.path.expanduser(str(runtime_python)))
    worker = Path(__file__).resolve().parents[2] / 'scripts/dns64_worker.py'
    items = {}
    for sample, binding, destination in zip(samples, bindings, destinations):
        items.setdefault(str(destination), {'sample_id': sample_id(sample),
            'input_path': str(sample.prompt_path), 'input_sha256': binding['reference_sha256'],
            'output_path': str(destination)})
    request_path, result_path = output / 'worker_request.json', output / 'worker_result.json'
    request = {'schema_version': 1, 'weights': str(weights), 'weights_sha256': weight_sha,
        'device': str(device), 'dry': dry, 'dataset_rate': dataset_rate, 'items': list(items.values())}
    atomic_json(request_path, request)
    with (output / 'worker.log').open('w') as log:
        subprocess.run([str(runtime_python), str(worker), '--request', str(request_path),
                        '--result', str(result_path)], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout_seconds)
    result = json.loads(result_path.read_text())
    if (result.get('schema_version') != 1 or result.get('status') != 'complete' or
            result.get('request_sha256') != file_hash(request_path) or
            result.get('weights_sha256') != weight_sha):
        raise ValueError('DNS64 worker result provenance differs from its request')
    runtime = result.get('runtime') or {}
    kernel = Path(__file__).resolve().parents[1] / 'models/dns64_kernel.py'
    if (runtime.get('worker_sha256') != file_hash(worker) or
            runtime.get('kernel_sha256') != file_hash(kernel) or
            not runtime.get('packages') or not runtime.get('executable') or
            os.path.abspath(os.path.expanduser(runtime['executable'])) != runtime_python):
        raise ValueError('DNS64 worker runtime provenance differs from the configured worker')
    rows = result.get('rows', [])
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {r['sample_id'] for r in items.values()}:
        raise ValueError('DNS64 worker output identities differ from its request')
    enhanced = {}
    for destination, item in items.items():
        row = by_id[item['sample_id']]
        if row['output_path'] != destination or row['sha256'] != file_hash(destination):
            raise ValueError('DNS64 worker output path or content differs')
        info, source_info = sf.info(destination), sf.info(item['input_path'])
        expected_frames = (source_info.frames * dataset_rate + source_info.samplerate - 1) // source_info.samplerate
        if (info.channels != 1 or info.frames != expected_frames or info.samplerate != dataset_rate or
                row['frames'] != info.frames or row['rate'] != info.samplerate):
            raise ValueError('DNS64 worker output audio format differs')
        if not np.isfinite(sf.read(destination)[0]).all():
            raise ValueError('DNS64 worker output is nonfinite')
        enhanced[Path(destination)] = info.frames
    return result, enhanced


def denoise_dns64(dataset_root, subset_manifest, reference_directory, weights, output,
                  device='cpu', dry=0.0, dataset_rate=16000, runtime_python=None, timeout_seconds=600):
    """Mirror the historical dataset-rate resampling/mixing recipe on selected pairs.

    DNS64 is loaded strictly from a local state dict. No model-name fallback or
    automatic download is used. The original wrapper is left unchanged.
    """
    import soundfile as sf
    if not math.isfinite(dry) or not 0 <= dry <= 1 or dataset_rate <= 0:
        raise ValueError('dry must be in [0, 1] and dataset_rate must be positive')
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be finite and positive')
    dataset_root, subset_manifest, reference_directory, weights, output = [
        Path(p).expanduser().resolve() for p in
        (dataset_root, subset_manifest, reference_directory, weights, output)]
    if not weights.is_file():
        raise FileNotFoundError(f'DNS64 weights missing: {weights}')
    config = OmegaConf.create({'root_path': str(dataset_root), 'use_hf_dataset': False,
        'name': 'LibriTTS', 'manifest_variant': 'speaker', 'manifest_filename': str(subset_manifest)})
    dataset = ZeroShotDataset(OmegaConf.create({}), config, logging.getLogger('rvcbench.denoise'))
    samples, lineage = bind_reference_stage(dataset.get_zero_shot_samples(), dataset._dataset_root,
                                           reference_directory, 'dns64_input')
    destinations = [output / 'denoised_audio' / s.speaker_id / s.prompt_path.name for s in samples]
    identities = {}
    for destination, binding in zip(destinations, lineage['bindings']):
        identity = (binding['speaker_id'], binding['clean_prompt_sha256'], binding['reference_sha256'])
        if destination in identities and identities[destination] != identity:
            raise ValueError('Distinct references collide at a denoised output path')
        identities[destination] = identity
    if len({sample_id(s) for s in samples}) != len(samples):
        raise ValueError('Duplicate selected sample identities')
    for sample in samples:
        info = sf.info(sample.prompt_path)
        if info.channels != 1 or info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f'Invalid mono reference: {sample.prompt_path}')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'stage': 'denoising', 'variant': 'dns64_dataset_rate_v1',
        'status': 'running', 'requested': len(samples), 'verified': 0, 'rows': [],
        'weights_path': str(weights), 'weights_sha256': file_hash(weights),
        'subset_manifest': str(subset_manifest), 'subset_manifest_sha256': file_hash(subset_manifest),
        'input_stage': lineage, 'dataset_rate': dataset_rate, 'model_rate': 16000,
        'dry': dry, 'device': str(device), 'runtime': provenance(Path(__file__).resolve().parents[2]),
        'writer': 'torchaudio.save_default_wav', 'source_sha256': file_hash(Path(__file__)),
        'reference_reuse_policy': 'enhance_unique_reference_once_preserve_all_pair_rows',
        'limits': ['Historical numeric equivalence must be checked separately.',
                   'The optional denoiser package has legacy Hydra dependency constraints.']}
    path = output / 'stage_manifest.json'
    atomic_json(path, manifest)
    try:
        if runtime_python:
            manifest['worker_python'] = str(runtime_python)
            manifest['worker_result_path'] = str(output / 'worker_result.json')
            atomic_json(path, manifest)
            worker_result, enhanced = _run_worker(runtime_python, output, weights,
                manifest['weights_sha256'], samples, lineage['bindings'], destinations,
                device, dry, dataset_rate, timeout_seconds)
            manifest['worker_result'] = worker_result
        else:
            import torchaudio
            manifest.update(torchaudio=torchaudio.__version__, audio_backends=torchaudio.list_audio_backends())
            model, sources = _load_dns64(weights, device)
            manifest['denoiser_source_files'] = sources
            if model.sample_rate != 16000:
                raise ValueError('DNS64 model sample rate differs from the declared architecture')
            enhanced = {}
        for sample, binding, destination in zip(samples, lineage['bindings'], destinations):
            if file_hash(sample.prompt_path) != binding['reference_sha256']:
                raise ValueError('Reference input changed during denoising')
            if destination not in enhanced:
                from src.models.dns64_kernel import enhance_reference
                length = enhance_reference(model, sample.prompt_path, destination, device, dry, dataset_rate)
                enhanced[destination] = length
            length = enhanced[destination]
            manifest['rows'].append({'sample_id': sample_id(sample), 'speaker_id': sample.speaker_id,
                'clean_prompt_path': binding['clean_prompt_path'],
                'clean_prompt_sha256': binding['clean_prompt_sha256'],
                'input_reference_path': str(sample.prompt_path),
                'input_reference_sha256': binding['reference_sha256'],
                'reference_path': str(destination), 'reference_sha256': file_hash(destination),
                'output_rate': dataset_rate, 'output_frames': length})
            manifest['verified'] += 1
            atomic_json(path, manifest)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic_json(path, manifest)
    return manifest
