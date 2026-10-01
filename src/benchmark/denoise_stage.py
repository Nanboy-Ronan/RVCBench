"""Finite DNS64 reference enhancement with explicit weights and input lineage."""
import logging
import math
from pathlib import Path

from omegaconf import OmegaConf

from .artifacts import atomic_json, file_hash, provenance, sample_id
from .reference_stages import bind_reference_stage
from src.datasets.zero_shot import ZeroShotDataset


def _load_dns64(weights, device):
    import torch
    from denoiser import pretrained
    model = pretrained.dns64(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    sources = {str(p.relative_to(Path(pretrained.__file__).parent)): file_hash(p)
               for p in sorted(Path(pretrained.__file__).parent.rglob('*.py'))}
    return model.eval().to(device), sources


def denoise_dns64(dataset_root, subset_manifest, reference_directory, weights, output,
                  device='cpu', dry=0.0, dataset_rate=16000):
    """Mirror the historical dataset-rate resampling/mixing recipe on selected pairs.

    DNS64 is loaded strictly from a local state dict. No model-name fallback or
    automatic download is used. The original wrapper is left unchanged.
    """
    import soundfile as sf
    import torch
    import torchaudio

    if not math.isfinite(dry) or not 0 <= dry <= 1 or dataset_rate <= 0:
        raise ValueError('dry must be in [0, 1] and dataset_rate must be positive')
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
        'torchaudio': torchaudio.__version__, 'audio_backends': torchaudio.list_audio_backends(),
        'writer': 'torchaudio.save_default_wav', 'source_sha256': file_hash(Path(__file__)),
        'reference_reuse_policy': 'enhance_unique_reference_once_preserve_all_pair_rows',
        'limits': ['Historical numeric equivalence must be checked separately.',
                   'The optional denoiser package has legacy Hydra dependency constraints.']}
    path = output / 'stage_manifest.json'
    atomic_json(path, manifest)
    try:
        model, sources = _load_dns64(weights, device)
        manifest['denoiser_source_files'] = sources
        if model.sample_rate != 16000:
            raise ValueError('DNS64 model sample rate differs from the declared architecture')
        to_model = (torchaudio.transforms.Resample(dataset_rate, 16000)
                    if dataset_rate != 16000 else None)
        to_dataset = (torchaudio.transforms.Resample(16000, dataset_rate)
                      if dataset_rate != 16000 else None)
        enhanced = {}
        for sample, binding, destination in zip(samples, lineage['bindings'], destinations):
            if file_hash(sample.prompt_path) != binding['reference_sha256']:
                raise ValueError('Reference input changed during denoising')
            if destination not in enhanced:
                audio, rate = torchaudio.load(str(sample.prompt_path))
                if not torch.isfinite(audio).all():
                    raise ValueError('Nonfinite reference input')
                if rate != dataset_rate:
                    audio = torchaudio.transforms.Resample(rate, dataset_rate)(audio)
                length = audio.shape[-1]
                source = (to_model(audio) if to_model else audio).to(device)
                with torch.no_grad():
                    estimate = model(source.unsqueeze(0)).squeeze(0)
                if estimate.shape != source.shape or not torch.isfinite(estimate).all():
                    raise ValueError('Invalid DNS64 output shape or values')
                if dry:
                    estimate = (1 - dry) * estimate + dry * source
                estimate = estimate.cpu()
                if to_dataset:
                    estimate = to_dataset(estimate)
                estimate = torch.nn.functional.pad(estimate[..., :length],
                    (0, max(0, length - estimate.shape[-1]))).clamp(-1, 1)
                destination.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(destination), estimate, dataset_rate)
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
