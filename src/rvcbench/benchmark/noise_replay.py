"""Reconstruct historical GR outputs from a frozen batch-noise archive."""
import logging
import math
from pathlib import Path

from omegaconf import OmegaConf

from .artifacts import atomic_json, file_hash, sample_id, provenance
from rvcbench.datasets.zero_shot import ZeroShotDataset


def replay_gr_noise(dataset_root, subset_manifest, noise_archive, historical_directory,
                    output, batch_size=8, sample_rate=24000, hop_length=512,
                    rng_seed=None, epsilon=0.03137255, device='cpu'):
    """Replay selected outputs, preserving the full cohort's batch positions.

    By default this replays archived perturbations. With rng_seed, the full
    cohort RNG stream is regenerated and verified against every archived batch. Each
    output must match its historical WAV before the stage is marked complete.
    The supported historical input format is mono PCM16 at the declared rate.
    """
    import soundfile as sf
    import torch

    if rng_seed is not None and (not isinstance(rng_seed, int) or
            not -(2**63) <= rng_seed < 2**64 or not math.isfinite(epsilon) or epsilon < 0):
        raise ValueError('Invalid RNG seed or Gaussian standard deviation')
    if batch_size <= 0 or sample_rate <= 0 or hop_length <= 0:
        raise ValueError('Batch size, sample rate and hop length must be positive')
    dataset_root, subset_manifest, noise_archive, historical_directory, output = [
        Path(p).expanduser().resolve() for p in
        (dataset_root, subset_manifest, noise_archive, historical_directory, output)]
    config = OmegaConf.create({'root_path': str(dataset_root), 'use_hf_dataset': False,
                              'name': 'LibriTTS', 'manifest_variant': 'speaker'})
    logger = logging.getLogger('rvcbench.noise_replay')
    cohort = ZeroShotDataset(OmegaConf.create({}), config, logger)
    config.manifest_filename = str(subset_manifest)
    selected = ZeroShotDataset(OmegaConf.create({}), config, logger).get_zero_shot_samples()
    if len({sample_id(s) for s in selected}) != len(selected):
        raise ValueError('Duplicate selected sample identities')
    speakers, information = {}, {}
    for sample in cohort.get_zero_shot_samples():
        counts = [len(str(sample.extra.get(k) or '').split(' '))
                  for k in ('prompt_phonemes', 'target_phonemes')]
        if not all(1 <= count <= 640 for count in counts):
            continue  # Historical TextAudioSpeakerDataset filter.
        if not sample.prompt_path.is_file() or not sample.target_path.is_file():
            raise FileNotFoundError('Historical cohort audio is missing; refusing to change batch positions')
        info = sf.info(sample.prompt_path)
        if info.channels != 1 or info.subtype != 'PCM_16' or info.samplerate != sample_rate:
            raise ValueError(f'Unsupported historical reference format: {sample.prompt_path}')
        if info.frames <= 0:
            raise ValueError(f'Empty reference: {sample.prompt_path}')
        information[sample_id(sample)] = info
        speakers.setdefault(sample.speaker_id, []).append(sample)
    # weights_only excludes arbitrary pickle objects; mmap avoids materializing
    # the multi-gigabyte archive. No archive tensor is modified.
    archive = torch.load(noise_archive, map_location='cpu', weights_only=True, mmap=True)
    if set(archive) != set(speakers):
        raise ValueError('Noise archive speakers differ from the filtered historical cohort')
    positions = {}
    for speaker, samples in speakers.items():
        batches = archive[speaker]
        if len(batches) != (len(samples) + batch_size - 1) // batch_size:
            raise ValueError(f'Archive batch count differs for speaker {speaker}')
        for batch_index, start in enumerate(range(0, len(samples), batch_size)):
            batch = samples[start:start + batch_size]
            frames = [information[sample_id(s)].frames for s in batch]
            tensor = batches[batch_index]
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or
                    tuple(tensor.shape) != (len(batch), 1, max(frames))):
                raise ValueError(f'Archive batch shape/dtype differs for {speaker}, batch {batch_index}')
            # For the historical center=False STFT with symmetric (n_fft-hop)/2
            # padding, the spectral frame count is waveform_frames // hop.
            order = torch.sort(torch.tensor([n // hop_length for n in frames]),
                               descending=True).indices.tolist()
            for slot, index in enumerate(order):
                positions[sample_id(batch[index])] = (batch[index], batch_index, slot)
    planned, destinations = [], set()
    for sample in selected:
        sid = sample_id(sample)
        if sid not in positions:
            raise ValueError(f'Selected sample absent from historical filtered cohort: {sid}')
        original, batch, slot = positions[sid]
        if (original.prompt_path != sample.prompt_path or original.target_path != sample.target_path or
                original.prompt_text != sample.prompt_text or original.target_text != sample.target_text):
            raise ValueError(f'Subset differs from historical cohort: {sid}')
        destination = output / 'protected_audio' / str(sample.speaker_id) / sample.prompt_path.name
        if destination in destinations:
            raise ValueError('Selected references collide at an output path')
        destinations.add(destination)
        historical = historical_directory / str(sample.speaker_id) / sample.prompt_path.name
        if not historical.is_file():
            raise FileNotFoundError(f'Historical comparison WAV missing: {historical}')
        planned.append((sample, batch, slot, destination, historical))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'stage': 'protection', 'variant': ('gr_seeded_batch_rng_v1' if rng_seed is not None
                                                               else 'gr_archived_noise_replay_v1'),
        'status': 'running', 'requested': len(planned), 'verified': 0, 'rows': [],
        'noise_archive': str(noise_archive), 'noise_archive_sha256': file_hash(noise_archive),
        'subset_manifest': str(subset_manifest), 'subset_manifest_sha256': file_hash(subset_manifest),
        'cohort_pairs': sum(map(len, speakers.values())),
        'cohort_source_manifest_sha256': file_hash(dataset_root / 'metadata.parquet'),
        'batch_size': batch_size, 'sample_rate': sample_rate, 'hop_length': hop_length,
        'pcm_divisor': 32768, 'torch': torch.__version__, 'soundfile': sf.__version__,
        'source_sha256': file_hash(Path(__file__)),
        'runtime': provenance(Path(__file__).resolve().parents[3]),
        'source_files': {str(p.relative_to(Path(__file__).resolve().parents[3])): file_hash(p)
                         for p in (Path(__file__).resolve(), Path(__file__).with_name('artifacts.py'),
                                   Path(__file__).parents[1] / 'datasets/zero_shot.py',
                                   Path(__file__).parents[1] / 'datasets/manifest_utils.py')},
        'limits': ['Uses archived noise; does not reconstruct the original RNG stream.',
                   'Historical WAV comparison is required; no alternate batch permutation is searched.']}
    if rng_seed is not None:
        manifest['rng_verification'] = {'seed': rng_seed, 'epsilon': epsilon, 'device': str(device),
            'requested_batches': sum(len(v) for v in archive.values()), 'verified_batches': 0}
        manifest['limits'] = ['Entire cohort RNG stream must match the archived batches.',
                              'Selected WAVs must match historical outputs; cross-runtime RNG equivalence is not assumed.']
    path = output / 'stage_manifest.json'
    atomic_json(path, manifest)
    try:
        regenerated = {}
        if rng_seed is not None:
            generator = torch.Generator(device=device).manual_seed(rng_seed)
            selected_slots = {(s.speaker_id, b, slot): s for s, b, slot, _, _ in planned}
            for speaker, cohort_samples in speakers.items():
                for batch, start in enumerate(range(0, len(cohort_samples), batch_size)):
                    samples = cohort_samples[start:start + batch_size]
                    shape = (len(samples), 1, max(information[sample_id(s)].frames for s in samples))
                    generated = (torch.randn(shape, dtype=torch.float32, device=device,
                                             generator=generator) * epsilon).cpu()
                    if not torch.equal(generated, archive[speaker][batch]):
                        raise ValueError(f'Regenerated RNG differs for {speaker}, batch {batch}')
                    for slot in range(len(samples)):
                        sample = selected_slots.get((speaker, batch, slot))
                        if sample is not None:
                            frames = information[sample_id(sample)].frames
                            regenerated[sample_id(sample)] = generated[slot, 0, :frames].clone()
                    manifest['rng_verification']['verified_batches'] += 1
                    atomic_json(path, manifest)
        for sample, batch, slot, destination, historical in planned:
            audio, rate = sf.read(sample.prompt_path, dtype='int16')
            noise = (regenerated[sample_id(sample)] if rng_seed is not None else
                     archive[sample.speaker_id][batch][slot, 0, :len(audio)])
            if not torch.isfinite(noise).all():
                raise ValueError(f'Nonfinite archived noise for {sample_id(sample)}')
            result = (torch.from_numpy(audio).float() / 32768 + noise).clamp(-1, 1)
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, result.numpy(), rate, subtype='PCM_16')
            row = {'sample_id': sample_id(sample), 'speaker_id': sample.speaker_id,
                   'archive_batch': batch, 'archive_slot': slot,
                   'clean_prompt_path': str(sample.prompt_path), 'clean_prompt_sha256': file_hash(sample.prompt_path),
                   'reference_path': str(destination), 'reference_sha256': file_hash(destination),
                   'historical_path': str(historical), 'historical_sha256': file_hash(historical)}
            manifest['rows'].append(row)
            if row['reference_sha256'] != row['historical_sha256']:
                raise ValueError(f'Replayed WAV differs from historical output: {row["sample_id"]}')
            manifest['verified'] += 1
            atomic_json(path, manifest)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic_json(path, manifest)
    return manifest
