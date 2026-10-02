"""Audio-only Enkidu production on an explicit cohort and selected outputs."""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from .artifacts import atomic_json, file_hash, input_records, sample_id, provenance
from rvcbench.datasets.zero_shot import ZeroShotDataset


class AudioReferences:
    def __init__(self, samples, rate):
        self.samples, self.rate = samples, rate

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        import soundfile as sf
        import torch
        import torchaudio
        for sample in self.samples:
            audio, rate = sf.read(sample.prompt_path, dtype='int16')
            # Match the training loader's PCM normalization/resampling order.
            audio = torch.from_numpy(audio).float().unsqueeze(0)
            if rate != self.rate:
                audio = torchaudio.functional.resample(audio, rate, self.rate)
            audio = audio / 32768
            yield SimpleNamespace(wav=audio.unsqueeze(0), wav_len=[audio.shape[-1]],
                                  path_out=[str(sample.prompt_path)])


def protect_enkidu(dataset_root, subset_manifest, model_directory, output,
                   device='cpu', seed=42, epochs=10):
    import soundfile as sf
    import torch
    from rvcbench.protection.enkidu import EnkiduProtector
    from rvcbench.utils.seeding import configure_seeds

    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError('epochs must be a positive integer')
    root, subset, model_root, output = [Path(p).expanduser().resolve() for p in
                                      (dataset_root, subset_manifest, model_directory, output)]
    required = ('hyperparams.yaml', 'embedding_model.ckpt', 'classifier.ckpt',
                'mean_var_norm_emb.ckpt', 'label_encoder.ckpt')
    for name in required:
        if not (model_root / name).is_file():
            raise FileNotFoundError(f'Enkidu model asset missing: {model_root / name}')
    logger = logging.getLogger('rvcbench.enkidu')
    config = OmegaConf.create({'root_path': str(root), 'use_hf_dataset': False,
                              'name': 'LibriTTS', 'manifest_variant': 'speaker'})
    cohort = ZeroShotDataset(OmegaConf.create({}), config, logger).get_zero_shot_samples()
    config.manifest_filename = str(subset)
    selected = ZeroShotDataset(OmegaConf.create({}), config, logger).get_zero_shot_samples()
    groups, by_id = {}, {}
    for sample in cohort:
        counts = [len(str(sample.extra.get(k) or '').split(' '))
                  for k in ('prompt_phonemes', 'target_phonemes')]
        if not all(1 <= n <= 640 for n in counts):
            continue
        info = sf.info(sample.prompt_path)
        if info.channels != 1 or info.subtype != 'PCM_16' or info.frames <= 512:
            raise ValueError(f'Invalid Enkidu PCM16 reference: {sample.prompt_path}')
        by_id[sample_id(sample)] = sample
        groups.setdefault(sample.speaker_id, []).append(sample)
    if len(by_id) != sum(map(len, groups.values())) or len({sample_id(s) for s in selected}) != len(selected):
        raise ValueError('Duplicate Enkidu sample identities')
    clean = input_records(selected)
    if any(not r['prompt_sha256'] or not r['target_sha256'] for r in clean):
        raise ValueError('Selected Enkidu inputs are incomplete')
    for sample in selected:
        original = by_id.get(sample_id(sample))
        if not original or any(getattr(original, k) != getattr(sample, k) for k in
                ('prompt_path', 'target_path', 'prompt_text', 'target_text')):
            raise ValueError('Selected Enkidu pair differs from its training cohort')
    destinations = [output / 'protected_audio' / s.speaker_id / s.prompt_path.name for s in selected]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Selected references collide at an Enkidu output path')
    settings = OmegaConf.load(Path(__file__).parents[3] / 'configs/enkidu_on_libritts.yaml').protection
    settings.perturbation_epochs = epochs
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'schema_version': 1, 'stage': 'protection', 'variant': 'enkidu_audio_only_cohort_v1',
        'status': 'running', 'requested': len(selected), 'verified': 0, 'rows': [],
        'seed': seed, 'device': str(device), 'settings': OmegaConf.to_container(settings),
        'training_pairs': len(by_id), 'training_speakers': list(groups),
        'training_requested_steps': len(by_id) * epochs, 'training_verified_steps': 0,
        'cohort_manifest_sha256': file_hash(root / 'metadata.parquet'),
        'subset_manifest_sha256': file_hash(subset),
        'training_references': [{'sample_id': sample_id(s), 'path': str(s.prompt_path),
                                 'sha256': file_hash(s.prompt_path)} for g in groups.values() for s in g],
        'model_assets': {name: file_hash(model_root / name) for name in required},
        'source_hashes': {str(p.relative_to(Path(__file__).parents[3])): file_hash(p) for p in
            (Path(__file__), Path(__file__).parents[1] / 'protection/enkidu.py',
             Path(__file__).parents[1] / 'protection/base_protector.py')},
        'runtime': provenance(Path(__file__).parents[3]),
        'limits': ['Audio-only input bypasses historical random text-feature fallbacks; RNG equivalence is unproven.',
                   'Historical cumulative-gradient optimization is preserved. Historical output equivalence requires separate comparison.']}
    path = output / 'stage_manifest.json'
    def progress(speaker, epoch, batch, losses):
        manifest['training_verified_steps'] += 1
        elapsed = time.perf_counter() - started
        manifest['elapsed_seconds'] = elapsed
        manifest['estimated_training_remaining_seconds'] = elapsed * (
            manifest['training_requested_steps'] - manifest['training_verified_steps']) / manifest['training_verified_steps']
        manifest['progress'] = {'speaker': str(speaker), 'epoch': epoch + 1, 'batch': batch + 1, 'losses': losses}
        atomic_json(path, manifest)
    atomic_json(path, manifest)
    try:
        configure_seeds(seed)
        loaders = {sid: AudioReferences(samples, 16000) for sid, samples in groups.items()}
        data = SimpleNamespace(speakers_ids=list(groups), speaker_dataloaders=loaders)
        model_config = OmegaConf.create({'checkpoint_path': str(model_root), 'source': str(model_root)})
        protector = EnkiduProtector(model_config, model_cache=output / 'model_cache',
            progress_callback=progress, speaker_data=data, output_dir=output / 'protected_audio',
            config=settings, dataset_config=SimpleNamespace(speaker_id=None, sampling_rate=24000),
            logger=logger, device=device)
        for parameter in protector.model.parameters():
            parameter.requires_grad_(False)
        protector.model.eval()
        protector.generate_perturbations()
        manifest['noise_archive_sha256'] = file_hash(output / 'protected_audio/enkidu.noise')
        for noise in protector.noises.values():
            if not all(torch.isfinite(t).all() for t in noise.values()):
                raise ValueError('Nonfinite Enkidu noise')
        protector.speaker_data = SimpleNamespace(speakers_ids=list(dict.fromkeys(s.speaker_id for s in selected)),
            speaker_dataloaders={sid: AudioReferences([s for s in selected if s.speaker_id == sid], 16000)
                                 for sid in dict.fromkeys(s.speaker_id for s in selected)})
        protector.save_protected_audio()
        for sample, row, destination in zip(selected, clean, destinations):
            if (file_hash(sample.prompt_path) != row['prompt_sha256'] or
                    file_hash(sample.target_path) != row['target_sha256']):
                raise ValueError('Selected Enkidu inputs changed during production')
            info, original_info = sf.info(destination), sf.info(sample.prompt_path)
            expected = (original_info.frames * 16000 + original_info.samplerate - 1) // original_info.samplerate
            if info.channels != 1 or info.samplerate != 16000 or info.frames != expected:
                raise ValueError('Invalid Enkidu output format')
            manifest['rows'].append({'sample_id': sample_id(sample), 'speaker_id': sample.speaker_id,
                'clean_prompt_sha256': row['prompt_sha256'], 'reference_path': str(destination),
                'reference_sha256': file_hash(destination), 'output_rate': 16000, 'output_frames': info.frames})
            manifest['verified'] += 1
        if manifest['training_verified_steps'] != manifest['training_requested_steps']:
            raise ValueError('Incomplete Enkidu training cohort')
        for row in manifest['training_references']:
            if file_hash(row['path']) != row['sha256']:
                raise ValueError('Enkidu training input changed during production')
        if any(file_hash(model_root / name) != value for name, value in manifest['model_assets'].items()):
            raise ValueError('Enkidu model assets changed during production')
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic_json(path, manifest)
    return manifest
