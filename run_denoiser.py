import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import hydra
import torch
import torchaudio
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from denoiser import pretrained
from src.datasets.audio_only import SimpleAllSpeakerData
from src.evaluation import fidelity
from src.utils.logger import log_config, setup_exp, setup_logger
from src.utils.seeding import configure_seeds


def _safe_get(conf, key, default=None):
    """Safely get a config value, compatible with old OmegaConf versions."""
    try:
        if hasattr(OmegaConf, 'select'):
            return OmegaConf.select(conf, key, default=default)
        # Old OmegaConf - use dict-like access
        keys = key.split('.')
        current = conf
        for k in keys:
            if isinstance(current, DictConfig) and k in current:
                current = current[k]
            else:
                return default
        return current
    except:
        return default


def _merge_dataset_config(denoiser_conf: DictConfig, dataset_conf: DictConfig) -> DictConfig:
    if denoiser_conf is None:
        raise ValueError("denoiser configuration is required")
    if dataset_conf is None:
        raise ValueError("dataset configuration is required for denoiser runs")
    if _safe_get(denoiser_conf, "dataset") is None:
        OmegaConf.set_struct(denoiser_conf, False)
        result = OmegaConf.merge(denoiser_conf, {"dataset": dataset_conf})
        OmegaConf.set_struct(result, True)
        return result
    return denoiser_conf


def _load_denoiser_model(denoiser_conf: DictConfig, device: torch.device):
    model_name = str(_safe_get(denoiser_conf, "model", default="dns64")).lower()
    model_path = _safe_get(denoiser_conf, "model_path", default=None)

    flags: Dict[str, object] = {
        "model_path": model_path,
        "dns48": False,
        "dns64": False,
        "master64": False,
        "valentini_nc": False,
    }

    if model_path:
        pass
    elif model_name in {"dns64", "dns_64"}:
        flags["dns64"] = True
    elif model_name in {"master64", "master_64"}:
        flags["master64"] = True
    elif model_name in {"valentini_nc", "valentini-nc"}:
        flags["valentini_nc"] = True
    else:
        flags["dns48"] = True

    args = SimpleNamespace(**flags)
    model = pretrained.get_model(args).to(device)
    model.eval()
    return model


def _prepare_resamplers(dataset_sr: int, model_sr: int):
    to_model = None
    to_dataset = None
    if dataset_sr != model_sr:
        to_model = torchaudio.transforms.Resample(orig_freq=dataset_sr, new_freq=model_sr)
        to_dataset = torchaudio.transforms.Resample(orig_freq=model_sr, new_freq=dataset_sr)
    return to_model, to_dataset


def _ensure_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = waveform.shape[-1]
    if current_len == target_len:
        return waveform
    if current_len > target_len:
        return waveform[..., :target_len]
    pad = target_len - current_len
    return F.pad(waveform, (0, pad))


def _write_audio(path: Path, audio: torch.Tensor, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio, sample_rate=sample_rate)


def _denoise_dataset(
    dataset,
    model,
    denoiser_conf: DictConfig,
    output_dir: Path,
    device: torch.device,
    logger,
) -> Dict[str, object]:
    dataset_sr = int(dataset.dataset_config.sampling_rate)
    model_sr = getattr(model, "sample_rate", dataset_sr)
    dry = float(_safe_get(denoiser_conf, "dry", default=0.0) or 0.0)
    to_model, to_dataset = _prepare_resamplers(dataset_sr, model_sr)

    processed = 0
    start_time = time.perf_counter()

    logger.info("[Denoiser] Writing outputs to %s", output_dir)
    dataset_root = getattr(dataset, "dataset_root", None) or getattr(dataset, "_dataset_root", None)
    for speaker_id in dataset.speakers_ids:
        speaker_dir = output_dir / str(speaker_id)
        loader = dataset.speaker_dataloaders.get(speaker_id)
        if loader is None:
            logger.warning("[Denoiser] No dataloader for speaker %s; skipping.", speaker_id)
            continue

        for batch in loader:
            batch = batch.to("cpu")
            wavs = batch.wav
            wav_lens = batch.wav_len
            paths = batch.path_out

            for idx in range(wavs.size(0)):
                target_len = int(wav_lens[idx].item())
                source_wave = wavs[idx: idx + 1, :, :target_len].squeeze(0)

                if to_model is not None:
                    source_for_model = to_model(source_wave)
                else:
                    source_for_model = source_wave

                source_for_model = source_for_model.to(device)
                with torch.no_grad():
                    estimate = model(source_for_model.unsqueeze(0)).squeeze(0)
                
                # DEBUG: Check model output
                if processed == 0:
                    logger.info(f"[DEBUG] First sample - source_for_model shape: {source_for_model.shape}, min: {source_for_model.min():.4f}, max: {source_for_model.max():.4f}")
                    logger.info(f"[DEBUG] First sample - estimate shape: {estimate.shape}, min: {estimate.min():.4f}, max: {estimate.max():.4f}")
                
                if dry:
                    estimate = (1.0 - dry) * estimate + dry * source_for_model
                estimate = estimate.cpu()

                if to_dataset is not None:
                    if processed == 0:
                        logger.info(f"[DEBUG] Before resample back - estimate shape: {estimate.shape}, min: {estimate.min():.4f}, max: {estimate.max():.4f}")
                    estimate = to_dataset(estimate)
                    if processed == 0:
                        logger.info(f"[DEBUG] After resample back - estimate shape: {estimate.shape}, min: {estimate.min():.4f}, max: {estimate.max():.4f}")

                estimate = _ensure_length(estimate, target_len)
                estimate = torch.clamp(estimate, -1.0, 1.0)

                original_path = Path(str(paths[idx])) if paths else speaker_dir / f"sample_{processed:05d}.wav"
                rel_path = None
                if dataset_root is not None:
                    try:
                        rel_path = original_path.resolve().relative_to(Path(dataset_root).resolve())
                    except Exception:
                        rel_path = None
                if rel_path is not None:
                    out_path = (output_dir / rel_path).with_suffix(".wav")
                else:
                    new_name = original_path.stem + ".wav"
                    out_path = speaker_dir / new_name

                _write_audio(out_path, estimate, dataset_sr)
                processed += 1

    elapsed = time.perf_counter() - start_time
    return {
        "total_speakers": len(dataset.speakers_ids),
        "total_files": processed,
        "elapsed_seconds": elapsed,
        "model_sample_rate": model_sr,
        "dry": dry,
    }


def _resolve_existing_dir(denoiser_conf: DictConfig) -> Optional[Path]:
    evaluation_conf = _safe_get(denoiser_conf, "evaluation", default=None)
    if evaluation_conf is None:
        return None
    if not isinstance(evaluation_conf, dict):
        evaluation_conf = OmegaConf.to_container(evaluation_conf, resolve=True)
    override = evaluation_conf.get("denoised_audio_dir") if isinstance(evaluation_conf, dict) else None
    if not override:
        return None
    resolved = Path(to_absolute_path(str(override))).expanduser().resolve()
    return resolved


@hydra.main(version_base="1.3", config_path="configs", config_name=None)
def main(conf: DictConfig):
    base_dir = Path(to_absolute_path(conf.base_dir))
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")

    denoiser_conf = _safe_get(conf, "denoiser")
    if denoiser_conf is None:
        raise ValueError("'denoiser' section is required in the configuration.")
    
    # Dataset config is already in conf.dataset, no need to merge

    exp_dir = setup_exp(base_dir / "results", conf.run_name)
    logger = setup_logger("benchmark", exp_dir, conf.run_name, screen=True, tofile=True)
    logger.info("Using device: %s", device)
    log_config(logger, conf)

    seed_value = _safe_get(conf, "seed", default=None)
    if seed_value is None:
        seed_value = _safe_get(denoiser_conf, "seed", default=None)
    if seed_value is None:
        logger.info("No seed specified; defaulting to 42.")
        seed_value = 42
    configure_seeds(seed_value, logger=logger)

    dataset = SimpleAllSpeakerData(conf, conf.dataset, logger)

    # Load clean audio dataset for fidelity evaluation if specified
    clean_dataset = None
    clean_root_path = _safe_get(conf, "clean_audio_root_path", default=None)
    if clean_root_path:
        logger.info(f"[Denoiser] Loading clean audio dataset from {clean_root_path} for fidelity evaluation")
        clean_dataset_conf = conf.dataset.copy()
        OmegaConf.set_struct(clean_dataset_conf, False)
        clean_dataset_conf.root_path = clean_root_path
        OmegaConf.set_struct(clean_dataset_conf, True)
        clean_dataset = SimpleAllSpeakerData(conf, clean_dataset_conf, logger)
    
    evaluate_only = bool(_safe_get(denoiser_conf, "evaluate_only", default=False))
    override_dir = _resolve_existing_dir(denoiser_conf)

    if evaluate_only:
        if override_dir is None:
            raise ValueError(
                "denoiser.evaluate_only is True but denoiser.evaluation.denoised_audio_dir was not provided."
            )
        if not override_dir.exists():
            raise FileNotFoundError(f"Provided denoised audio directory not found: {override_dir}")
        denoised_output_dir = override_dir
        processing_stats = None
        logger.info("[Denoiser] Evaluation-only mode; skipping enhancement stage.")
    else:
        if override_dir is not None:
            logger.info(
                "[Denoiser] denoiser.evaluation.denoised_audio_dir provided but evaluate_only is False; ignoring override."
            )
        output_subdir = _safe_get(denoiser_conf, "output_subdir", default="denoised_audio")
        denoised_output_dir = exp_dir / str(output_subdir)
        denoised_output_dir.mkdir(parents=True, exist_ok=True)

        model = _load_denoiser_model(denoiser_conf, device)
        processing_stats = _denoise_dataset(dataset, model, denoiser_conf, denoised_output_dir, device, logger)

    logger.info("=" * 50)
    logger.info("Evaluating fidelity of denoised audio...")
    # Use clean dataset for fidelity evaluation if available, otherwise use protected dataset
    reference_dataset = clean_dataset if clean_dataset is not None else dataset
    train_transcripts = reference_dataset.get_train_files_map()
    original_files = list(train_transcripts.keys())
    fidelity_metrics = fidelity.evaluate(
        original_files,
        str(denoised_output_dir),
        logger,
        target_sr=int(conf.dataset.sampling_rate),
        transcript_map=train_transcripts,
        bootstrap_config=_safe_get(conf, "evaluation.bootstrap", default=None),
        sample_metrics_path=exp_dir / "denoiser_fidelity_sample_metrics.csv",
    )
    logger.info("Fidelity Metrics: %s", fidelity_metrics)

    metrics_path_value = _safe_get(conf, "output_paths.metrics_file", default="results/metrics.json")
    metrics_filename = Path(metrics_path_value).name
    metrics_file = exp_dir / metrics_filename
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, object] = {
        "run_name": conf.run_name,
        "config": OmegaConf.to_container(conf, resolve=True),
        "denoiser": {
            "output_path": str(denoised_output_dir),
            "processing": processing_stats,
        },
        "fidelity_evaluation": fidelity_metrics,
    }
    all_metrics = fidelity.make_serializable(all_metrics)
    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
        handle.write("\n")
    logger.info("Saved metrics to %s", metrics_file)


if __name__ == "__main__":
    main()
