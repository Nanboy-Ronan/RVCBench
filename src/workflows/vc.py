import importlib
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from src.evaluation import generation

_SPEAKER_SLUG_PATTERN = re.compile(r"[^0-9A-Za-z_.-]")


_ADVERSARY_REGISTRY: Dict[str, Dict[str, str]] = {
    "ots": {
        "bertvits2": "src.adversary.bertvit2_ots:BertVits2ZeroShotAdversary",
        "ozspeech": "src.adversary.ozspeech_ots:OzSpeechZeroShotAdversary",
        "higgs_audio": "src.adversary.higgs_audio_ots:HiggsAudioZeroShotAdversary",
        "cosyvoice": "src.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cozyvoice": "src.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cozyvoice2": "src.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cosyvoice2": "src.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "sparktts": "src.adversary.sparktts_ots:SparkTTSZeroShotAdversary",
        "vall_e": "src.adversary.vall_e_ots:VallEZeroShotAdversary",
        "styletts2": "src.adversary.styletts2_ots:StyleTTS2ZeroShotAdversary",
        "glowtts": "src.adversary.glowtts_ots:GlowTTSZeroShotAdversary",
        "glm_tts": "src.adversary.glmtts_ots:GLMTTSZeroShotAdversary",
        "glmtts": "src.adversary.glmtts_ots:GLMTTSZeroShotAdversary",
        "kimi_audio": "src.adversary.kimi_audio_ots:KimiAudioZeroShotAdversary",
        "moss_ttsd": "src.adversary.moss_ttsd_ots:MossTTSDZeroShotAdversary",
        "playdiffusion": "src.adversary.playdiffusion_ots:PlayDiffusionZeroShotAdversary",
        "bark_voice_clone": "src.adversary.bark_voice_clone_ots:BarkVoiceCloneZeroShotAdversary",
        "fishspeech": "src.adversary.fishspeech_ots:FishSpeechZeroShotAdversary",
        "qwen3_omni": "src.adversary.qwen3_omni_ots:Qwen3OmniZeroShotAdversary",
        "mgm_omni": "src.adversary.mgm_omni_ots:MGMOmniZeroShotAdversary",
        "vibevoice": "src.adversary.vibevoice_ots:VibeVoiceZeroShotAdversary",
    },
    "finetune": {
        "bertvits2": "src.adversary.bertvits2_finetune:BertVits2FinetuneAdversary",
    },
}


def _speaker_slug(value: str) -> str:
    token = str(value or "").strip()
    slug = _SPEAKER_SLUG_PATTERN.sub("_", token)
    return slug or "unknown"


def _locate_clone_path(
    target_path: Path,
    metadata: dict,
    available_files: dict,
    generated_audio_dir: Path,
    sample_index: int,
    logger,
    sample=None,
) -> Optional[Path]:
    # Match the behavior of _cloned_filename() in base_adversary.py
    # which uses both prompt_path and target_path to create unique filenames
    
    speaker_hint = target_path.parent.name or metadata.get("speaker_id")
    if not speaker_hint or not generated_audio_dir.exists():
        return None
    
    speaker_dir = generated_audio_dir / _speaker_slug(speaker_hint)
    if not speaker_dir.exists():
        return None
    
    target_stem = target_path.stem
    
    # Try combined prompt_to_target pattern first (new naming scheme)
    prompt_path = metadata.get("prompt_path")
    if prompt_path:
        try:
            prompt_stem = Path(str(prompt_path)).stem
            # Sanitize the stems to match what _cloned_filename does in base_adversary.py
            # Preserve + and - signs for SNR identification
            prompt_stem_sanitized = re.sub(r'[^0-9A-Za-z_+\-]', '_', str(prompt_stem).strip())
            target_stem_sanitized = re.sub(r'[^0-9A-Za-z_+\-]', '_', str(target_stem).strip())
            candidate = speaker_dir / f"{prompt_stem_sanitized}_to_{target_stem_sanitized}_cloned.wav"
            if candidate.exists():
                name = candidate.name
                if name in available_files and available_files[name] == candidate:
                    available_files.pop(name, None)
                return candidate
        except Exception:
            pass
    
    # Fallback to prompt-only pattern (if prompt exists but no combined file)
    if prompt_path:
        try:
            prompt_stem = Path(str(prompt_path)).stem
            prompt_stem_sanitized = re.sub(r'[^0-9A-Za-z_+\-]', '_', str(prompt_stem).strip())
            candidate = speaker_dir / f"{prompt_stem_sanitized}_cloned.wav"
            if candidate.exists():
                name = candidate.name
                if name in available_files and available_files[name] == candidate:
                    available_files.pop(name, None)
                return candidate
        except Exception:
            pass
    
    # Fallback to target-only pattern (backward compatibility)
    candidate = speaker_dir / f"{target_stem}_cloned.wav"
    if candidate.exists():
        name = candidate.name
        if name in available_files and available_files[name] == candidate:
            available_files.pop(name, None)
        return candidate

    if not available_files:
        raise RuntimeError(
            f"No generated audio files found in {generated_audio_dir}; cannot locate cloned audio."
        )

    return generation._resolve_generated_file(target_path, sample_index, metadata, available_files, logger)


def _collect_evaluation_pairs(
    dataset,
    generated_audio_dir: Path,
    max_samples: Optional[int],
    logger,
    original_resolver: Optional[Callable[[Path, dict], Optional[Path]]] = None,
) -> List[Tuple[Path, Path, dict]]:
    samples = list(dataset.iter_zero_shot_samples(max_samples=max_samples))

    available_files = {}
    if generated_audio_dir.exists():
        for path in generated_audio_dir.rglob("*.wav"):
            available_files.setdefault(path.name, path)

    evaluation_pairs: List[Tuple[Path, Path, dict]] = []
    missing = 0
    for idx, sample in enumerate(samples):
        if not sample.target_path:
            logger.warning("[VC] Sample %s has no target path; skipping.", sample)
            missing += 1
            continue

        target_path = Path(str(sample.target_path))
        metadata = {
            "speaker_id": str(sample.speaker_id),
            "text": sample.target_text,
            "prompt_path": sample.prompt_path,
        }
        if original_resolver is not None:
            override_path = original_resolver(target_path, metadata)
            if override_path is not None:
                target_path = Path(str(override_path))
        if not target_path.exists():
            logger.warning("[VC] Ground-truth audio %s not found; skipping sample.", target_path)
            missing += 1
            continue
        clone_path = _locate_clone_path(
            target_path,
            metadata,
            available_files,
            generated_audio_dir,
            idx,
            logger,
            sample,
        )
        
        if clone_path is None:
            logger.warning("[VC] Could not locate generated audio for sample %d: %s; skipping.", idx, target_path)
            missing += 1
            continue

        if clone_path is None:
            missing += 1
            continue

        evaluation_pairs.append((target_path, clone_path, metadata))

    total = len(samples)
    if missing:
        # This should be happen after above change
        logger.warning(
            "[VC] Missing cloned audio for %d of %d samples; excluding them from evaluation.",
            missing,
            total,
        )
    else:
        logger.info("[VC] Located cloned audio for all %d samples.", total)

    return evaluation_pairs


def _create_original_resolver(
    evaluation_conf: Optional[Dict],
    logger,
) -> Optional[Callable[[Path, dict], Optional[Path]]]:
    if not evaluation_conf:
        return None

    original_dir = evaluation_conf.get("original_audio_dir") if isinstance(evaluation_conf, dict) else evaluation_conf.get("original_audio_dir")
    if not original_dir:
        return None

    root_path = Path(to_absolute_path(str(original_dir))).expanduser().resolve()
    if not root_path.exists():
        logger.warning(
            "[VC] Provided original_audio_dir %s does not exist; falling back to dataset paths.",
            root_path,
        )
        return None

    logger.info("[VC] Indexing original audio files under %s for evaluation.", root_path)
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for wav_path in root_path.rglob("*.wav"):
        key = wav_path.name
        if key in index:
            duplicates.setdefault(key, []).append(wav_path)
            continue
        index[key] = wav_path.resolve()

    if duplicates:
        dup_preview = ", ".join(list(duplicates.keys())[:5])
        logger.warning(
            "[VC] Detected duplicate filenames in original_audio_dir; using first occurrence for %s%s",
            dup_preview,
            "..." if len(duplicates) > 5 else "",
        )

    def resolver(target_path: Path, metadata: dict) -> Optional[Path]:
        candidate = index.get(target_path.name)
        if candidate is None:
            return None
        return candidate

    return resolver

def _select_adversary(conf: DictConfig, dataset_conf: DictConfig, device: torch.device, logger):
    mode = str(conf.vc.mode).lower()
    model = str(conf.vc.get("model", "bertvits2")).lower()

    mode_registry = _ADVERSARY_REGISTRY.get(mode)
    if mode_registry is None:
        raise ValueError(f"Unsupported vc.mode '{conf.vc.mode}'. Expected one of: {', '.join(_ADVERSARY_REGISTRY)}")

    target = mode_registry.get(model)
    if target is None:
        raise ValueError(
            f"Unsupported adversary model '{model}' for mode '{mode}'. Available models: {', '.join(mode_registry.keys())}"
        )

    module_path, class_name = target.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import adversary module '{module_path}' for model '{model}'."
        ) from exc

    try:
        adversary_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Adversary class '{class_name}' not found in module '{module_path}'."
        ) from exc

    return adversary_cls(conf, dataset_conf, device, logger)


def run_vc_workflow(
    conf: DictConfig,
    base_dir: Path,
    device: torch.device,
    dataset,
    exp_dir: Path,
    logger,
    protected_audio_dir: Optional[Path] = None,
) -> Tuple[dict, Optional[Path], Path]:
    vc_conf = conf.vc
    dataset_conf = conf.dataset

    evaluation_conf = vc_conf.get("evaluation") if hasattr(vc_conf, "get") else None
    if evaluation_conf is not None and not isinstance(evaluation_conf, dict):
        evaluation_conf = OmegaConf.to_container(evaluation_conf, resolve=True)

    bootstrap_conf = evaluation_conf.get("bootstrap") if evaluation_conf else None
    if bootstrap_conf is None:
        bootstrap_conf = OmegaConf.select(conf, "evaluation.bootstrap", default=None)

    evaluate_only = bool(vc_conf.get("evaluate_only", False)) if hasattr(vc_conf, "get") else bool(getattr(vc_conf, "evaluate_only", False))

    override_generated_dir = None
    manual_max_samples = None
    max_generated_audio_seconds = None
    if evaluation_conf:
        generated_dir_value = evaluation_conf.get("generated_audio_dir")
        if generated_dir_value:
            override_generated_dir = Path(to_absolute_path(str(generated_dir_value))).expanduser().resolve()

        raw_manual_max = evaluation_conf.get("max_samples")
        if raw_manual_max is not None:
            try:
                manual_max_samples = int(raw_manual_max)
            except (TypeError, ValueError):
                logger.warning(
                    "[VC] Invalid evaluation.max_samples '%s'; ignoring override.",
                    raw_manual_max,
                )

        raw_max_duration = evaluation_conf.get("generated_audio_max_seconds")
        if raw_max_duration is not None:
            try:
                max_generated_audio_seconds = float(raw_max_duration)
            except (TypeError, ValueError):
                logger.warning(
                    "[VC] Invalid evaluation.generated_audio_max_seconds '%s'; ignoring override.",
                    raw_max_duration,
                )

    if max_generated_audio_seconds is None:
        raw_global_max = OmegaConf.select(conf, "evaluation.generated_audio_max_seconds", default=None)
        if raw_global_max is not None:
            try:
                max_generated_audio_seconds = float(raw_global_max)
            except (TypeError, ValueError):
                logger.warning(
                    "[VC] Invalid evaluation.generated_audio_max_seconds '%s'; ignoring override.",
                    raw_global_max,
                )

    original_resolver = _create_original_resolver(evaluation_conf, logger)

    if evaluate_only:
        if override_generated_dir is None:
            raise ValueError(
                "vc.evaluate_only is True but evaluation.generated_audio_dir was not provided."
            )
        generated_audio_dir = override_generated_dir
        if not generated_audio_dir.exists():
            raise FileNotFoundError(
                f"Generated audio directory {generated_audio_dir} not found for evaluation."
            )
        logger.info("[VC] Evaluation-only mode: skipping adversary attack.")
        logger.info("[VC] Using generated audio from %s", generated_audio_dir)
        synthesis_time_sec = None
        max_samples = manual_max_samples
    else:
        if override_generated_dir is not None:
            logger.info(
                "[VC] evaluation.generated_audio_dir is specified but evaluate_only is False; "
                "the path override will be ignored."
            )
        generated_audio_dir = exp_dir / "generated_audio"
        generated_audio_dir.mkdir(parents=True, exist_ok=True)
        adversary = _select_adversary(conf, dataset_conf, device, logger)

        raw_max_samples = getattr(adversary, "max_samples", None)
        if manual_max_samples is not None:
            max_samples = manual_max_samples
        elif raw_max_samples is not None:
            try:
                max_samples = int(raw_max_samples)
            except (TypeError, ValueError):
                logger.warning(
                    "[VC] Invalid max_samples '%s' for adversary %s; processing full dataset.",
                    raw_max_samples,
                    getattr(adversary, "__class__", type(adversary)).__name__,
                )
                max_samples = None
        else:
            max_samples = None

        synth_start = time.perf_counter()
        adversary.attack(
            output_path=str(generated_audio_dir),
            dataset=dataset,
            protected_audio_path=str(protected_audio_dir) if protected_audio_dir else None,
        )
        synthesis_time_sec = time.perf_counter() - synth_start

    logger.info("=" * 50)
    logger.info("Evaluating quality of generated audio...")
    evaluation_pairs = _collect_evaluation_pairs(
        dataset,
        generated_audio_dir,
        max_samples,
        logger,
        original_resolver=original_resolver,
    )
    generation_metrics = generation.evaluate_pairs(
        evaluation_pairs,
        str(generated_audio_dir),
        device,
        logger,
        synthesis_time_sec=synthesis_time_sec,
        bootstrap_config=bootstrap_conf,
        max_generated_audio_seconds=max_generated_audio_seconds,
    )
    logger.info(f"Generation Metrics: {generation_metrics}")

    return generation_metrics, protected_audio_dir, generated_audio_dir
