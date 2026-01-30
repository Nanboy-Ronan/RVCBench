import json
from dataclasses import replace
from pathlib import Path

from src.utils.env import configure_offline_env

configure_offline_env(default=False)

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src import datasets
from src.workflows.vc import run_vc_workflow
from src.utils.logger import log_config, setup_exp, setup_logger
from src.utils.seeding import configure_seeds


def _resolve_protected_audio_dir(conf: DictConfig) -> Path:
    """
    Resolve the protected audio directory from a few likely config keys.
    Preference order: protected_audio_dir (top-level), vc.protected_audio_dir,
    vc.evaluation.protected_audio_dir.
    """
    candidates = [
        OmegaConf.select(conf, "protected_audio_dir", default=None),
    ]
    for value in candidates:
        if value:
            return Path(to_absolute_path(str(value))).expanduser().resolve()
    return None


def _override_prompt_paths(dataset, protected_dir: Path, logger):
    """
    Update zero-shot samples to pull prompts from the protected audio directory.
    Falls back to the original prompt path if a protected copy is missing.
    """
    protected_dir = protected_dir.expanduser().resolve()
    samples = dataset.get_zero_shot_samples()
    updated_samples = []
    missing = 0

    for sample in samples:
        prompt_path = sample.prompt_path
        new_prompt = prompt_path
        if prompt_path:
            original = Path(str(prompt_path))
            speaker_hint = str(sample.speaker_id)
            candidates = [
                protected_dir / speaker_hint / original.name,
            ]
            parent_name = original.parent.name
            if parent_name and parent_name != speaker_hint:
                candidates.append(protected_dir / parent_name / original.name)
            candidates.append(protected_dir / original.name)

            selected = next((c for c in candidates if c.exists()), None)
            if selected is not None:
                new_prompt = selected
            else:
                missing += 1
                raise RuntimeError(f"Protected audio not found for prompt: {original}")
                if logger:
                    logger.warning(
                        "[VC Protect] Missing protected audio for %s; using clean prompt.", original
                    )
                    
        updated_samples.append(replace(sample, prompt_path=new_prompt))

    dataset._zero_shot_samples = updated_samples
    dataset._zero_shot_samples_loaded = True
    dataset._zero_shot_eval_cache = None
    return missing, len(samples)


@hydra.main(version_base="1.3", config_path="configs", config_name=None)
def main(conf: DictConfig):
    base_dir = Path(to_absolute_path(conf.base_dir))
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")

    protected_audio_dir = _resolve_protected_audio_dir(conf)
    if protected_audio_dir is None:
        raise ValueError(
            "protected_audio_dir is required. "
            "Set it via `protected_audio_dir=...`, `vc.protected_audio_dir=...`, "
            "or `vc.evaluation.protected_audio_dir=...`."
        )
    if not protected_audio_dir.exists():
        raise FileNotFoundError(f"Protected audio directory not found: {protected_audio_dir}")

    exp_dir = setup_exp(base_dir / "results", conf.run_name)
    logger = setup_logger("benchmark", exp_dir, conf.run_name, screen=True, tofile=True)
    logger.info(f"Using device: {device}")
    logger.info(f"Using protected audio from: {protected_audio_dir}")
    log_config(logger, conf)

    seed_value = OmegaConf.select(conf, "seed", default=None)
    if seed_value is None:
        seed_value = OmegaConf.select(conf, "adversary.seed", default=None)
    if seed_value is None:
        seed_value = OmegaConf.select(conf, "vc.seed", default=None)
    if seed_value is None:
        logger.info("No seed specified in config; defaulting to 42.")
        seed_value = 42
    configure_seeds(seed_value, logger=logger)

    dataset = datasets.AllSpeakerData(conf, conf.dataset, logger)
    missing_count, total = _override_prompt_paths(dataset, protected_audio_dir, logger)
    if missing_count:
        logger.warning(
            "[VC Protect] %d/%d prompts do not have protected copies; falling back to clean audio.",
            missing_count,
            total,
        )
    else:
        logger.info("[VC Protect] Found protected prompts for all %d samples.", total)

    generation_metrics, _, generated_audio_dir = run_vc_workflow(
        conf, base_dir, device, dataset, exp_dir, logger, protected_audio_dir=protected_audio_dir
    )

    logger.info("=" * 50)
    all_metrics = {
        "run_name": conf.run_name,
        "config": OmegaConf.to_container(conf, resolve=True),
        "vc": {
            "mode": conf.vc.mode,
            "protected_audio_path": str(protected_audio_dir),
            "generated_audio_path": str(generated_audio_dir),
        },
        "generation_evaluation": generation_metrics,
    }
    metrics_filename = Path(conf.output_paths.metrics_file).name
    metrics_file = exp_dir / metrics_filename
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)
        f.write("\n")
    logger.info(f"Saved metrics to {metrics_file}")
    logger.info(
        "Run metrics saved. Preview run_name=%s, vc_mode=%s, protected_audio_path=%s, generated_audio_path=%s",
        all_metrics["run_name"],
        all_metrics["vc"]["mode"],
        all_metrics["vc"]["protected_audio_path"],
        all_metrics["vc"]["generated_audio_path"],
    )


if __name__ == "__main__":
    main()
