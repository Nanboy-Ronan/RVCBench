import json
import os
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


@hydra.main(version_base="1.3", config_path="configs", config_name=None)
def main(conf: DictConfig):
    base_dir = Path(to_absolute_path(conf.base_dir))
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")

    exp_dir = setup_exp(base_dir / "results", conf.run_name)
    logger = setup_logger("benchmark", exp_dir, conf.run_name, screen=True, tofile=True)
    logger.info(f"Using device: {device}")
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

    generation_metrics, _, generated_audio_dir = run_vc_workflow(
        conf, base_dir, device, dataset, exp_dir, logger
    )

    logger.info("=" * 50)
    all_metrics = {
        "run_name": conf.run_name,
        "config": OmegaConf.to_container(conf, resolve=True),
        "vc": {
            "mode": conf.vc.mode,
            "generated_audio_path": str(generated_audio_dir),
        },
        "generation_evaluation": generation_metrics,
    }
    metrics_filename = Path(conf.output_paths.metrics_file).name
    metrics_file = exp_dir / metrics_filename
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        # Some metric helpers return objects (e.g., AudioMetaData) that are not JSON serializable;
        # default=str keeps the dump resilient without altering the structure.
        json.dump(all_metrics, f, indent=2, default=str)
        f.write("\n")
    logger.info(f"Saved metrics to {metrics_file}")
    # Avoid JSON serialization issues in logging; the metrics file already contains the full object.
    logger.info("Run metrics saved. Preview run_name=%s, vc_mode=%s, generated_audio_path=%s",
                all_metrics["run_name"], all_metrics["vc"]["mode"], all_metrics["vc"]["generated_audio_path"])


if __name__ == "__main__":
    # python run_vc.py --config-name ots_vc/clean/bert_ots
    main()
