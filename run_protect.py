import json
import torch
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src import datasets
from src.evaluation import fidelity
from src.workflows.vc import run_vc_workflow
from src.utils.logger import log_config, setup_exp, setup_logger
from src import protection
from src.utils.seeding import configure_seeds

@hydra.main(version_base="1.3", config_path="configs", config_name=None)
def main(conf: DictConfig):
    # 1) Setup basics and resolve paths (Hydra changes the CWD, so fall back to to_absolute_path)
    base_dir = Path(to_absolute_path(conf.base_dir))
    device = torch.device(conf.device if torch.cuda.is_available() else "cpu")
    protection_conf = conf.protection if "protection" in conf else None
    bootstrap_conf = None
    protection_eval_conf = OmegaConf.select(conf, "protection.evaluation", default=None)
    if protection_eval_conf is not None and not isinstance(protection_eval_conf, dict):
        protection_eval_conf = OmegaConf.to_container(protection_eval_conf, resolve=True)

    evaluate_only = False
    override_protected_dir = None
    if protection_conf is not None:
        evaluate_only = bool(protection_conf.get("evaluate_only", False)) if hasattr(
            protection_conf, "get"
        ) else bool(getattr(protection_conf, "evaluate_only", False))

    if isinstance(protection_eval_conf, dict):
        bootstrap_conf = protection_eval_conf.get("bootstrap")
        protected_dir_value = protection_eval_conf.get("protected_audio_dir")
        if protected_dir_value:
            override_protected_dir = Path(to_absolute_path(str(protected_dir_value))).expanduser().resolve()

    if bootstrap_conf is None:
        bootstrap_conf = OmegaConf.select(conf, "evaluation.bootstrap", default=None)

    exp_dir = setup_exp(base_dir / "results", conf['run_name'])

    logger = setup_logger("benchmark", exp_dir, conf.run_name, screen=True, tofile=True)
    logger.info(f"Using device: {device}")
    log_config(logger, conf)

    seed_value = OmegaConf.select(conf, "seed", default=None)
    if seed_value is None:
        seed_value = OmegaConf.select(conf, "protection.seed", default=None)
    if seed_value is None:
        seed_value = OmegaConf.select(conf, "adversary.seed", default=None)
    if seed_value is None:
        logger.info("No seed specified in config; defaulting to 42.")
        seed_value = 42
    configure_seeds(seed_value, logger=logger)

    # 2) Dataset
    # 1. Load Dataset
    logger.info("=" * 50)
    logger.info("Loading dataset...")

    dataset = datasets.AllSpeakerData(conf.protection, conf.dataset, logger)

    has_protection = protection_conf is not None

    if not has_protection:
        logger.info("=" * 50)
        logger.info("Running zero-shot VC evaluation without protection stage...")
        generation_metrics, safespeech_protected_path, generated_audio_dir = run_vc_workflow(
            conf, base_dir, device, dataset, exp_dir, logger
        )

        logger.info("=" * 50)
        all_metrics = {
            "run_name": conf.run_name,
            "config": OmegaConf.to_container(conf, resolve=True),
            "vc": {
                "mode": conf.vc.mode,
                "protected_audio_path": str(safespeech_protected_path),
                "generated_audio_path": str(generated_audio_dir),
            },
            "generation_evaluation": generation_metrics,
        }
        metrics_filename = Path(conf.output_paths.metrics_file).name
        metrics_file = exp_dir / metrics_filename
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
            f.write("\n")
        logger.info(f"Saved metrics to {metrics_file}")
        logger.info("Run metrics:\n%s", json.dumps(all_metrics, indent=2))
        return

    # 3) Protection (SafeSpeech)
    logger.info("=" * 50)

    if evaluate_only:
        if override_protected_dir is None:
            raise ValueError(
                "protection.evaluate_only is True but protection.evaluation.protected_audio_dir was not provided."
            )
        if not override_protected_dir.exists():
            raise FileNotFoundError(
                f"Protected audio directory {override_protected_dir} not found for evaluation."
            )
        logger.info("[Protection] Evaluation-only mode: skipping protection stage.")
        logger.info("[Protection] Using protected audio from %s", override_protected_dir)
        safespeech_protected_path = override_protected_dir
    else:
        if override_protected_dir is not None:
            logger.info(
                "[Protection] protection.evaluation.protected_audio_dir is specified but evaluate_only is False; "
                "the path override will be ignored."
            )
        logger.info(f"Applying protection: {conf.protection.name}")

        run_protected_audio_dir = exp_dir / "protected_audio"
        run_protected_audio_dir.mkdir(parents=True, exist_ok=True)
        protector_name = getattr(protection, conf.protection.name)
        protector = protector_name(
            model_config=conf.model,
            dataset_config=conf.dataset,
            logger=logger,
            output_dir=str(run_protected_audio_dir),
            config=conf.protection,
            device=device,
        )

        protector.protect()
        safespeech_protected_path = Path(protector.output_dir)

    # 4) Fidelity evaluation
    logger.info("=" * 50)
    logger.info("Evaluating fidelity of protected audio...")
    train_transcripts = dataset.get_train_files_map()
    original_train_files = list(train_transcripts.keys())
    logger.info(f"Protected audio saved to {safespeech_protected_path}")
    fidelity_metrics = fidelity.evaluate(
        original_train_files,
        str(safespeech_protected_path),
        logger,
        target_sr=conf.protection.sampling_rate, 
        transcript_map=train_transcripts,
        bootstrap_config=bootstrap_conf,
        sample_metrics_path=exp_dir / "fidelity_sample_metrics.csv",
    )
    logger.info(f"Fidelity Metrics: {fidelity_metrics}")

    # 5) Save results
    logger.info("=" * 50)
    all_metrics = {
        "run_name": conf.run_name,
        "config": OmegaConf.to_container(conf, resolve=True),
        "fidelity_evaluation": fidelity_metrics,
        "protection": {
            "protected_audio_path": str(safespeech_protected_path),
        },
    }
    metrics_filename = Path(conf.output_paths.metrics_file).name
    metrics_file = exp_dir / metrics_filename
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = fidelity.make_serializable(all_metrics)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
        f.write("\n")
    logger.info(f"Saved metrics to {metrics_file}")
    logger.info("Run metrics:\n%s", json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    '''
    # Call by python run_protect.py --config-name safespeech_on_libritts
    # '''
    main()
