import logging
import os
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S"
    )

    # Prevent duplicate handlers if setup_logger is invoked multiple times
    if lg.handlers:
        for handler in list(lg.handlers):
            lg.removeHandler(handler)

    # Always capture everything at the logger level so handlers can filter
    lg.setLevel(logging.DEBUG if tofile else level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        lg.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        lg.addHandler(sh)

    return lg

def setup_exp(base_dir, run_name):
    exp_dir = os.path.join(base_dir, run_name, get_timestamp())
    os.makedirs(exp_dir, exist_ok=True)
    exp_dir = Path(exp_dir)
    protected_audio_dir = exp_dir / "protected_audio"
    generated_audio_dir = exp_dir / "generated_audio"
    purturbed_noise_dir = exp_dir / "purturbed_noise"
    protected_audio_dir.mkdir(parents=True, exist_ok=True)
    generated_audio_dir.mkdir(parents=True, exist_ok=True)
    purturbed_noise_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir

def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def log_config(logger, conf, *, resolve=True):
    """Dump the active Hydra config to the configured logger."""
    if conf is None:
        logger.info("Configuration: <none>")
        return

    try:
        conf_serialized = OmegaConf.to_yaml(conf, resolve=resolve)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.error("Failed to serialize config: %s", exc)
        return

    logger.info("Active configuration:\n%s", conf_serialized)
