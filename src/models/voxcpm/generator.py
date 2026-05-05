"""VoxCPM generator wrapper."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class VoxCPMGeneratorConfig:
    """Configuration options for VoxCPM voice cloning."""

    model_path: str = "openbmb/VoxCPM2"
    code_path: Optional[str] = None
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    optimize: bool = False
    load_denoiser: bool = False
    zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base"
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    min_len: int = 2
    max_len: int = 4096
    normalize: bool = False
    denoise: bool = False
    retry_badcase: bool = True
    retry_badcase_max_times: int = 3
    retry_badcase_ratio_threshold: float = 6.0


class VoxCPMGenerator(BaseModel):
    """Thin wrapper around the upstream VoxCPM Python API."""

    def __init__(self, config: VoxCPMGeneratorConfig, device: torch.device, logger) -> None:
        materialised_config = replace(config)
        super().__init__(
            model_name_or_path=str(materialised_config.model_path),
            device=device,
            logger=logger,
        )
        self.config = materialised_config
        self._voxcpm_cls = None
        self._pipeline = None
        self._supports_reference_audio: Optional[bool] = None
        self._warned_reference_audio_ignored = False

    def load_model(self) -> None:
        if self._pipeline is not None:
            return

        self._ensure_pythonpath()

        try:
            module = importlib.import_module("voxcpm")
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'voxcpm'. Install it with `pip install voxcpm` "
                "or set adversary.code_path to a local VoxCPM checkout."
            ) from exc

        self._voxcpm_cls = getattr(module, "VoxCPM", None)
        if self._voxcpm_cls is None:
            raise ImportError("voxcpm does not expose VoxCPM.")

        load_kwargs = {
            "load_denoiser": bool(self.config.load_denoiser),
            "zipenhancer_model_id": str(self.config.zipenhancer_model_id),
            "local_files_only": bool(self.config.local_files_only),
            "optimize": bool(self.config.optimize),
            "device": str(self.device),
        }
        if self.config.cache_dir not in (None, ""):
            load_kwargs["cache_dir"] = str(self.config.cache_dir)

        self._pipeline = self._voxcpm_cls.from_pretrained(
            str(self.config.model_path),
            **load_kwargs,
        )
        self.model = getattr(self._pipeline, "tts_model", self._pipeline)
        tts_model = getattr(self._pipeline, "tts_model", None)
        self._supports_reference_audio = bool(
            tts_model is not None and tts_model.__class__.__name__ == "VoxCPM2Model"
        )

    def generate(
        self,
        *,
        text: str,
        reference_wav_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        self.ensure_model()
        assert self._pipeline is not None

        kwargs = {
            "text": str(text),
            "cfg_value": float(self.config.cfg_value),
            "inference_timesteps": int(self.config.inference_timesteps),
            "min_len": int(self.config.min_len),
            "max_len": int(self.config.max_len),
            "normalize": bool(self.config.normalize),
            "denoise": bool(self.config.denoise),
            "retry_badcase": bool(self.config.retry_badcase),
            "retry_badcase_max_times": int(self.config.retry_badcase_max_times),
            "retry_badcase_ratio_threshold": float(self.config.retry_badcase_ratio_threshold),
        }
        if reference_wav_path and self._supports_reference_audio:
            kwargs["reference_wav_path"] = str(reference_wav_path)
        elif reference_wav_path and not self._supports_reference_audio and not self._warned_reference_audio_ignored:
            self.logger.info(
                "Ignoring reference_wav_path because the loaded checkpoint does not support VoxCPM2 reference-audio cloning."
            )
            self._warned_reference_audio_ignored = True
        if prompt_wav_path and prompt_text:
            kwargs["prompt_wav_path"] = str(prompt_wav_path)
            kwargs["prompt_text"] = str(prompt_text)

        wav = self._pipeline.generate(**kwargs)
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        sample_rate = int(getattr(getattr(self._pipeline, "tts_model", None), "sample_rate", 48000))
        return audio, sample_rate

    def _ensure_pythonpath(self) -> None:
        raw = self.config.code_path
        if raw in (None, ""):
            return

        root = Path(str(raw)).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"VoxCPM code_path not found: {root}")
        root = root.resolve()

        candidates = [root]
        src_dir = root / "src"
        if src_dir.exists():
            candidates.insert(0, src_dir)

        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
