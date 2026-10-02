"""VoxCPM generator wrapper."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from rvcbench.models.model import BaseModel


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
    seed: Optional[int] = None
    native_seed_policy: str = 'source_index'


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
        self.device = torch.device(device)
        self._voxcpm_cls = None
        self._pipeline = None
        self._supports_reference_audio: Optional[bool] = None
        self.last_native_seed = None
        self.last_native_requested_seed = None

    def _device_scope(self):
        return torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()

    def load_model(self) -> None:
        if self._pipeline is not None:
            return

        self._ensure_pythonpath()
        if self.config.native_seed_policy != 'source_index':
            raise ValueError('VoxCPM native_seed_policy must be source_index')

        try:
            module = importlib.import_module("voxcpm")
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'voxcpm'. Install it with `pip install voxcpm` "
                "or set adversary.code_path to a local VoxCPM checkout."
            ) from exc

        self._voxcpm_cls = getattr(module, "VoxCPM", None)
        if self.config.code_path and not Path(module.__file__).resolve().is_relative_to(Path(self.config.code_path).expanduser().resolve()):
            raise RuntimeError('VoxCPM runtime was imported from outside configured code_path')
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

        try:
            with self._device_scope():
                self._pipeline = self._voxcpm_cls.from_pretrained(str(self.config.model_path), **load_kwargs)
        except BaseException:
            self.close()
            raise
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
        sample_index: int = 0,
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
        elif reference_wav_path and not self._supports_reference_audio:
            raise ValueError('Loaded VoxCPM model does not support reference_wav_path; select VoxCPM2 or explicitly use transcript-prompt conditioning only')
        if bool(prompt_wav_path) != bool(prompt_text):
            raise ValueError('VoxCPM prompt_wav_path and prompt_text must be supplied together')
        if prompt_wav_path and prompt_text:
            kwargs["prompt_wav_path"] = str(prompt_wav_path)
            kwargs["prompt_text"] = str(prompt_text)

        self.last_native_requested_seed = None if self.config.seed is None else int(self.config.seed) + int(sample_index)
        kwargs['seed'] = self.last_native_requested_seed
        with self._device_scope(), torch.inference_mode():
            wav = self._pipeline.generate(**kwargs)
        effective_seed = getattr(getattr(self._pipeline, 'tts_model', None), 'last_successful_seed', None)
        self.last_native_seed = int(effective_seed) if effective_seed is not None else self.last_native_requested_seed
        audio = wav.detach().cpu().float().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
        if audio.ndim > 2 or (audio.ndim == 2 and 1 not in audio.shape):
            raise RuntimeError('VoxCPM must return one mono waveform')
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not audio.size or not np.isfinite(audio).all():
            raise RuntimeError('VoxCPM returned empty or nonfinite waveform')

        sample_rate = int(getattr(getattr(self._pipeline, "tts_model", None), "sample_rate", 48000))
        return audio, sample_rate

    def close(self):
        self._pipeline = self.model = self._voxcpm_cls = None
        self._supports_reference_audio = None
        self._model_ready = False
        self.last_native_seed = None
        self.last_native_requested_seed = None

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
