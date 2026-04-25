"""Qwen3-TTS generator wrapper."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class Qwen3TTSGeneratorConfig:
    """Configuration options for Qwen3-TTS voice cloning."""

    checkpoint_path: str
    torch_dtype: Optional[str] = "auto"
    device_map: Optional[str] = "auto"
    use_flash_attn2: bool = False
    attn_implementation: Optional[str] = None
    seed: Optional[int] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


class Qwen3TTSGenerator(BaseModel):
    """Thin wrapper around the qwen-tts Python package."""

    def __init__(self, config: Qwen3TTSGeneratorConfig, device: torch.device, logger) -> None:
        materialised_config = replace(config)
        super().__init__(
            model_name_or_path=str(materialised_config.checkpoint_path),
            device=device,
            logger=logger,
        )
        self.config = materialised_config
        self._model_cls = None
        self._model = None

    def load_model(self) -> None:
        if self._model is not None:
            return

        try:
            module = importlib.import_module("qwen_tts")
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'qwen-tts'. Install it with `pip install -U qwen-tts`."
            ) from exc

        self._model_cls = getattr(module, "Qwen3TTSModel", None)
        if self._model_cls is None:
            raise ImportError("qwen_tts does not expose Qwen3TTSModel.")

        load_kwargs: Dict[str, Any] = {}
        dtype = self._parse_torch_dtype(self.config.torch_dtype)
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        if self.config.device_map not in (None, ""):
            load_kwargs["device_map"] = self.config.device_map

        attn_implementation = self.config.attn_implementation
        if not attn_implementation and self.config.use_flash_attn2:
            attn_implementation = "flash_attention_2"
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self._model = self._model_cls.from_pretrained(
            str(self.config.checkpoint_path),
            **load_kwargs,
        )
        self.model = self._model

    def create_voice_clone_prompt(
        self,
        *,
        ref_audio: Any,
        ref_text: Optional[str],
        x_vector_only_mode: bool = False,
    ) -> Any:
        self.ensure_model()
        assert self._model is not None

        kwargs: Dict[str, Any] = {
            "ref_audio": ref_audio,
            "x_vector_only_mode": bool(x_vector_only_mode),
        }
        if ref_text:
            kwargs["ref_text"] = ref_text
        return self._model.create_voice_clone_prompt(**kwargs)

    def generate(
        self,
        *,
        text: str,
        language: Optional[str],
        ref_audio: Any = None,
        ref_text: Optional[str] = None,
        voice_clone_prompt: Any = None,
        x_vector_only_mode: bool = False,
        sample_index: int = 0,
    ) -> Tuple[np.ndarray, int]:
        self.ensure_model()
        assert self._model is not None

        self._apply_seed(sample_index)

        kwargs: Dict[str, Any] = {
            "text": text,
        }
        if language:
            kwargs["language"] = language
        if voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = voice_clone_prompt
        else:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
            kwargs["x_vector_only_mode"] = bool(x_vector_only_mode)

        kwargs.update(self.config.generation_kwargs)
        wavs, sample_rate = self._model.generate_voice_clone(**kwargs)
        if not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio for the provided prompt.")

        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        wav = np.clip(wav, -1.0, 1.0)
        return wav, int(sample_rate)

    def _apply_seed(self, sample_index: int) -> None:
        if self.config.seed is None:
            return
        seed = int(self.config.seed) + int(sample_index)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _parse_torch_dtype(self, value: Optional[str]) -> Optional[Any]:
        if value is None:
            return None
        if not isinstance(value, str):
            return value

        token = value.strip().lower()
        if token in {"", "none"}:
            return None
        if token == "auto":
            return "auto"
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if token in dtype_map:
            return dtype_map[token]
        raise ValueError(f"Unsupported torch_dtype for Qwen3-TTS: {value}")
