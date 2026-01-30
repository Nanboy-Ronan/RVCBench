"""VibeVoice generator wrapper for off-the-shelf adversary runs."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class VibeVoiceGeneratorConfig:
    code_path: Path
    model_path: str
    checkpoint_path: Optional[str] = None
    torch_dtype: str = "auto"
    attn_implementation: str = "flash_attention_2"
    allow_attn_fallback: bool = True
    cfg_scale: float = 1.3
    disable_prefill: bool = False
    num_inference_steps: Optional[int] = 10
    max_new_tokens: Optional[int] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    verbose: bool = False


class VibeVoiceGenerator(BaseModel):
    """Thin wrapper around the community VibeVoice inference stack."""

    def __init__(
        self,
        config: VibeVoiceGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.model_path),
            device=device,
            logger=logger,
        )
        self.config = config

        self.config.code_path = Path(self.config.code_path).expanduser().resolve()
        self._ensure_repo_on_path()

        self.processor = None
        self.sample_rate = 24000
        self._torch_dtype = self._resolve_dtype(self.config.torch_dtype)
        self._attn_impl = (self.config.attn_implementation or "flash_attention_2").strip()
        self._generation_kwargs = dict(self.config.generation_kwargs or {})
        self._generation_kwargs.setdefault("generation_config", {"do_sample": False})
        self._target_device = torch.device(self._device_string())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        self._ensure_repo_on_path()

        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        try:
            from vibevoice.modular.lora_loading import load_lora_assets
        except ModuleNotFoundError:
            load_lora_assets = None

        self.logger.info("[VibeVoice] Loading processor from %s", self.config.model_path)
        self.processor = VibeVoiceProcessor.from_pretrained(self.config.model_path)
        self.sample_rate = getattr(
            getattr(self.processor, "audio_processor", None),
            "sampling_rate",
            24000,
        )

        load_kwargs = {
            "torch_dtype": self._torch_dtype,
            "attn_implementation": self._attn_impl,
            "low_cpu_mem_usage": False,
        }

        try:
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.config.model_path,
                **load_kwargs,
            )
        except Exception as exc:
            if not self.config.allow_attn_fallback or self._attn_impl != "flash_attention_2":
                raise
            self.logger.warning(
                "[VibeVoice] flash_attention_2 load failed (%s); retrying with sdpa.",
                exc,
            )
            load_kwargs["attn_implementation"] = "sdpa"
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.config.model_path,
                **load_kwargs,
            )
            self._attn_impl = "sdpa"

        if self.device.type == "cpu":
            self._target_device = torch.device("cpu")
        elif self.device.type == "mps":
            self._target_device = torch.device("mps")
            self.model.to(self._target_device)
        else:
            self._target_device = torch.device(self._device_string())
            self.model.to(self._target_device)

        if self.config.num_inference_steps is not None:
            self.model.set_ddpm_inference_steps(self.config.num_inference_steps)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path:
            if load_lora_assets is None:
                raise ModuleNotFoundError(
                    "vibevoice.modular.lora_loading is unavailable; install VibeVoice dependencies to load checkpoints."
                )
            self.logger.info("[VibeVoice] Loading LoRA assets from %s", checkpoint_path)
            report = load_lora_assets(self.model, checkpoint_path)
            loaded = [
                name
                for name, flag in (
                    ("language LoRA", getattr(report, "language_model", False)),
                    ("diffusion head LoRA", getattr(report, "diffusion_head_lora", False)),
                    ("diffusion head weights", getattr(report, "diffusion_head_full", False)),
                    ("acoustic connector", getattr(report, "acoustic_connector", False)),
                    ("semantic connector", getattr(report, "semantic_connector", False)),
                )
                if flag
            ]
            if loaded:
                self.logger.info("[VibeVoice] Loaded %s", ", ".join(loaded))
            else:
                self.logger.warning("[VibeVoice] No adapter components were loaded; please verify checkpoint contents.")

        self.model.eval()

    def generate(self, text: str, reference_audio: Path) -> Tuple[np.ndarray, int]:
        text = (text or "").strip()
        if not text:
            raise ValueError("VibeVoice requires non-empty text input.")
        if reference_audio is None or not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        self.ensure_model()
        assert self.processor is not None

        inputs = self.processor(
            text=[text],
            voice_samples=[str(reference_audio)],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(self._target_device)

        generation_kwargs = {
            "tokenizer": self.processor.tokenizer,
            "cfg_scale": self.config.cfg_scale,
            "verbose": bool(self.config.verbose),
            "is_prefill": not self.config.disable_prefill,
        }
        if self.config.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = self.config.max_new_tokens
        if self._generation_kwargs:
            generation_kwargs.update(self._generation_kwargs)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        speech_outputs = getattr(outputs, "speech_outputs", None)
        if not speech_outputs or speech_outputs[0] is None:
            raise RuntimeError("VibeVoice did not return any waveform.")

        audio = speech_outputs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()

        return audio, int(self.sample_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_repo_on_path(self) -> None:
        repo_path = self.config.code_path
        if not repo_path.exists():
            raise FileNotFoundError(f"VibeVoice code path not found: {repo_path}")
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        current = os.environ.get("PYTHONPATH", "")
        if repo_str not in [entry for entry in current.split(os.pathsep) if entry]:
            os.environ["PYTHONPATH"] = f"{repo_str}{os.pathsep}{current}" if current else repo_str

    def _device_string(self) -> str:
        if self.device.type == "cuda":
            index = 0 if self.device.index is None else self.device.index
            return f"cuda:{index}"
        if self.device.type == "mps":
            return "mps"
        return "cpu"

    def _resolve_dtype(self, value: Optional[str]) -> torch.dtype:
        token = (value or "auto").strip().lower()
        if token == "auto":
            if self.device.type == "cuda":
                return torch.bfloat16
            return torch.float32
        if token in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if token in {"fp16", "float16", "half"}:
            return torch.float16
        return torch.float32
