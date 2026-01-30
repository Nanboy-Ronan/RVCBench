"""Spark-TTS generator wrapper."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class SparkTTSGeneratorConfig:
    code_path: Path
    model_dir: str
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95


class SparkTTSGenerator(BaseModel):
    """Thin wrapper around Spark-TTS inference utilities."""

    def __init__(
        self,
        config: SparkTTSGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.model_dir),
            device=device,
            logger=logger,
        )
        self.config = config

        self._imports_loaded = False
        self._sparktts_cls = None

        self._model = None
        self._sample_rate: Optional[int] = None

        self._validate_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio: Path,
        prompt_text: str,
        sample_index: int,
    ) -> Tuple[np.ndarray, int]:
        """Generate an utterance conditioned on a prompt clip and text."""

        del sample_index  # Spark-TTS does not use a seed offset.

        self.ensure_model()

        prompt_text_arg = prompt_text or None

        try:
            wav = self._model.inference(
                text=text,
                prompt_speech_path=str(prompt_audio),
                prompt_text=prompt_text_arg,
                temperature=float(self.config.temperature),
                top_k=int(self.config.top_k),
                top_p=float(self.config.top_p),
            )
        except RuntimeError as exc:
            if prompt_text_arg is None:
                raise

            self.logger.warning(
                "[SparkTTS] Prompt transcript caused inference failure (%s); retrying without transcript.",
                exc,
            )
            wav = self._model.inference(
                text=text,
                prompt_speech_path=str(prompt_audio),
                prompt_text=None,
                temperature=float(self.config.temperature),
                top_k=int(self.config.top_k),
                top_p=float(self.config.top_p),
            )

        if isinstance(wav, torch.Tensor):
            wav_np = wav.detach().cpu().numpy()
        else:
            wav_np = np.asarray(wav)

        wav_np = np.atleast_1d(wav_np).astype(np.float32).flatten()

        sample_rate = self._sample_rate
        assert sample_rate is not None
        return wav_np, sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"Spark-TTS code path not found: {self.config.code_path}")

    def load_model(self) -> None:
        if self._model is not None:
            return

        self._ensure_imports()
        assert self._sparktts_cls is not None

        model_dir = Path(self.config.model_dir)
        if not model_dir.is_absolute():
            model_dir = (self.config.code_path / model_dir).resolve()

        if not model_dir.exists():
            raise FileNotFoundError(f"Spark-TTS model directory not found: {model_dir}")

        self._model = self._sparktts_cls(model_dir, device=self.device)
        self._sample_rate = int(getattr(self._model, "sample_rate", 16000))
        self.logger.info("[SparkTTS] Loaded model from %s", model_dir)

    def _ensure_imports(self) -> None:
        if self._imports_loaded:
            return

        import sys

        code_path = str(self.config.code_path)
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        cache_root = self.config.code_path / ".cache" / "huggingface"
        os.environ.setdefault("HF_HOME", str(cache_root))
        cache_root.mkdir(parents=True, exist_ok=True)

        try:
            sparktts_mod = __import__("cli.SparkTTS", fromlist=["SparkTTS"])
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive
            raise ModuleNotFoundError(
                "Failed to import Spark-TTS. Ensure 'code_path' points to the"
                " Spark-TTS repository and dependencies are installed."
            ) from exc

        self._sparktts_cls = getattr(sparktts_mod, "SparkTTS", None)
        if self._sparktts_cls is None:
            raise ImportError("SparkTTS class not found in cli.SparkTTS")

        self._imports_loaded = True
