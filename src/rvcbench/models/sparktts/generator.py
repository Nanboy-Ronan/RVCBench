"""Spark-TTS generator wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import importlib
import numpy as np
import torch

from rvcbench.models.model import BaseModel
from .assets import model_directory, checkpoint_files, checked_bicodec_loading


@dataclass
class SparkTTSGeneratorConfig:
    code_path: Path
    model_dir: str
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    seed: Optional[int] = None
    retry_without_prompt_text: bool = False


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

        if not isinstance(config.retry_without_prompt_text, bool):
            raise ValueError("SparkTTS retry_without_prompt_text must be boolean")
        self.last_conditioning_variant = None
        self.initialization_metadata = None
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

        self.last_conditioning_variant = None
        self.ensure_model()
        if self.config.seed is not None:
            from rvcbench.utils.seeding import configure_seeds
            configure_seeds(int(self.config.seed) + int(sample_index), logger=None)

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
            if prompt_text_arg is None or not self.config.retry_without_prompt_text:
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

            self.last_conditioning_variant = "prompt_audio_only_after_transcript_failure"

        if self.last_conditioning_variant is None:
            self.last_conditioning_variant = "prompt_audio_and_text" if prompt_text_arg else "prompt_audio_only"

        if isinstance(wav, torch.Tensor):
            wav_np = wav.detach().cpu().numpy()
        else:
            wav_np = np.asarray(wav)

        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim != 1 or not wav_np.size or not np.isfinite(wav_np).all():
            raise ValueError("SparkTTS output must be nonempty finite mono audio")

        sample_rate = self._sample_rate
        assert sample_rate is not None
        return wav_np, sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def close(self):
        self._model = None
        self._sample_rate = None
        self.initialization_metadata = None
        self._model_ready = False
        self.last_conditioning_variant = None

    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"Spark-TTS code path not found: {self.config.code_path}")

    def load_model(self) -> None:
        if self._model is not None:
            return

        model_dir = model_directory(self.config.code_path, self.config.model_dir)
        checkpoint_files(model_dir)
        self._ensure_imports()
        assert self._sparktts_cls is not None
        codec = importlib.import_module('sparktts.models.bicodec').BiCodec
        receipts = []
        with checked_bicodec_loading(codec, receipts):
            self._model = self._sparktts_cls(model_dir, device=self.device)
        if not receipts:
            self._model = None
            raise RuntimeError('SparkTTS native startup did not validate its BiCodec checkpoint')
        self.initialization_metadata = {'bicodec': receipts}
        self._sample_rate = int(getattr(self._model, "sample_rate", 16000))
        self.logger.info("[SparkTTS] Loaded model from %s", model_dir)

    def _ensure_imports(self) -> None:
        if self._imports_loaded:
            return

        import sys

        code_path = str(self.config.code_path)
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

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
