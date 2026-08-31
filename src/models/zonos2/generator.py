"""ZONOS2 generator wrapper (Zyphra/ZONOS2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class Zonos2GeneratorConfig:
    """Configuration required to run the ZONOS2 generator."""

    checkpoint: str = "Zyphra/ZONOS2"
    seed: Optional[int] = 42
    clean_speaker_background: bool = False
    accurate_mode: bool = True
    max_tokens: Optional[int] = None


class Zonos2Generator(BaseModel):
    """Thin wrapper around the upstream `zonos2` offline `TTSLLM` API.

    NOTE: TTSLLM's internal scheduler communicates over ZMQ IPC sockets and
    does not expose a `device`/`gpu_id` argument, so GPU placement for this
    model is controlled entirely via CUDA_VISIBLE_DEVICES on the process
    (set before this generator is constructed), not via `device: cuda:N`.
    """

    def __init__(
        self,
        config: Zonos2GeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.checkpoint),
            device=device,
            logger=logger,
        )
        self.config = config
        self._tts = None
        self._sampling_params_cls = None

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if self._tts is not None:
            return

        try:
            from zonos2.message import TTSSamplingParams
            from zonos2.tts import TTSLLM
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'zonos2'. Clone Zyphra/ZONOS2 and run `uv sync`."
            ) from exc

        self._sampling_params_cls = TTSSamplingParams
        self._tts = TTSLLM(model_path=self.config.checkpoint)
        self.model = getattr(self._tts, "model", None)
        self.logger.info("[ZONOS2] Loaded model from %s", self.config.checkpoint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio,
        prompt_text: Optional[str],
        sample_index: int,
        language: str = "en_us",
    ) -> Tuple[np.ndarray, int]:
        """Generate an utterance conditioned on a prompt clip and text."""

        del sample_index, prompt_text  # ZONOS2 conditions on reference audio only.

        self.ensure_model()
        assert self._tts is not None
        assert self._sampling_params_cls is not None

        speaker_embedding = self._tts.embed_speaker_file(str(prompt_audio))

        sampling_params = self._sampling_params_cls(seed=self.config.seed)
        result = self._tts.generate_one(
            text,
            sampling_params,
            language=language,
            speaker_embedding=speaker_embedding,
            clean_speaker_background=self.config.clean_speaker_background,
            accurate_mode=self.config.accurate_mode,
            max_tokens=self.config.max_tokens,
        )

        # result["audio"] is raw float32 PCM bytes (see TTSLLM.save_audio).
        audio_bytes = result["audio"]
        wav_np = np.frombuffer(audio_bytes, dtype=np.float32).copy()
        wav_np = np.atleast_1d(wav_np).astype(np.float32).flatten()

        sample_rate = int(result.get("sample_rate", 44100))
        return wav_np, sample_rate
