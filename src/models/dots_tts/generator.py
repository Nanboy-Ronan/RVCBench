"""dots.tts generator wrapper (rednote-hilab/dots.tts)."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class DotsTTSGeneratorConfig:
    """Configuration required to run the dots.tts generator."""

    checkpoint: str = "rednote-hilab/dots.tts-soar"
    precision: str = "bfloat16"
    optimize: bool = True
    num_steps: int = 10
    guidance_scale: float = 1.2


class DotsTTSGenerator(BaseModel):
    """Thin wrapper around the upstream `dots.tts` Python runtime."""

    def __init__(
        self,
        config: DotsTTSGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.checkpoint),
            device=device,
            logger=logger,
        )
        self.config = config
        self._runtime = None

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if self._runtime is not None:
            return

        try:
            module = importlib.import_module("dots_tts.runtime")
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'dots.tts'. Install it with `pip install dots.tts`."
            ) from exc

        runtime_cls = getattr(module, "DotsTtsRuntime", None)
        if runtime_cls is None:
            raise ImportError("dots_tts.runtime does not expose DotsTtsRuntime.")

        # DotsTtsRuntime.from_pretrained() takes no device argument; it always
        # binds to torch.device("cuda"), i.e. whatever the *current default*
        # CUDA device is. Set that explicitly so `device: cuda:N` in the
        # benchmark config is actually honoured instead of always landing on
        # GPU 0.
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        self._runtime = runtime_cls.from_pretrained(
            self.config.checkpoint,
            precision=self.config.precision,
            optimize=self.config.optimize,
        )
        self.model = getattr(self._runtime, "model", None)
        self.logger.info("[DotsTTS] Loaded runtime from %s", self.config.checkpoint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio,
        prompt_text: Optional[str],
        sample_index: int,
        language: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate an utterance conditioned on a prompt clip and text."""

        del sample_index  # dots.tts does not take a seed offset here.

        self.ensure_model()
        assert self._runtime is not None

        result = self._runtime.generate(
            text=text,
            prompt_audio_path=str(prompt_audio),
            prompt_text=prompt_text or "",
            language=language,
            num_steps=int(self.config.num_steps),
            guidance_scale=float(self.config.guidance_scale),
        )

        audio = result["audio"]
        if isinstance(audio, torch.Tensor):
            wav_np = audio.detach().float().cpu().numpy()
        else:
            wav_np = np.asarray(audio, dtype=np.float32)

        wav_np = np.atleast_1d(wav_np).astype(np.float32).flatten()
        sample_rate = int(result.get("sample_rate", 48000))
        return wav_np, sample_rate
