"""dots.tts generator wrapper (rednote-hilab/dots.tts)."""
from __future__ import annotations

import importlib
import sys
import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from rvcbench.models.model import BaseModel


@dataclass
class DotsTTSGeneratorConfig:
    """Configuration required to run the dots.tts generator."""

    checkpoint: str = "rednote-hilab/dots.tts-soar"
    precision: str = "bfloat16"
    optimize: bool = True
    num_steps: int = 10
    guidance_scale: float = 1.2
    code_path: Optional[Path] = None


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
        self._runtime_matmul_precision = None

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if self._runtime is not None:
            return

        if self.config.code_path:
            root = Path(self.config.code_path).expanduser().resolve()
            import_root = root.parent if root.name == 'dots_tts' else (root / 'src' if (root / 'src').is_dir() else root)
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))

        try:
            module = importlib.import_module("dots_tts.runtime")
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'dots.tts'. Install it with `pip install dots.tts`."
            ) from exc

        runtime_cls = getattr(module, "DotsTtsRuntime", None)
        if runtime_cls is None:
            raise ImportError("dots_tts.runtime does not expose DotsTtsRuntime.")
        if self.config.code_path and not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('dots_tts.runtime was imported from outside the configured code_path')

        # DotsTtsRuntime.from_pretrained() takes no device argument; it always
        # selects CUDA when available, using the current CUDA device. Scope
        # that selection so cuda:N is honoured without changing the caller.
        with self._runtime_context():
            self._runtime = runtime_cls.from_pretrained(
                self.config.checkpoint,
                precision=self.config.precision,
                optimize=self.config.optimize,
            )
            self._runtime_matmul_precision = torch.get_float32_matmul_precision()
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

        with self._runtime_context():
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

    @contextlib.contextmanager
    def _runtime_context(self):
        device = torch.device(self.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError(f'dots.tts runtime does not support requested device {device}')
        if device.type == 'cpu' and torch.cuda.is_available():
            raise ValueError('dots.tts selects CUDA automatically on GPU hosts; an explicit CPU request '
                             'requires a CPU-only runtime')
        threads = torch.get_num_threads()
        precision = torch.get_float32_matmul_precision()
        context = torch.cuda.device(device) if device.type == 'cuda' else contextlib.nullcontext()
        try:
            with context:
                if device.type == 'cpu':
                    torch.set_num_threads(1)
                if self._runtime_matmul_precision is not None:
                    torch.set_float32_matmul_precision(self._runtime_matmul_precision)
                elif device.type == 'cuda' and self.config.precision in ('fp32', 'torch.float32', 'float32'):
                    torch.set_float32_matmul_precision('high')
                yield
        finally:
            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)
            if torch.get_float32_matmul_precision() != precision:
                torch.set_float32_matmul_precision(precision)

    def close(self):
        self._runtime = self.model = None
        self._runtime_matmul_precision = None
        self._model_ready = False
