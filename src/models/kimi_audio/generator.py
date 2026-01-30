"""Kimi Audio generator wrapper."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class KimiAudioGeneratorConfig:
    """Configuration options needed to run the Kimi Audio generator."""

    code_path: Path
    model_path: str
    load_detokenizer: bool = True
    audio_temperature: float = 0.8
    audio_top_k: int = 10
    text_temperature: float = 0.0
    text_top_k: int = 5
    audio_repetition_penalty: float = 1.0
    audio_repetition_window_size: int = 64
    text_repetition_penalty: float = 1.0
    text_repetition_window_size: int = 16
    max_new_tokens: Optional[int] = -1
    sample_rate: int = 24000


class KimiAudioGenerator(BaseModel):
    """Thin wrapper around MoonshotAI's Kimi Audio multi-modal model."""

    def __init__(
        self,
        config: KimiAudioGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        materialised_config = replace(config, code_path=Path(config.code_path))
        super().__init__(
            model_name_or_path=str(materialised_config.model_path),
            device=device,
            logger=logger,
        )
        self.config = materialised_config

        self._kimi_cls = None
        self._model = None
        self._imports_ready = False
        self.last_generated_text: Optional[str] = None

        self._validate_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, messages: List[dict[str, Any]], sample_index: int = 0) -> Tuple[np.ndarray, int]:
        """Generate an utterance conditioned on a chat-style prompt."""

        del sample_index  # Kimi handles stochasticity internally via temperatures
        self.ensure_model()
        assert self._model is not None

        if self.device.type != "cuda":
            raise RuntimeError("Kimi Audio generation currently requires a CUDA-capable device.")

        cuda_index = 0 if self.device.index is None else int(self.device.index)
        torch.cuda.set_device(cuda_index)

        max_new_tokens = self.config.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = -1

        wav, text = self._model.generate(
            messages,
            output_type="both",
            audio_temperature=float(self.config.audio_temperature),
            audio_top_k=int(self.config.audio_top_k),
            text_temperature=float(self.config.text_temperature),
            text_top_k=int(self.config.text_top_k),
            audio_repetition_penalty=float(self.config.audio_repetition_penalty),
            audio_repetition_window_size=int(self.config.audio_repetition_window_size),
            text_repetition_penalty=float(self.config.text_repetition_penalty),
            text_repetition_window_size=int(self.config.text_repetition_window_size),
            max_new_tokens=int(max_new_tokens),
        )

        if wav is None:
            raise RuntimeError("Kimi Audio did not return waveform data. Ensure 'load_detokenizer' is enabled.")

        self.last_generated_text = text
        wav_np = wav.detach().cpu().view(-1).float().numpy()
        return wav_np, int(self.config.sample_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"Kimi Audio code path not found: {self.config.code_path}")

    def _ensure_imports(self) -> None:
        if self._imports_ready:
            return

        import sys

        code_path_str = str(self.config.code_path)
        if code_path_str not in sys.path:
            sys.path.insert(0, code_path_str)

        try:
            from kimia_infer.api.kimia import KimiAudio as _KimiAudio
            # Patch transformers safetensors loader to tolerate missing metadata from
            # certain community checkpoints.
            try:
                import transformers.modeling_utils as _modeling_utils  # type: ignore
            except Exception:  # pragma: no cover - best effort only
                _modeling_utils = None
            else:
                safe_open_ref = getattr(_modeling_utils, "safe_open", None)

                if safe_open_ref is not None:
                    def _wrap_safe_open(*args, **kwargs):
                        context = safe_open_ref(*args, **kwargs)

                        class _SafeContext:
                            def __enter__(self_inner):
                                handle = context.__enter__()


                                class _SafeHandle:
                                    def __init__(self_handle, inner_handle):
                                        self_handle._inner = inner_handle

                                    def metadata(self_handle):
                                        metadata = getattr(self_handle._inner, "metadata", lambda: None)() or {}
                                        if metadata.get("format") in (None, ""):
                                            metadata = dict(metadata)
                                            metadata["format"] = "pt"
                                        return metadata

                                    def __getattr__(self_handle, name):
                                        return getattr(self_handle._inner, name)

                                return _SafeHandle(handle)

                            def __exit__(self_inner, exc_type, exc, tb):
                                return context.__exit__(exc_type, exc, tb)

                            def __getattr__(self_inner, name):
                                return getattr(context, name)

                        return _SafeContext()

                    _modeling_utils.safe_open = _wrap_safe_open
        except ModuleNotFoundError as exc:  # pragma: no cover - happens when deps missing
            missing = exc.name or "kimia_infer"
            hint = (
                "The Kimi Audio repository ships key components via a git submodule. "
                "Run 'git submodule update --init --recursive' inside checkpoints/Kimi-Audio "
                "and install its Python requirements before re-running."
            )
            raise ImportError(
                f"Missing dependency '{missing}' required by Kimi Audio. {hint}"
            ) from exc
        except ImportError as exc:  # pragma: no cover - happens when deps missing
            raise ImportError(
                "Unable to import Kimi Audio inference module. Ensure its dependencies are installed."
            ) from exc

        self._kimi_cls = _KimiAudio
        self._imports_ready = True

    def _resolve_model_path(self) -> str:
        raw_path = self.config.model_path
        candidate = Path(raw_path)
        if candidate.exists():
            return str(candidate.resolve())

        nested_candidate = self.config.code_path / raw_path
        if nested_candidate.exists():
            return str(nested_candidate.resolve())

        return str(raw_path)

    def load_model(self) -> None:
        if self._model is not None:
            return

        if self.device.type != "cuda":
            raise RuntimeError("Kimi Audio currently supports CUDA inference only.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but is required for Kimi Audio.")

        cuda_index = 0 if self.device.index is None else int(self.device.index)
        torch.cuda.set_device(cuda_index)

        self._ensure_imports()
        assert self._kimi_cls is not None

        model_path = self._resolve_model_path()
        self.logger.info("[KimiAudio] Loading model from %s", model_path)

        self._model = self._kimi_cls(
            model_path=model_path,
            load_detokenizer=bool(self.config.load_detokenizer),
        )

        self.logger.info("[KimiAudio] Model ready on cuda:%d", cuda_index)
