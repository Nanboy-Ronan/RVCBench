"""MOSS-TTSD generator wrapper for zero-shot cloning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import whisper

from src.models.model import BaseModel


@dataclass
class MossTTSDGeneratorConfig:
    """Configuration required to run the MOSS-TTSD generator."""

    code_path: Path
    model_path: str
    spt_config_path: Path
    spt_checkpoint_path: Path
    system_prompt: str = (
        "You are a speech synthesizer that clones natural, realistic, and human-like audio from reference text."
    )
    torch_dtype: Union[str, torch.dtype] = torch.bfloat16
    attn_implementation: str = "flash_attention_2"
    use_normalize: bool = True
    silence_duration: float = 0.0
    seed: Optional[int] = None


class MossTTSDGenerator(BaseModel):
    """Thin wrapper around the upstream MOSS-TTSD inference utilities."""

    def __init__(
        self,
        config: MossTTSDGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.model_path),
            device=device,
            logger=logger,
        )
        self.config = config

        self._torch_dtype = self._resolve_dtype(config.torch_dtype)
        self._generation_utils = None
        self._tokenizer = None
        self._model = None
        self._spt = None

        self._validate_paths()
        self.whisper_model = whisper.load_model("base.en", device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio: Optional[Path],
        prompt_text: str,
        sample_index: int,
    ) -> Tuple[np.ndarray, int]:
        """Generate audio conditioned on dialogue text and a reference clip."""

        self.ensure_model()
        self._set_seed(sample_index)

        with torch.no_grad():
            result = self.whisper_model.transcribe(
                str(prompt_audio),
                language="en",
                fp16=self.whisper_model.device.type != "cpu",
            )
        original_text = result.get("text", "").strip()

        item = {"text": f"[S1]{text} "}
        item["prompt_audio_speaker1"] = str(prompt_audio)
        item["prompt_audio_speaker2"] = str(prompt_audio)
        item["prompt_text_speaker1"] = original_text
        item["prompt_text_speaker2"] = original_text

        process_batch = getattr(self._generation_utils, "process_batch")
        actual_texts, audio_results = process_batch(
            batch_items=[item],
            tokenizer=self._tokenizer,
            model=self._model,
            spt=self._spt,
            device=self.device,
            system_prompt=self.config.system_prompt,
            start_idx=0,
            use_normalize=bool(self.config.use_normalize),
            silence_duration=float(self.config.silence_duration),
        )

        self.logger.debug(
            "[MOSS-TTSD] Actual text info: %s",
            actual_texts[0] if actual_texts else "<missing>",
        )

        if not audio_results or audio_results[0] is None:
            raise RuntimeError("MOSS-TTSD failed to generate audio for the provided prompt.")

        audio_data = audio_results[0]["audio_data"]
        sample_rate = int(audio_results[0]["sample_rate"])

        if isinstance(audio_data, torch.Tensor):
            wav = audio_data.detach().cpu().numpy()
        else:
            wav = np.asarray(audio_data)

        wav = np.atleast_1d(wav).astype(np.float32)
        if wav.ndim > 1:
            wav = wav.reshape(-1)
        return wav, sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"MOSS-TTSD code path not found: {self.config.code_path}")
        if not self.config.spt_config_path.exists():
            raise FileNotFoundError(
                f"MOSS-TTSD tokenizer config path not found: {self.config.spt_config_path}"
            )
        if not self.config.spt_checkpoint_path.exists():
            raise FileNotFoundError(
                f"MOSS-TTSD tokenizer checkpoint not found: {self.config.spt_checkpoint_path}"
            )

    def _ensure_imports(self) -> None:
        if self._generation_utils is not None:
            return

        import sys

        code_path_str = str(self.config.code_path)
        if code_path_str not in sys.path:
            sys.path.insert(0, code_path_str)

        xy_path = self.config.code_path / "XY_Tokenizer"
        xy_path_str = str(xy_path)
        if xy_path.exists() and xy_path_str not in sys.path:
            sys.path.insert(0, xy_path_str)

        try:
            import generation_utils  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive
            raise ModuleNotFoundError(
                "Failed to import MOSS-TTSD generation utilities. Ensure 'code_path' points to the "
                "MOSS-TTSD repository."
            ) from exc

        self._generation_utils = generation_utils

    def load_model(self) -> None:
        if self._model is not None and self._spt is not None:
            return

        self._ensure_imports()

        load_model_fn = getattr(self._generation_utils, "load_model")
        tokenizer, model, spt = load_model_fn(
            self.config.model_path,
            str(self.config.spt_config_path),
            str(self.config.spt_checkpoint_path),
            torch_dtype=self._torch_dtype,
            attn_implementation=self.config.attn_implementation,
        )

        self._tokenizer = tokenizer
        self._model = model.to(self.device)
        self._spt = spt.to(self.device)
        self._model.eval()
        self._spt.eval()


    def _resolve_dtype(self, dtype: Union[str, torch.dtype]) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        lookup = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        key = str(dtype).lower()
        if key not in lookup:
            raise ValueError(f"Unsupported torch dtype for MOSS-TTSD: {dtype}")
        return lookup[key]

    def _set_seed(self, sample_index: int) -> None:
        if self.config.seed is None:
            return
        seed = int(self.config.seed) + int(sample_index)
        try:
            from accelerate.utils import set_seed

            set_seed(seed)
        except Exception:  # pragma: no cover - best effort fallback
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
