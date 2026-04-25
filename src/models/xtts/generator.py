"""Coqui XTTS-v2 generator wrapper."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

from src.models.model import BaseModel


@dataclass
class XttsGeneratorConfig:
    checkpoint: str = "coqui/XTTS-v2"
    cache_dir: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    config_path: Optional[Path] = None
    vocab_path: Optional[Path] = None
    speaker_file_path: Optional[Path] = None
    use_deepspeed: bool = False
    strict_checkpoint: bool = True
    temperature: float = 0.75
    length_penalty: float = 1.0
    repetition_penalty: float = 10.0
    top_k: int = 50
    top_p: float = 0.85
    do_sample: bool = True
    num_beams: int = 1
    speed: float = 1.0
    enable_text_splitting: bool = False
    gpt_cond_len: int = 30
    gpt_cond_chunk_len: int = 6
    max_ref_len: int = 10
    sound_norm_refs: bool = False


class XttsGenerator(BaseModel):
    """Thin wrapper around Coqui XTTS-v2 inference."""

    def __init__(self, config: XttsGeneratorConfig, device, logger):
        super().__init__(
            model_name_or_path=str(config.checkpoint),
            device=device,
            logger=logger,
        )
        self.config = config
        self._runtime_config = None
        self.sample_rate: int = 24000
        self._validate_config()

    def generate(
        self,
        *,
        text: str,
        reference_audio: Path,
        language: str,
    ) -> Tuple[np.ndarray, int]:
        self.ensure_model()
        assert self.model is not None
        assert self._runtime_config is not None

        reference_path = Path(reference_audio)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_path}")

        target_text = (text or "").strip()
        if not target_text:
            raise ValueError("XTTS target text cannot be empty.")

        language_token = str(language or "en").strip()
        if not language_token:
            language_token = "en"

        outputs = self.model.synthesize(
            target_text,
            self._runtime_config,
            speaker_wav=str(reference_path),
            language=language_token,
            temperature=float(self.config.temperature),
            length_penalty=float(self.config.length_penalty),
            repetition_penalty=float(self.config.repetition_penalty),
            top_k=int(self.config.top_k),
            top_p=float(self.config.top_p),
            do_sample=bool(self.config.do_sample),
            num_beams=int(self.config.num_beams),
            speed=float(self.config.speed),
            enable_text_splitting=bool(self.config.enable_text_splitting),
            gpt_cond_len=int(self.config.gpt_cond_len),
            gpt_cond_chunk_len=int(self.config.gpt_cond_chunk_len),
            max_ref_len=int(self.config.max_ref_len),
            sound_norm_refs=bool(self.config.sound_norm_refs),
        )
        waveform = np.asarray(outputs["wav"], dtype=np.float32).reshape(-1)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        waveform = np.clip(waveform, -1.0, 1.0)
        return waveform, int(self.sample_rate)

    def _validate_config(self) -> None:
        checkpoint_path = self.config.checkpoint_path
        config_path = self.config.config_path
        vocab_path = self.config.vocab_path
        speaker_file_path = self.config.speaker_file_path
        cache_dir = self.config.cache_dir

        for candidate in (checkpoint_path, config_path, vocab_path, speaker_file_path, cache_dir):
            if candidate is None:
                continue
            if not isinstance(candidate, Path):
                raise TypeError(f"XTTS config paths must be pathlib.Path instances, got {type(candidate)!r}")

    def load_model(self) -> None:
        xtts_dir = self._resolve_checkpoint_dir()
        config_path = self.config.config_path or (xtts_dir / "config.json")
        if not config_path.exists():
            raise FileNotFoundError(f"XTTS config.json not found: {config_path}")

        self._ensure_transformers_compat()
        try:
            xtts_config_module = importlib.import_module("TTS.tts.configs.xtts_config")
            xtts_model_module = importlib.import_module("TTS.tts.models.xtts")
        except ImportError as exc:
            raise ImportError(
                "Unable to import Coqui XTTS runtime. "
                "Ensure the 'TTS' package is installed and compatible with the current transformers version."
            ) from exc

        XttsConfig = getattr(xtts_config_module, "XttsConfig", None)
        Xtts = getattr(xtts_model_module, "Xtts", None)
        if XttsConfig is None or Xtts is None:
            raise RuntimeError("Coqui TTS installation does not expose XTTS runtime classes.")

        runtime_config = XttsConfig()
        runtime_config.load_json(str(config_path))

        model = Xtts.init_from_config(runtime_config)
        with self._legacy_torch_load_defaults():
            model.load_checkpoint(
                runtime_config,
                checkpoint_dir=str(xtts_dir),
                checkpoint_path=str(self.config.checkpoint_path) if self.config.checkpoint_path else None,
                vocab_path=str(self.config.vocab_path) if self.config.vocab_path else None,
                eval=True,
                strict=bool(self.config.strict_checkpoint),
                use_deepspeed=bool(self.config.use_deepspeed),
                speaker_file_path=(
                    str(self.config.speaker_file_path)
                    if self.config.speaker_file_path is not None
                    else None
                ),
            )
        model.to(self.device)
        model.eval()

        self._runtime_config = runtime_config
        self.model = model

        audio_cfg = getattr(runtime_config, "audio", None)
        if audio_cfg is not None and getattr(audio_cfg, "output_sample_rate", None) is not None:
            self.sample_rate = int(audio_cfg.output_sample_rate)
        elif getattr(runtime_config, "output_sample_rate", None) is not None:
            self.sample_rate = int(runtime_config.output_sample_rate)
        else:
            self.sample_rate = 24000

    def _resolve_checkpoint_dir(self) -> Path:
        checkpoint_value = str(self.config.checkpoint).strip()
        if not checkpoint_value:
            raise ValueError("XTTS checkpoint cannot be empty.")

        local_candidate = Path(checkpoint_value).expanduser()
        if local_candidate.exists():
            return local_candidate.resolve()

        try:
            absolute_candidate = Path(checkpoint_value).resolve()
        except Exception:
            absolute_candidate = local_candidate
        if absolute_candidate.exists():
            return absolute_candidate

        cache_dir = self.config.cache_dir
        snapshot_kwargs = {"repo_id": checkpoint_value}
        if cache_dir is not None:
            snapshot_kwargs["cache_dir"] = str(cache_dir)

        if self.logger:
            self.logger.info("[XTTS] Downloading checkpoint '%s' via Hugging Face Hub.", checkpoint_value)
        downloaded_path = snapshot_download(**snapshot_kwargs)
        return Path(downloaded_path).resolve()

    def _ensure_transformers_compat(self) -> None:
        """Backfill symbols that Coqui XTTS expects from older transformers exports."""
        try:
            transformers_module = importlib.import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "transformers is required to run XTTS-v2."
            ) from exc

        if hasattr(transformers_module, "BeamSearchScorer"):
            return

        try:
            beam_search_module = importlib.import_module("transformers.generation.beam_search")
            beam_search_scorer = getattr(beam_search_module, "BeamSearchScorer")
        except Exception as exc:
            raise ImportError(
                "Current transformers installation does not expose BeamSearchScorer required by Coqui XTTS."
            ) from exc

        setattr(transformers_module, "BeamSearchScorer", beam_search_scorer)
        if self.logger:
            self.logger.info(
                "[XTTS] Added transformers.BeamSearchScorer compatibility alias for Coqui XTTS."
            )

    @contextmanager
    def _legacy_torch_load_defaults(self):
        """Restore pre-2.6 torch.load behaviour expected by Coqui XTTS checkpoints."""
        original_torch_load = torch.load

        def _compat_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = _compat_torch_load
        try:
            yield
        finally:
            torch.load = original_torch_load
