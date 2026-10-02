"""PlayDiffusion generator wrapper for off-the-shelf voice cloning."""
from __future__ import annotations

from copy import deepcopy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from rvcbench.models.model import BaseModel


@dataclass
class PlayDiffusionGeneratorConfig:
    code_path: Path
    preset_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    hf_repo_id: str = "PlayHT/inpainter"
    hf_revision: Optional[str] = None
    num_steps: int = 30
    init_temp: float = 1.0
    init_diversity: float = 1.0
    guidance: float = 0.5
    rescale: float = 0.7
    top_k: int = 25
    audio_token_syllable_ratio: Optional[float] = None
    vocoder_checkpoint: str = "v090_g_01105000"
    tokenizer_file: str = "tokenizer-multi_bpe16384_merged_extended_58M.json"
    speech_tokenizer_checkpoint: str = "xlsr2_1b_v2_custom.pt"
    kmeans_layer_checkpoint: str = "kmeans_10k.npy"
    voice_encoder_checkpoint: str = "voice_encoder_1992000.pt"
    inpainter_checkpoint: str = "last_250k_fixed.pkl"
    speech_tokenizer_sample_rate: int = 16000


class PlayDiffusionGenerator(BaseModel):
    """Thin wrapper around the PlayDiffusion inference stack."""

    def __init__(
        self,
        config: PlayDiffusionGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path="PlayDiffusion",
            device=device,
            logger=logger,
        )
        self.config = config
        self.config.code_path = Path(self.config.code_path).expanduser().resolve()
        if self.config.preset_dir is not None:
            self.config.preset_dir = Path(self.config.preset_dir).expanduser().resolve()
        if self.config.cache_dir is not None:
            self.config.cache_dir = Path(self.config.cache_dir).expanduser().resolve()

        self._engine = None
        self._tts_input_cls = None

        self._validate_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio: Path,
        prompt_text: Optional[str] = None,
        sample_index: int = 0,
    ) -> Tuple[int, np.ndarray]:
        del prompt_text, sample_index  # PlayDiffusion does not accept prompt transcripts directly

        if not text or not str(text).strip():
            raise ValueError("PlayDiffusion requires non-empty text input.")
        if prompt_audio is None or not Path(prompt_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {prompt_audio}")

        self.ensure_model()
        assert self._engine is not None
        assert self._tts_input_cls is not None

        tts_input = self._tts_input_cls(
            output_text=str(text).strip(),
            voice=str(prompt_audio),
            num_steps=int(self.config.num_steps),
            init_temp=float(self.config.init_temp),
            init_diversity=float(self.config.init_diversity),
            guidance=float(self.config.guidance),
            rescale=float(self.config.rescale),
            topk=int(self.config.top_k),
            audio_token_syllable_ratio=self.config.audio_token_syllable_ratio,
        )

        sample_rate, pcm = self._engine.tts(tts_input)
        return int(sample_rate), np.asarray(pcm)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"PlayDiffusion code path not found: {self.config.code_path}")
        if self.config.preset_dir is not None and not self.config.preset_dir.exists():
            raise FileNotFoundError(f"PlayDiffusion preset_dir not found: {self.config.preset_dir}")

    def _ensure_imports(self) -> None:
        code_path_str = str(self.config.code_path / "src")
        if code_path_str not in sys.path:
            sys.path.insert(0, code_path_str)

    def _build_local_preset(self) -> dict:
        assert self.config.preset_dir is not None

        def _resolve(filename: str) -> str:
            path = self.config.preset_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"PlayDiffusion preset file not found: {path}")
            return str(path)

        kmeans_path = _resolve(self.config.kmeans_layer_checkpoint)
        preset = {
            "vocoder": {
                "checkpoint": _resolve(self.config.vocoder_checkpoint),
                "kmeans_layer_checkpoint": kmeans_path,
            },
            "tokenizer": {
                "vocab_file": _resolve(self.config.tokenizer_file),
            },
            "speech_tokenizer": {
                "checkpoint": _resolve(self.config.speech_tokenizer_checkpoint),
                "kmeans_layer_checkpoint": kmeans_path,
                "sample_rate": int(self.config.speech_tokenizer_sample_rate),
            },
            "voice_encoder": {
                "checkpoint": _resolve(self.config.voice_encoder_checkpoint),
            },
            "inpainter": {
                "checkpoint": _resolve(self.config.inpainter_checkpoint),
            },
        }
        return preset

    def load_model(self) -> None:
        self._ensure_imports()
        if self.config.preset_dir is None:
            from huggingface_hub import snapshot_download
            self.config.preset_dir = Path(snapshot_download(
                repo_id=self.config.hf_repo_id, revision=self.config.hf_revision,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
                allow_patterns=[getattr(self.config, field) for field in PRESET_FILES]))
        preset = self._build_local_preset()

        from playdiffusion.inference import PlayDiffusion
        from playdiffusion.pydantic_models.models import TTSInput

        class ConfiguredPlayDiffusion(PlayDiffusion):
            def load_preset(self, *args, **kwargs):
                # The manager mutates the preset. Give its one initialization
                # a private copy and avoid constructing a default manager first.
                return deepcopy(preset)

        self._engine = ConfiguredPlayDiffusion(device=str(self.device))
        self._tts_input_cls = TTSInput

    def close(self):
        self._engine = self._tts_input_cls = None
        self._model_ready = False


PRESET_FILES = ('vocoder_checkpoint', 'tokenizer_file', 'speech_tokenizer_checkpoint',
                'kmeans_layer_checkpoint', 'voice_encoder_checkpoint', 'inpainter_checkpoint')
