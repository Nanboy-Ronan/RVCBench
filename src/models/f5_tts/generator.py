"""F5-TTS generator wrapper."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio

from src.models.model import BaseModel


@dataclass
class F5TTSGeneratorConfig:
    """Configuration options for F5-TTS voice cloning."""

    model: str = "F5TTS_v1_Base"
    ckpt_file: str = ""
    vocab_file: str = ""
    ode_method: str = "euler"
    use_ema: bool = True
    vocoder_local_path: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    infer_kwargs: Dict[str, Any] = field(default_factory=dict)


class F5TTSGenerator(BaseModel):
    """Thin wrapper around the upstream f5-tts Python API."""

    def __init__(self, config: F5TTSGeneratorConfig, device: torch.device, logger) -> None:
        materialised_config = replace(config)
        super().__init__(
            model_name_or_path=str(materialised_config.ckpt_file or materialised_config.model),
            device=device,
            logger=logger,
        )
        self.config = materialised_config
        self._api_cls = None
        self._api = None
        self._utils_module = None

    def load_model(self) -> None:
        if self._api is not None:
            return

        try:
            module = importlib.import_module("f5_tts.api")
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'f5-tts'. Install it with `pip install -U f5-tts`."
            ) from exc

        self._api_cls = getattr(module, "F5TTS", None)
        if self._api_cls is None:
            raise ImportError("f5_tts.api does not expose F5TTS.")

        init_kwargs: Dict[str, Any] = {
            "model": str(self.config.model),
            "ode_method": str(self.config.ode_method),
            "use_ema": bool(self.config.use_ema),
            "device": str(self.device),
        }
        if self.config.ckpt_file:
            init_kwargs["ckpt_file"] = self._resolve_optional_path(self.config.ckpt_file)
        if self.config.vocab_file:
            init_kwargs["vocab_file"] = self._resolve_optional_path(self.config.vocab_file)
        if self.config.vocoder_local_path:
            init_kwargs["vocoder_local_path"] = self._resolve_optional_path(self.config.vocoder_local_path)
        if self.config.hf_cache_dir:
            init_kwargs["hf_cache_dir"] = self._resolve_optional_path(self.config.hf_cache_dir)

        self._api = self._api_cls(**init_kwargs)
        self.model = getattr(self._api, "ema_model", None)
        self._utils_module = importlib.import_module("f5_tts.infer.utils_infer")

    def transcribe(self, ref_audio: str, language: Optional[str] = None) -> str:
        self.ensure_model()
        assert self._api is not None
        return str(self._api.transcribe(ref_audio, language=language)).strip()

    def generate(
        self,
        *,
        ref_audio: str,
        ref_text: str,
        gen_text: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        self.ensure_model()
        assert self._api is not None
        assert self._utils_module is not None

        if seed is not None:
            try:
                from f5_tts.model.utils import seed_everything

                seed_everything(int(seed))
            except Exception:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

        wav, sample_rate = self._infer_sequential(
            ref_audio=str(ref_audio),
            ref_text=str(ref_text),
            gen_text=str(gen_text),
        )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        return audio, int(sample_rate)

    def _resolve_optional_path(self, value: str) -> str:
        raw = str(value).strip()
        if not raw:
            return raw
        path = Path(raw).expanduser()
        if path.exists():
            return str(path.resolve())
        return raw

    def _infer_sequential(self, *, ref_audio: str, ref_text: str, gen_text: str) -> Tuple[np.ndarray, int]:
        assert self._api is not None
        utils = self._utils_module

        preprocess_ref_audio_text = getattr(utils, "preprocess_ref_audio_text")
        chunk_text = getattr(utils, "chunk_text")
        infer_batch_process = getattr(utils, "infer_batch_process")
        target_sample_rate = int(getattr(utils, "target_sample_rate"))

        ref_file, processed_ref_text = preprocess_ref_audio_text(
            ref_audio,
            ref_text,
            show_info=lambda *_args, **_kwargs: None,
        )

        audio, sample_rate = torchaudio.load(ref_file)
        max_chars = int(
            len(processed_ref_text.encode("utf-8"))
            / (audio.shape[-1] / sample_rate)
            * (22 - audio.shape[-1] / sample_rate)
            * float(self.config.infer_kwargs.get("speed", 1.0))
        )
        max_chars = max(1, max_chars)
        gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
        if not gen_text_batches:
            raise RuntimeError("F5-TTS produced no text chunks for generation.")

        generated_waves = []
        cross_fade_duration = float(self.config.infer_kwargs.get("cross_fade_duration", 0.15))

        for gen_text_batch in gen_text_batches:
            result_iter = infer_batch_process(
                (audio, sample_rate),
                processed_ref_text,
                [gen_text_batch],
                self._api.ema_model,
                self._api.vocoder,
                mel_spec_type=self._api.mel_spec_type,
                progress=None,
                target_rms=float(self.config.infer_kwargs.get("target_rms", 0.1)),
                cross_fade_duration=cross_fade_duration,
                nfe_step=int(self.config.infer_kwargs.get("nfe_step", 32)),
                cfg_strength=float(self.config.infer_kwargs.get("cfg_strength", 2.0)),
                sway_sampling_coef=float(self.config.infer_kwargs.get("sway_sampling_coef", -1.0)),
                speed=float(self.config.infer_kwargs.get("speed", 1.0)),
                fix_duration=self.config.infer_kwargs.get("fix_duration"),
                device=str(self.device),
            )
            generated_wave, _generated_sr, _generated_spec = next(result_iter)
            if generated_wave is None:
                continue
            generated_waves.append(np.asarray(generated_wave, dtype=np.float32))

        if not generated_waves:
            raise RuntimeError("F5-TTS returned no audio for the provided prompt.")

        combined = generated_waves[0]
        for next_wave in generated_waves[1:]:
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(combined), len(next_wave))
            if cross_fade_samples <= 0:
                combined = np.concatenate([combined, next_wave])
                continue

            prev_overlap = combined[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]
            fade_out = np.linspace(1.0, 0.0, cross_fade_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, cross_fade_samples, dtype=np.float32)
            cross_faded = prev_overlap * fade_out + next_overlap * fade_in
            combined = np.concatenate(
                [
                    combined[:-cross_fade_samples],
                    cross_faded,
                    next_wave[cross_fade_samples:],
                ]
            )

        return combined, target_sample_rate
