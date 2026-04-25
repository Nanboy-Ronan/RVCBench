"""OpenVoice V2 generator wrapper."""

from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from src.models.model import BaseModel


_LANGUAGE_TO_MELO = {
    "en": "EN",
    "english": "EN",
    "en-us": "EN",
    "en-gb": "EN",
    "zh": "ZH",
    "zh-cn": "ZH",
    "zh-tw": "ZH",
    "chinese": "ZH",
    "es": "ES",
    "spanish": "ES",
    "fr": "FR",
    "french": "FR",
    "ja": "JP",
    "jp": "JP",
    "japanese": "JP",
    "ko": "KR",
    "kr": "KR",
    "korean": "KR",
}


@dataclass
class OpenVoiceGeneratorConfig:
    code_path: Path
    converter_config_path: Path
    converter_checkpoint_path: Path
    base_speaker_dir: Path
    melo_code_path: Optional[Path] = None
    device: str = "cuda:0"
    speed: float = 1.0
    tau: float = 0.3
    enable_watermark: bool = False
    melo_use_hf: bool = True
    default_melo_language: str = "EN"
    source_speaker_key: Optional[str] = None


class OpenVoiceGenerator(BaseModel):
    """Inference bridge for OpenVoice V2 + MeloTTS base speakers."""

    def __init__(
        self,
        config: OpenVoiceGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.converter_checkpoint_path),
            device=device,
            logger=logger,
        )
        self.config = config

        self._converter_cls = None
        self._tts_cls = None

        self._converter = None
        self._tts_models: Dict[str, object] = {}
        self._target_se_cache: Dict[Path, torch.Tensor] = {}
        self._sample_rate: Optional[int] = None

        self._validate_paths()

    def generate(
        self,
        *,
        text: str,
        reference_audio: Path,
        language: Optional[str],
        source_speaker_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        self.ensure_model()
        assert self._converter is not None

        reference_path = Path(reference_audio)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_path}")

        target_text = (text or "").strip()
        if not target_text:
            raise ValueError("OpenVoice target text cannot be empty.")

        melo_language = self._normalise_language(language)
        tts_model = self._get_tts_model(melo_language)
        speaker_key, speaker_id, source_se = self._resolve_source_speaker(
            tts_model,
            melo_language,
            explicit_key=source_speaker_key or self.config.source_speaker_key,
        )
        target_se = self._get_target_se(reference_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            src_path = Path(handle.name)
        try:
            tts_model.tts_to_file(
                target_text,
                speaker_id,
                str(src_path),
                speed=float(self.config.speed),
                quiet=True,
            )
            converted = self._converter.convert(
                audio_src_path=str(src_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=None,
                tau=float(self.config.tau),
                message="default",
            )
        finally:
            try:
                src_path.unlink(missing_ok=True)
            except Exception:
                pass

        wav = np.asarray(converted, dtype=np.float32).reshape(-1)
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        wav = np.clip(wav, -1.0, 1.0)
        if self.logger:
            self.logger.debug(
                "[OpenVoice] Generated using Melo language=%s base speaker=%s",
                melo_language,
                speaker_key,
            )
        return wav, self.sample_rate

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is not None:
            return int(self._sample_rate)
        if self._converter is not None:
            hps = getattr(self._converter, "hps", None)
            if hps is not None and getattr(hps, "data", None) is not None:
                self._sample_rate = int(hps.data.sampling_rate)
                return int(self._sample_rate)
        self._sample_rate = 24000
        return 24000

    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"OpenVoice code path not found: {self.config.code_path}")
        if self.config.melo_code_path is not None and not self.config.melo_code_path.exists():
            raise FileNotFoundError(f"MeloTTS code path not found: {self.config.melo_code_path}")
        if not self.config.converter_config_path.exists():
            raise FileNotFoundError(
                f"OpenVoice converter config not found: {self.config.converter_config_path}"
            )
        if not self.config.converter_checkpoint_path.exists():
            raise FileNotFoundError(
                f"OpenVoice converter checkpoint not found: {self.config.converter_checkpoint_path}"
            )
        if not self.config.base_speaker_dir.exists():
            raise FileNotFoundError(
                f"OpenVoice base speaker directory not found: {self.config.base_speaker_dir}"
            )

    def load_model(self) -> None:
        if self._converter is not None:
            return

        self._ensure_imports()
        assert self._converter_cls is not None

        converter = self._converter_cls(
            str(self.config.converter_config_path),
            device=str(self.device),
            enable_watermark=bool(self.config.enable_watermark),
        )
        converter.load_ckpt(str(self.config.converter_checkpoint_path))
        self._converter = converter
        self.model = converter
        self._sample_rate = self.sample_rate

    def _ensure_imports(self) -> None:
        candidate_paths = [self.config.code_path]
        if self.config.melo_code_path is not None:
            candidate_paths.append(self.config.melo_code_path)

        ordered_paths = [str(path) for path in candidate_paths if path is not None]
        for path_str in reversed(ordered_paths):
            if path_str in sys.path:
                sys.path.remove(path_str)
            sys.path.insert(0, path_str)

        for prefix in ("openvoice", "melo"):
            for module_name in list(sys.modules):
                if module_name == prefix or module_name.startswith(f"{prefix}."):
                    sys.modules.pop(module_name, None)

        try:
            openvoice_module = importlib.import_module("openvoice.api")
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            raise ImportError(
                "Unable to import OpenVoice from "
                f"{self.config.code_path}. Underlying error: {details}"
            ) from exc

        try:
            melo_module = importlib.import_module("melo.api")
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            raise ImportError(
                "Unable to import MeloTTS from "
                f"{self.config.melo_code_path}. Underlying error: {details}"
            ) from exc

        base_cls = getattr(openvoice_module, "OpenVoiceBaseClass", None)
        if base_cls is not None:
            init_signature = inspect.signature(base_cls.__init__)
            has_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in init_signature.parameters.values()
            )
            if not has_var_kwargs:
                original_init = base_cls.__init__

                def _patched_init(instance, config_path, device="cuda:0", **_ignored_kwargs):
                    return original_init(instance, config_path, device=device)

                base_cls.__init__ = _patched_init

        self._converter_cls = getattr(openvoice_module, "ToneColorConverter", None)
        self._tts_cls = getattr(melo_module, "TTS", None)
        if self._converter_cls is None:
            raise RuntimeError("openvoice.api does not expose ToneColorConverter.")
        if self._tts_cls is None:
            raise RuntimeError("melo.api does not expose TTS.")

    def _normalise_language(self, value: Optional[str]) -> str:
        token = str(value or "").strip()
        if not token:
            return str(self.config.default_melo_language)
        mapped = _LANGUAGE_TO_MELO.get(token.lower())
        if mapped is not None:
            return mapped
        return token.upper()

    def _get_tts_model(self, language: str):
        if language in self._tts_models:
            return self._tts_models[language]
        self.ensure_model()
        assert self._tts_cls is not None
        model = self._tts_cls(
            language=language,
            device=str(self.device),
            use_hf=bool(self.config.melo_use_hf),
        )
        self._tts_models[language] = model
        return model

    def _get_target_se(self, reference_path: Path) -> torch.Tensor:
        resolved = reference_path.resolve()
        cached = self._target_se_cache.get(resolved)
        if cached is not None:
            return cached
        assert self._converter is not None
        target_se = self._converter.extract_se([str(resolved)])
        self._target_se_cache[resolved] = target_se
        return target_se

    def _resolve_source_speaker(self, tts_model, language: str, explicit_key: Optional[str]):
        speaker_map = getattr(tts_model.hps.data, "spk2id", None)
        if speaker_map is None and hasattr(tts_model.hps.data, "keys"):
            try:
                speaker_map = tts_model.hps.data["spk2id"]
            except Exception:
                speaker_map = None
        if speaker_map is not None and not isinstance(speaker_map, dict):
            try:
                speaker_map = dict(speaker_map)
            except Exception:
                speaker_map = None
        if not speaker_map:
            raise RuntimeError(f"MeloTTS speaker map missing for language {language}.")

        if explicit_key:
            if explicit_key not in speaker_map:
                available = ", ".join(sorted(speaker_map.keys()))
                raise ValueError(
                    f"Configured OpenVoice source speaker '{explicit_key}' is unavailable for {language}. "
                    f"Available speakers: {available}"
                )
            speaker_key = explicit_key
        else:
            speaker_key = sorted(speaker_map.keys())[0]

        speaker_id = speaker_map[speaker_key]
        source_se = self._load_source_embedding(speaker_key)
        return speaker_key, speaker_id, source_se

    def _load_source_embedding(self, speaker_key: str) -> torch.Tensor:
        candidates = [
            self.config.base_speaker_dir / f"{speaker_key}.pth",
            self.config.base_speaker_dir / f"{speaker_key.lower()}.pth",
            self.config.base_speaker_dir / f"{speaker_key.lower().replace('_', '-')}.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return torch.load(candidate, map_location=str(self.device))
        attempted = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"OpenVoice base speaker embedding not found for speaker '{speaker_key}'. Tried: {attempted}"
        )
