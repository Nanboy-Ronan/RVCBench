"""CosyVoice generator wrapper for zero-shot cloning."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from rvcbench.models.model import BaseModel
from .assets import COMMON_FILES, canonical_variant, checkpoint_files, model_directory


@dataclass
class CosyVoiceGeneratorConfig:
    """Configuration options needed to invoke CosyVoice locally."""

    code_path: Path
    model_dir: Path
    variant: str = "cosyvoice2"
    stream: bool = False
    speed: float = 1.0
    text_frontend: bool = False
    load_jit: bool = False
    load_trt: bool = False
    load_vllm: bool = False
    fp16: bool = False
    trt_concurrent: int = 1
    matcha_code_path: Optional[Path] = None


class CosyVoiceGenerator(BaseModel):
    """Thin wrapper around the CosyVoice/CosyVoice2 CLI helpers."""

    _COMMON_REQUIRED_FILES = COMMON_FILES

    def __init__(self, config: CosyVoiceGeneratorConfig, device, logger):
        materialised_config = replace(
            config,
            code_path=Path(config.code_path),
            model_dir=Path(config.model_dir),
        )
        super().__init__(
            model_name_or_path=str(materialised_config.model_dir),
            device=device,
            logger=logger,
        )
        self.config = replace(
            materialised_config,
            model_dir=self._resolve_model_dir(
                Path(materialised_config.model_dir),
                Path(materialised_config.code_path),
                str(materialised_config.variant or "cosyvoice2"),
            ),
        )

        self._model = None
        self.sample_rate: Optional[int] = None

        self._validate_paths()
        checkpoint_files(self.config.model_dir, self.config.variant)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        text: str,
        prompt_audio_16k: torch.Tensor,
        prompt_text: str,
        zero_shot_speaker_id: str = "",
    ) -> Tuple[np.ndarray, int]:
        self.ensure_model()
        assert self._model is not None

        text_value = (text or "").strip()
        prompt_text_value = (prompt_text or "").strip()

        inference_iterator = self._model.inference_zero_shot(
            tts_text=text_value,
            prompt_text=prompt_text_value,
            prompt_speech_16k=prompt_audio_16k,
            zero_shot_spk_id=zero_shot_speaker_id,
            stream=bool(self.config.stream),
            speed=float(self.config.speed),
            text_frontend=bool(self.config.text_frontend),
        )

        chunks: List[np.ndarray] = []
        for result in inference_iterator:
            candidate = result.get("tts_speech") if isinstance(result, dict) else None
            if not isinstance(candidate, torch.Tensor):
                raise ValueError("CosyVoice chunk must contain a tts_speech tensor")
            if candidate.ndim == 2 and candidate.shape[0] == 1:
                candidate = candidate.squeeze(0)
            if candidate.ndim != 1 or not candidate.numel() or not torch.isfinite(candidate).all():
                raise ValueError("CosyVoice chunk must be nonempty finite mono audio")
            chunk = candidate.detach().to(device="cpu", dtype=torch.float32).numpy()
            if not np.isfinite(chunk).all():
                raise ValueError("CosyVoice chunk exceeds finite float32 audio range")
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("CosyVoice returned no audio for the provided prompt.")

        waveform = np.concatenate(chunks, axis=-1)
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

        sample_rate = int(self.sample_rate) if self.sample_rate is not None else 24000
        return waveform, sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def close(self):
        self._model = self.model = None
        self._model_ready = False
        self.sample_rate = None

    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"CosyVoice code_path not found: {self.config.code_path}")
        if not self.config.model_dir.exists():
            raise FileNotFoundError(f"CosyVoice model_dir not found: {self.config.model_dir}")

    def _ensure_pythonpath(self) -> None:
        paths = [self.config.code_path]
        third_party = self.config.code_path / "third_party" / "Matcha-TTS"
        if third_party.exists():
            paths.append(third_party)

        for candidate in paths:
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)

        self._ensure_matcha_dependency()

    def _ensure_modelscope_stub(self) -> None:
        try:
            importlib.import_module("modelscope")
            return
        except ImportError:
            pass

        stub = types.ModuleType("modelscope")

        def _snapshot_download_stub(*_args, **_kwargs):
            raise RuntimeError(
                "modelscope.snapshot_download was requested but 'modelscope' is unavailable. "
                "Provide local CosyVoice assets or install modelscope."
            )

        stub.snapshot_download = _snapshot_download_stub  # type: ignore[attr-defined]
        sys.modules.setdefault("modelscope", stub)

    def load_model(self) -> None:
        checkpoint_files(self.config.model_dir, self.config.variant)
        if self._model is not None:
            return

        self._validate_transformers_version()
        self._ensure_pythonpath()
        self._ensure_modelscope_stub()

        try:
            cosyvoice_module = importlib.import_module("cosyvoice.cli.cosyvoice")
        except ImportError as exc:
            raise ImportError(
                "Unable to import CosyVoice CLI helpers. Ensure its dependencies are installed."
            ) from exc

        CosyVoiceClass = getattr(cosyvoice_module, "CosyVoice", None)
        CosyVoice2Class = getattr(cosyvoice_module, "CosyVoice2", None)
        if CosyVoiceClass is None or CosyVoice2Class is None:
            raise RuntimeError("CosyVoice CLI module does not expose expected classes.")

        variant = canonical_variant(self.config.variant)
        target_cls = CosyVoice2Class if variant == "cosyvoice2" else CosyVoiceClass
        options = dict(load_jit=bool(self.config.load_jit), load_trt=bool(self.config.load_trt),
                       fp16=bool(self.config.fp16), trt_concurrent=int(self.config.trt_concurrent))
        if variant == "cosyvoice2":
            options["load_vllm"] = bool(self.config.load_vllm)
        elif self.config.load_vllm:
            raise ValueError("load_vllm is supported only by CosyVoice2")
        self._model = target_cls(str(self.config.model_dir), **options)
        self.model = self._model
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000))

    def _validate_transformers_version(self) -> None:
        requirements = self.config.code_path / 'requirements.txt'
        if not requirements.is_file():
            return
        pinned = re.search(r'^\s*transformers==([^\s;#]+)', requirements.read_text(), re.MULTILINE)
        if pinned is None:
            return
        installed = importlib.metadata.version('transformers')
        if installed != pinned.group(1):
            raise RuntimeError(
                f'CosyVoice source requires transformers=={pinned.group(1)}; found {installed}. '
                'Use an isolated environment matching the upstream pin. '
                'Newer versions can generate invalid speech without raising inference errors.'
            )

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------
    def _resolve_model_dir(self, supplied: Path, code_path: Path, variant: str) -> Path:
        directory = model_directory(supplied if supplied else code_path, variant)
        if self.logger and directory != Path(supplied if supplied else code_path).expanduser().resolve():
            self.logger.info("Resolved CosyVoice model_dir at %s (requested %s)", directory, supplied)
        return directory

    # ------------------------------------------------------------------
    # Third-party dependency helpers
    # ------------------------------------------------------------------
    def _ensure_matcha_dependency(self) -> None:
        if self.config.matcha_code_path is not None:
            candidate = Path(self.config.matcha_code_path).expanduser().resolve()
            if not self._is_matcha_root(candidate):
                raise FileNotFoundError(f'Configured Matcha-TTS source is missing: {candidate}')
            sys.path.insert(0, str(candidate))
            module = importlib.import_module('matcha')
            origin = getattr(module, '__file__', None)
            if not origin or not Path(origin).resolve().is_relative_to(candidate):
                raise RuntimeError(f'Matcha-TTS already loaded from a different source: {origin}')
            if self.logger:
                self.logger.info('Loaded configured Matcha-TTS dependency from %s', candidate)
            return
        try:
            importlib.import_module("matcha")
            return
        except ImportError:
            pass

        search_roots = [
            self.config.code_path / "third_party",
            self.config.code_path.parent,
            Path("checkpoints"),
            Path.cwd(),
        ]

        visited = set()
        for root in search_roots:
            if root is None:
                continue
            try:
                resolved = root.expanduser().resolve()
            except Exception:
                continue
            if resolved in visited or not resolved.exists() or not resolved.is_dir():
                continue
            visited.add(resolved)

            candidate = self._locate_matcha_package(resolved)
            if candidate is None:
                continue

            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                importlib.import_module("matcha")
            except ImportError:
                continue
            if self.logger:
                self.logger.info("Loaded Matcha-TTS dependency from %s", candidate)
            return

        if self.logger:
            self.logger.warning(
                "Matcha-TTS package not found. Ensure the CosyVoice submodule is initialised or install matcha-tts."
            )

    def _locate_matcha_package(self, root: Path) -> Optional[Path]:
        direct = root / "Matcha-TTS"
        if self._is_matcha_root(direct):
            return direct

        try:
            for match in root.rglob("Matcha-TTS"):
                if self._is_matcha_root(match):
                    return match
        except Exception:
            pass
        return None

    def _is_matcha_root(self, directory: Path) -> bool:
        return directory.exists() and (directory / "matcha").is_dir()
