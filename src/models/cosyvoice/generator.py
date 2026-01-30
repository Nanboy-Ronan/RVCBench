"""CosyVoice generator wrapper for zero-shot cloning."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


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


class CosyVoiceGenerator(BaseModel):
    """Thin wrapper around the CosyVoice/CosyVoice2 CLI helpers."""

    _COMMON_REQUIRED_FILES = (
        "campplus.onnx",
        "flow.pt",
        "hift.pt",
        "llm.pt",
    )
    _CONFIG_FILENAMES = ("cosyvoice2.yaml", "cosyvoice.yaml", "configuration.json")

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
            if candidate is None:
                continue
            chunk = candidate.squeeze(0).detach().cpu().numpy().astype(np.float32)
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("CosyVoice returned no audio for the provided prompt.")

        waveform = np.concatenate(chunks, axis=-1)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

        sample_rate = int(self.sample_rate) if self.sample_rate is not None else 24000
        return waveform, sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
        if self._model is not None:
            return

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

        variant = (self.config.variant or "cosyvoice2").lower().strip()
        if variant in {"cosyvoice2", "cozyvoice2", "cosy2"}:
            target_cls = CosyVoice2Class
        elif variant in {"cosyvoice", "cozyvoice", "cosy1"}:
            target_cls = CosyVoiceClass
        else:
            raise ValueError(
                f"Unknown CosyVoice variant '{self.config.variant}'. Expected 'cosyvoice' or 'cosyvoice2'."
            )

        self._model = target_cls(
            str(self.config.model_dir),
            load_jit=bool(self.config.load_jit),
            load_trt=bool(self.config.load_trt),
            load_vllm=bool(self.config.load_vllm),
            fp16=bool(self.config.fp16),
            trt_concurrent=int(self.config.trt_concurrent),
        )
        self.model = self._model
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000))

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------
    def _resolve_model_dir(self, supplied: Path, code_path: Path, variant: str) -> Path:
        supplied = supplied.expanduser().resolve() if supplied else supplied

        search_roots = [
            supplied if supplied else None,
            supplied.parent if supplied else None,
            code_path,
            code_path.parent,
            Path("checkpoints"),
            Path.cwd(),
        ]

        unique_roots = []
        for root in search_roots:
            if root is None:
                continue
            try:
                resolved = Path(root).expanduser().resolve()
            except Exception:
                continue
            if resolved not in unique_roots and resolved.exists() and resolved.is_dir():
                unique_roots.append(resolved)

        target_names = {supplied.name} if supplied else set()
        if not target_names:
            target_names = {"CosyVoice2-0.5B", "cosyvoice2", "CosyVoice2"}
        lower_names = {name.lower() for name in list(target_names)}
        if "cosyvoice2-0.5b" in lower_names or "cosyvoice2" in lower_names:
            target_names.update({"cosyvoice2", "CosyVoice2-0.5B"})
        if "cosyvoice" in lower_names:
            target_names.add("cosyvoice")

        candidates = []
        for root in unique_roots:
            candidate = self._search_for_model_payload(root, target_names, variant)
            if candidate is None:
                continue
            # Only auto-accept when the user did not supply a path or when the
            # payload lives inside the supplied directory.
            if supplied:
                try:
                    if candidate.samefile(supplied) or supplied in candidate.parents:
                        if self.logger and candidate != supplied:
                            self.logger.info(
                                "Resolved CosyVoice model_dir at %s (requested %s)",
                                candidate,
                                supplied,
                            )
                        return candidate
                except Exception:
                    pass
                candidates.append(candidate)
            else:
                if self.logger:
                    self.logger.info(
                        "Resolved CosyVoice model_dir at %s (requested %s)",
                        candidate,
                        supplied,
                    )
                return candidate

        if supplied:
            hint = f" Closest match found: {candidates[0]}" if candidates else ""
            raise FileNotFoundError(
                f"CosyVoice model_dir not found or incomplete at {supplied}.{hint} "
                f"Provide a directory that contains {', '.join(self._COMMON_REQUIRED_FILES)}."
            )

        missing_path = supplied if supplied else Path("<unspecified>")
        raise FileNotFoundError(
            f"CosyVoice model_dir not found: {missing_path}. Provide a directory that contains "
            + ", ".join(self._COMMON_REQUIRED_FILES)
        )

    def _search_for_model_payload(self, root: Path, target_names, variant: str) -> Optional[Path]:
        direct_candidates = []
        for name in target_names:
            direct_candidates.extend(
                [
                    root / name,
                    root / "pretrained_models" / name,
                    root / "pretrained_models" / "pretrained_models" / name,
                ]
            )

        for candidate in direct_candidates:
            if candidate.exists() and candidate.is_dir() and self._has_model_payload(candidate, variant):
                return candidate

        try:
            for match in root.rglob("cosyvoice2.yaml"):
                candidate = match.parent
                if self._has_model_payload(candidate, variant):
                    return candidate
        except Exception:
            pass

        try:
            for match in root.rglob("cosyvoice.yaml"):
                candidate = match.parent
                if self._has_model_payload(candidate, variant):
                    return candidate
        except Exception:
            pass
        return None

    def _has_model_payload(self, directory: Path, variant: str) -> bool:
        if not directory.exists() or not directory.is_dir():
            return False
        common = all((directory / filename).exists() for filename in self._COMMON_REQUIRED_FILES)
        has_config = any((directory / filename).exists() for filename in self._CONFIG_FILENAMES)
        variant_norm = (variant or "cosyvoice2").lower().strip()
        if variant_norm in {"cosyvoice", "cozyvoice", "cosy1"}:
            required_tokenizers = ("speech_tokenizer_v1.onnx",)
        else:
            required_tokenizers = ("speech_tokenizer_v2.onnx",)
        has_tokenizer = all((directory / filename).exists() for filename in required_tokenizers)
        return common and has_config and has_tokenizer

    # ------------------------------------------------------------------
    # Third-party dependency helpers
    # ------------------------------------------------------------------
    def _ensure_matcha_dependency(self) -> None:
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
