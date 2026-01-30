"""Higgs Audio generator wrapper for off-the-shelf evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class HiggsAudioGeneratorConfig:
    code_path: Path
    model_path: str
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"
    scene_prompt_path: Optional[str] = None
    scene_prompt_text: Optional[str] = None
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    ras_win_len: int = 7
    ras_win_max_num_repeat: int = 2
    max_new_tokens: int = 2048
    chunk_method: Optional[str] = None
    chunk_max_word_num: int = 200
    chunk_max_num_turns: int = 1
    generation_chunk_buffer_size: Optional[int] = None
    seed: Optional[int] = None
    reference_prompt_text: str = "Here is a sample of the desired voice."
    use_static_kv_cache: bool = False
    # Controls how reference audio is injected into messages
    reference_role: str = "assistant"  # "assistant" (default in current wrapper) or "user" (official example style)
    reference_audio_first: bool = False  # if True, place ref audio before text in the conversation


class HiggsAudioGenerator(BaseModel):
    """Encapsulates Higgs Audio example generation workflow."""

    def __init__(
        self,
        config: HiggsAudioGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.model_path),
            device=device,
            logger=logger,
        )
        self.config = config

        self._imports_loaded = False
        self._generation_mod = None
        self._Message = None
        self._AudioContent = None
        self._prepare_chunk_text = None
        self._load_audio_tokenizer = None
        self._HiggsAudioModelClient = None
        self._scene_prompt_text: Optional[str] = None

        self._audio_tokenizer = None
        self._model_client = None

        self._device_str: Optional[str] = None
        self._device_id: Optional[int] = None
        self._tokenizer_device: Optional[str] = None

        self._validate_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        speaker_id: str,
        reference_audio: Path,
        sample_index: int,
    ) -> Tuple[np.ndarray, int]:
        """Generate audio conditioned on a reference clip and text prompt."""
        self.ensure_model()
        assert self._prepare_chunk_text is not None

        messages, audio_ids = self._build_reference_prompt(speaker_id, reference_audio)
        chunked_text = self._prepare_chunk_text(
            text,
            chunk_method=self.config.chunk_method,
            chunk_max_word_num=self.config.chunk_max_word_num,
            chunk_max_num_turns=self.config.chunk_max_num_turns,
        )

        # Diagnostic logging: examine how text was chunked (helps with Chinese truncation issues)
        logger = getattr(self, "logger", None)
        if logger is not None:
            try:
                logger.debug("[HiggsAudio] chunked_text=%s", chunked_text)
            except Exception:
                pass
        try:
            concat_wv, sr, _ = self._model_client.generate(
                messages=messages, # ([Message(role='system', content='Generate audio following instruction.', recipient=None), Message(role='user', content=AudioContent(audio_url='/tealab-data/rjin02/AudioWatermarkBench/results/em_on_libritts/20251216-124950/protected_audio/2300/2300_131720_000016_000008.wav', raw_audio=None, offset=None, duration=None, row_id=None, type='audio'), recipient=None)],)
                audio_ids=audio_ids,
                chunked_text=chunked_text, # (["'Our first engine compelled the inventing and making of a suitable engine indicator to indicate it-the Tabor."],)
                generation_chunk_buffer_size=self.config.generation_chunk_buffer_size,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                ras_win_len=self.config.ras_win_len,
                ras_win_max_num_repeat=self.config.ras_win_max_num_repeat,
                seed=(self.config.seed if self.config.seed is not None else 42),
            )
        except Exception as e:
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.error(
                    "[HiggsAudio] Generation failed: %s | speaker_id=%s | reference_audio=%s | text=%s | chunked_text=%s",
                    e,
                    speaker_id,
                    reference_audio,
                    text,
                    chunked_text,
                )
            raise
    
        wav = concat_wv.cpu().numpy() if hasattr(concat_wv, "cpu") else concat_wv
        return np.asarray(wav, dtype=np.float32), int(sr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"Higgs Audio code path not found: {self.config.code_path}")
        prompt_path = self.config.scene_prompt_path
        # Only enforce file existence when a path is provided and no inline text is supplied
        if prompt_path is not None and (self.config.scene_prompt_text is None):
            candidate = prompt_path.strip()
            if candidate.lower() == "empty":
                return
            path_obj = Path(candidate)
            if not path_obj.is_absolute():
                path_obj = (self.config.code_path / path_obj).resolve()
            if not path_obj.exists():
                raise FileNotFoundError(f"Scene prompt file not found: {path_obj}")

    def load_model(self) -> None:
        if self._model_client is not None and self._audio_tokenizer is not None:
            return

        self._ensure_imports()
        self._compute_device_info()

        tokenizer_device = self._tokenizer_device
        assert tokenizer_device is not None
        assert self._load_audio_tokenizer is not None
        if self._audio_tokenizer is None:
            self._audio_tokenizer = self._load_audio_tokenizer(
                self.config.audio_tokenizer_path,
                device=tokenizer_device,
            )

        if self._model_client is None:
            assert self._HiggsAudioModelClient is not None
            use_static_cache = self.config.use_static_kv_cache
            if self._device_str == "mps" and use_static_cache:
                use_static_cache = False
            self._model_client = self._HiggsAudioModelClient(
                model_path=self.config.model_path,
                audio_tokenizer=self._audio_tokenizer,
                device=self._device_str,
                device_id=self._device_id,
                max_new_tokens=self.config.max_new_tokens,
                use_static_kv_cache=use_static_cache,
            )

    def _ensure_imports(self) -> None:
        if self._imports_loaded:
            return

        import os
        import sys

        numba_cache_dir = self.config.code_path / ".numba_cache"
        try:
            numba_cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            from tempfile import gettempdir

            fallback_dir = Path(gettempdir()) / "higgs_audio_numba_cache"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            numba_cache_dir = fallback_dir

        os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
        os.environ.setdefault("LIBROSA_CACHE_DIR", str(numba_cache_dir))
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
        os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
        os.environ["USE_FLASH_ATTENTION"] = "0"

        code_path = str(self.config.code_path)
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        try:
            import transformers.utils.import_utils as import_utils

            def _no_flash_attn():
                return False

            import_utils.is_flash_attn_2_available = _no_flash_attn  # type: ignore[assignment]
        except Exception:
            pass

        try:
            generation_mod = __import__("examples.generation", fromlist=["*"])
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Unable to import 'examples.generation' from the Higgs Audio repository."
            ) from exc

        self._generation_mod = generation_mod
        self._Message = generation_mod.Message
        self._AudioContent = generation_mod.AudioContent
        self._prepare_chunk_text = generation_mod.prepare_chunk_text
        self._load_audio_tokenizer = generation_mod.load_higgs_audio_tokenizer
        self._HiggsAudioModelClient = generation_mod.HiggsAudioModelClient

        if self.config.scene_prompt_text is not None:
            self._scene_prompt_text = str(self.config.scene_prompt_text).strip()
        elif self.config.scene_prompt_path is not None:
            candidate = self.config.scene_prompt_path.strip()
            if candidate.lower() == "empty":
                self._scene_prompt_text = None
            else:
                prompt_path = Path(candidate)
                if not prompt_path.is_absolute():
                    prompt_path = (self.config.code_path / prompt_path).resolve()
                if not prompt_path.exists():
                    raise FileNotFoundError(f"Scene prompt file not found: {prompt_path}")
                self._scene_prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        else:
            self._scene_prompt_text = None

        self._imports_loaded = True

    def _compute_device_info(self) -> None:
        if self._device_str is not None:
            return

        if self.device.type == "cuda":
            index = 0 if self.device.index is None else self.device.index
            self._device_str = f"cuda:{index}"
            self._device_id = index
        elif self.device.type == "mps":
            self._device_str = "mps"
            self._device_id = None
        else:
            self._device_str = "cpu"
            self._device_id = None

        self._tokenizer_device = "cpu" if self._device_str == "mps" else self._device_str

    def _build_reference_prompt(self, speaker_id: str, ref_audio_path: Path) -> Tuple[List, List]:
        assert self._Message is not None
        assert self._AudioContent is not None

        prompt_template = self.config.reference_prompt_text
        if "{" in prompt_template:
            prompt_text = prompt_template.format(
                speaker=speaker_id,
                filename=ref_audio_path.stem,
            )
        else:
            prompt_text = prompt_template

        system_lines = ["Generate audio following instruction."]
        if prompt_text and str(prompt_text).strip():
            system_lines.append(str(prompt_text).strip())
        if self._scene_prompt_text:
            # Avoid double-wrapping if tags are already present in provided text
            if "<|scene_desc_start|>" in self._scene_prompt_text:
                system_lines.append(self._scene_prompt_text)
            else:
                system_lines.append(f"<|scene_desc_start|>\n{self._scene_prompt_text}\n<|scene_desc_end|>")
            # Log a short preview to confirm system prompt is active
            logger = getattr(self, "logger", None)
            if logger is not None:
                preview = self._scene_prompt_text.replace("\n", " ")[:120]
                try:
                    logger.info(
                        "[HiggsAudio] Scene prompt enabled (%d chars): %s...",
                        len(self._scene_prompt_text),
                        preview,
                    )
                except Exception:
                    pass
        system_message = self._Message(
            role="system",
            content="\n\n".join(system_lines),
        )

        assert self._audio_tokenizer is not None
        audio_tokens = self._audio_tokenizer.encode(str(ref_audio_path))
        messages = [system_message]

        # Build reference audio message with configurable role
        ref_role = str(getattr(self.config, "reference_role", "assistant")).strip().lower()
        if ref_role not in {"assistant", "user"}:
            ref_role = "assistant"
        ref_msg = self._Message(
            role=ref_role,
            content=self._AudioContent(audio_url=str(ref_audio_path)),
        )

        # Order reference audio vs. prompt text
        ref_first = bool(getattr(self.config, "reference_audio_first", False))
        if ref_first:
            messages.append(ref_msg)
        else:
            messages.append(ref_msg)
        return messages, [audio_tokens]
