"""Fish-Speech off-the-shelf adversary integration."""
from __future__ import annotations

import numbers
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary


def _install_rms_norm_if_missing() -> None:
    """Backfill torch.nn.RMSNorm for runtimes < 2.5."""
    if hasattr(torch.nn, "RMSNorm"):
        return

    class _CompatRMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps: float = 1e-6, *, elementwise_affine: bool = True):
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = float(eps)
            if not elementwise_affine:
                raise NotImplementedError("Fish-Speech requires affine RMSNorm support.")
            self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            dims = tuple(range(-len(self.normalized_shape), 0)) or (-1,)
            variance = input.pow(2).mean(dim=dims, keepdim=True)
            normalized = input * torch.rsqrt(variance + self.eps)
            return normalized * self.weight

    torch.nn.RMSNorm = _CompatRMSNorm


_install_rms_norm_if_missing()

class FishSpeechZeroShotAdversary(BaseAdversary):
    """Runs Fish-Speech's local inference engine for zero-shot cloning."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        # Resolve key filesystem locations eagerly so we can fail fast.
        self.code_path = self._resolve_path(self.config.get("code_path", "checkpoints/fish_speech"))
        self.llama_checkpoint_path = self._resolve_path(self.config["llama_checkpoint_path"])
        self.decoder_checkpoint_path = self._resolve_path(self.config["decoder_checkpoint_path"])
        self.decoder_config_name = str(self.config.get("decoder_config_name", "modded_dac_vq"))

        self.mode = str(self.config.get("mode", "tts"))
        self.half = bool(self.config.get("half", False))
        self.compile = bool(self.config.get("compile", False))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 1024))
        self.chunk_length = int(self.config.get("chunk_length", 200))
        self.top_p = float(self.config.get("top_p", 0.8))
        self.repetition_penalty = float(self.config.get("repetition_penalty", 1.1))
        self.temperature = float(self.config.get("temperature", 0.8))
        self.normalize = bool(self.config.get("normalize", True))
        self.use_memory_cache = str(self.config.get("use_memory_cache", "off"))
        self.format = str(self.config.get("format", "wav"))
        self.seed = self.config.get("seed")
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")
        self.default_reference_text = str(
            self.config.get(
                "default_reference_text",
                "Here is a sample exhibiting the desired voice characteristics.",
            )
        )

        if self.chunk_length < 100 or self.chunk_length > 300:
            raise ValueError("Fish-Speech requires chunk_length between 100 and 300.")
        if self.use_memory_cache not in {"on", "off"}:
            raise ValueError("use_memory_cache must be 'on' or 'off'.")

        self._ensure_code_path()
        self._validate_resources()
        self._model_manager = None
        self._tts_engine = None
        self._ServeTTSRequest = None
        self._ServeReferenceAudio = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, value) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = Path(to_absolute_path(str(path)))
        return path.resolve()

    def _ensure_code_path(self) -> None:
        if not self.code_path.exists():
            raise FileNotFoundError(f"Fish-Speech code path not found: {self.code_path}")
        if str(self.code_path) not in sys.path:
            sys.path.insert(0, str(self.code_path))

    def _validate_resources(self) -> None:
        if not self.llama_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Fish-Speech llama checkpoint path not found: {self.llama_checkpoint_path}"
            )
        if not self.decoder_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Fish-Speech decoder checkpoint not found: {self.decoder_checkpoint_path}"
            )

    def _ensure_model(self) -> None:
        if self._model_manager is not None and self._tts_engine is not None:
            return

        try:
            from tools.server.model_manager import ModelManager
            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
        except ModuleNotFoundError as exc:
            hint = (
                "Ensure Fish-Speech is installed (e.g., `pip install -e checkpoints/fish_speech`) "
                "and the code_path is correct."
            )
            raise ModuleNotFoundError(f"Unable to import Fish-Speech modules: {hint}") from exc

        device_str = self._map_device_to_string()

        self.logger.info(
            "[FishSpeech] Initialising model (device=%s, half=%s, compile=%s)...",
            device_str,
            self.half,
            self.compile,
        )
        self._model_manager = ModelManager(
            mode=self.mode,
            device=device_str,
            half=self.half,
            compile=self.compile,
            llama_checkpoint_path=str(self.llama_checkpoint_path),
            decoder_checkpoint_path=str(self.decoder_checkpoint_path),
            decoder_config_name=self.decoder_config_name,
        )
        self._tts_engine = self._model_manager.tts_inference_engine
        self._ServeTTSRequest = ServeTTSRequest
        self._ServeReferenceAudio = ServeReferenceAudio

    def _map_device_to_string(self) -> str:
        if self.device.type == "cuda":
            index = 0 if self.device.index is None else self.device.index
            return f"cuda:{index}"
        return self.device.type

    def _build_request(
        self,
        text: str,
        reference_path: Path,
        reference_text: Optional[str] = None,
    ):
        assert self._ServeTTSRequest is not None and self._ServeReferenceAudio is not None
        reference_bytes = reference_path.read_bytes()
        transcript = (reference_text or self.default_reference_text).strip() or self.default_reference_text
        reference = self._ServeReferenceAudio(audio=reference_bytes, text=transcript)
        return self._ServeTTSRequest(
            text=text,
            references=[reference],
            reference_id=None,
            format=self.format,
            max_new_tokens=self.max_new_tokens,
            chunk_length=self.chunk_length,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            normalize=self.normalize,
            seed=self.seed,
            use_memory_cache=self.use_memory_cache,
            streaming=False,
        )

    def _run_inference(self, request) -> Tuple[int, np.ndarray]:
        assert self._tts_engine is not None
        final_audio: Optional[Tuple[int, np.ndarray]] = None

        for result in self._tts_engine.inference(request):
            code = getattr(result, "code", None)
            if code == "error":
                error = getattr(result, "error", RuntimeError("Unknown Fish-Speech error"))
                raise RuntimeError(f"Fish-Speech inference failed: {error}")
            if code == "final":
                final_audio = result.audio

        if final_audio is None:
            raise RuntimeError("Fish-Speech returned no audio for the request.")

        sample_rate, audio_array = final_audio
        audio_array = np.asarray(audio_array, dtype=np.float32)
        return sample_rate, audio_array

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._ensure_model()
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[FishSpeech] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Fish-Speech adversary.")

        prompt_count = self._count_available_prompts(samples)

        self.logger.info(
            "[FishSpeech] Rendering %d utterances with %d prompt audios...",
            len(samples),
            prompt_count,
        )

        self._log_attack_plan("FishSpeech", samples, prompt_count)

        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                self.logger.warning(
                    "[FishSpeech] Sample %d missing prompt audio; skipping.",
                    idx,
                )
                continue
            request_text = (sample.target_text or "").strip()
            if not request_text:
                request_text = (sample.prompt_text or "").strip()
            if not request_text:
                request_text = self.default_reference_text
            reference_transcript = (sample.prompt_text or "").strip() or None
            request = self._build_request(
                text=request_text,
                reference_path=reference_path,
                reference_text=reference_transcript,
            )

            synth_start = time.perf_counter()
            sample_rate, audio_array = self._run_inference(request)
            synth_elapsed = time.perf_counter() - synth_start

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            output_name = self._cloned_filename(sample, idx)

            self._log_clone_request(
                "FishSpeech",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                request_text,
                prompt_transcript=reference_transcript,
            )

            output_path = speaker_dir / output_name
            sf.write(output_path, audio_array, sample_rate)
            self._record_synthesis_timing(output_path, synth_elapsed)

        self._flush_synthesis_timings()
        self.logger.info("[FishSpeech] Generated %d utterances.", len(samples))
