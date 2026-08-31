"""OpenAI-compatible local speech-server adversary integration."""
from __future__ import annotations

import io
import re
import textwrap
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from .base_adversary import BaseAdversary


class OpenAISpeechServerZeroShotAdversary(BaseAdversary):
    """Runs zero-shot cloning through a local /v1/audio/speech server.

    This is intended for open-weight servers such as SGLang-Omni or vLLM-Omni,
    not hosted provider APIs. Reference audio is passed as a file:// URI, so
    the server must be launched with permission to read the benchmark
    data/results directories.
    """

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.endpoint_url = str(
            self.config.get("endpoint_url", "http://localhost:8000/v1/audio/speech")
        )
        self.request_model = self.config.get("request_model")
        self.response_format = str(self.config.get("response_format", "wav"))
        self.timeout = float(self.config.get("timeout", 600))
        self.request_attempts = int(self.config.get("request_attempts", 3))
        self.retry_delay_sec = float(self.config.get("retry_delay_sec", 10))
        self.max_samples = self.config.get("max_samples")
        self.default_reference_text = str(
            self.config.get(
                "default_reference_text",
                "Here is a sample of the desired voice.",
            )
        )
        self.voice = self.config.get("voice")
        self.temperature = self.config.get("temperature")
        self.top_p = self.config.get("top_p")
        self.top_k = self.config.get("top_k")
        self.max_new_tokens = self.config.get("max_new_tokens")
        self.max_text_chars_per_request = self.config.get(
            "max_text_chars_per_request"
        )
        self.chunk_silence_sec = float(self.config.get("chunk_silence_sec", 0.0))
        if self.chunk_silence_sec < 0:
            raise ValueError("chunk_silence_sec must be non-negative.")
        self.repetition_penalty = self.config.get("repetition_penalty")
        self.speed = self.config.get("speed")
        self.min_reference_duration_sec = float(
            self.config.get("min_reference_duration_sec", 1.0)
        )
        self.reference_padding_sec = float(
            self.config.get("reference_padding_sec", 0.05)
        )
        self.extra_request_fields = self.config.get("extra_request_fields") or {}

        if self.response_format != "wav":
            raise ValueError("Benchmark output must use response_format='wav'.")

    def _audio_duration_sec(self, path: Optional[Path]) -> Optional[float]:
        if path is None:
            return None
        try:
            with sf.SoundFile(str(path)) as handle:
                if not handle.samplerate:
                    return None
                return handle.frames / handle.samplerate
        except Exception:
            return None

    def _prepare_reference_path(
        self,
        reference_path: Optional[Path],
        reference_duration: Optional[float],
        cache_dir: Path,
        sample_index: int,
        label: str,
    ) -> Optional[Path]:
        if (
            reference_path is None
            or reference_duration is None
            or reference_duration >= self.min_reference_duration_sec
        ):
            return reference_path

        audio, sample_rate = sf.read(
            str(reference_path), dtype="float32", always_2d=True
        )
        minimum_frames = int(
            np.ceil(
                (self.min_reference_duration_sec + self.reference_padding_sec)
                * sample_rate
            )
        )
        padding_frames = max(0, minimum_frames - audio.shape[0])
        if padding_frames:
            audio = np.pad(audio, ((0, padding_frames), (0, 0)))

        cache_dir.mkdir(parents=True, exist_ok=True)
        padded_path = cache_dir / f"{sample_index:06d}_{reference_path.stem}.wav"
        sf.write(str(padded_path), audio, sample_rate, subtype="PCM_16")
        self.logger.info(
            "[%s] Sample %d prompt audio is %.2fs; padded to %.2fs for the speech server.",
            label,
            sample_index,
            reference_duration,
            audio.shape[0] / sample_rate,
        )
        return padded_path

    def _build_payload(
        self,
        *,
        text: str,
        reference_path: Optional[Path],
        reference_text: Optional[str],
    ) -> dict:
        payload = {
            "input": text,
            "response_format": self.response_format,
            "stream": False,
        }
        if self.request_model:
            payload["model"] = str(self.request_model)
        if self.voice and reference_path is None:
            payload["voice"] = str(self.voice)

        if reference_path is not None:
            payload["references"] = [
                {
                    "audio_path": reference_path.resolve().as_uri(),
                    "text": reference_text or self.default_reference_text,
                }
            ]

        optional_fields = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "speed": self.speed,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value

        if self.extra_request_fields:
            payload.update(dict(self.extra_request_fields))
        return payload

    def _split_text(self, text: str) -> list[str]:
        if self.max_text_chars_per_request is None:
            return [text]
        try:
            max_chars = int(self.max_text_chars_per_request)
        except (TypeError, ValueError):
            self.logger.warning(
                "[SpeechServer] Invalid max_text_chars_per_request=%r; "
                "sending text in one request.",
                self.max_text_chars_per_request,
            )
            return [text]
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]

        chunks = []
        current = ""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            pieces = textwrap.wrap(
                sentence,
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [sentence]
            for piece in pieces:
                candidate = f"{current} {piece}".strip()
                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            chunks.append(current)
        return chunks or [text]

    @staticmethod
    def _write_concatenated_wavs(
        audio_responses: list[bytes],
        output_path: Path,
        silence_sec: float = 0.0,
    ):
        decoded = []
        sample_rate = None
        channels = None
        for audio_bytes in audio_responses:
            audio, current_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32", always_2d=True
            )
            if sample_rate is None:
                sample_rate = current_rate
                channels = audio.shape[1]
            elif current_rate != sample_rate or audio.shape[1] != channels:
                raise RuntimeError(
                    "Speech server returned incompatible WAV formats across text chunks."
                )
            decoded.append(audio)

        joined = []
        silence_frames = round(max(0.0, silence_sec) * sample_rate)
        silence = np.zeros((silence_frames, channels), dtype=np.float32)
        for index, audio in enumerate(decoded):
            if index and silence_frames:
                joined.append(silence)
            joined.append(audio)
        sf.write(
            str(output_path),
            np.concatenate(joined, axis=0),
            sample_rate,
            subtype="PCM_16",
        )

    def _post_generation(self, payload: dict) -> bytes:
        last_error = None
        for attempt in range(1, self.request_attempts + 1):
            try:
                response = requests.post(
                    self.endpoint_url, json=payload, timeout=self.timeout
                )
                if response.status_code < 400:
                    return response.content
                last_error = RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            except requests.RequestException as exc:
                last_error = exc

            if attempt < self.request_attempts:
                self.logger.warning(
                    "[SpeechServer] Request attempt %d/%d failed (%s); "
                    "retrying in %.1fs.",
                    attempt,
                    self.request_attempts,
                    last_error,
                    self.retry_delay_sec,
                )
                time.sleep(self.retry_delay_sec)

        raise RuntimeError(
            f"Speech server request failed after {self.request_attempts} attempts: "
            f"{last_error}"
        )

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[SpeechServer] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for speech-server adversary.")

        prompt_count = self._count_available_prompts(samples)
        label = str(self.config.get("name", "SpeechServer"))
        self._log_attack_plan(label, samples, prompt_count)
        reference_cache_dir = output_dir.parent / "padded_references"

        generated = 0
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            reference_duration = self._audio_duration_sec(reference_path)
            reference_path = self._prepare_reference_path(
                reference_path,
                reference_duration,
                reference_cache_dir,
                idx,
                label,
            )
            text = (sample.target_text or "").strip()
            if not text:
                text = (sample.prompt_text or "").strip()
            if not text:
                text = self.default_reference_text

            reference_text = (sample.prompt_text or "").strip() or self.default_reference_text
            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            output_name = self._cloned_filename(sample, idx)
            rendered_path = speaker_dir / output_name

            self._log_clone_request(
                label,
                idx,
                len(samples),
                speaker_id,
                reference_path,
                text,
                prompt_transcript=reference_text,
            )

            text_chunks = self._split_text(text)
            synth_start = time.perf_counter()
            audio_responses = []
            for chunk_index, text_chunk in enumerate(text_chunks, start=1):
                if len(text_chunks) > 1:
                    self.logger.info(
                        "[%s] Sample %d text chunk %d/%d (%d chars).",
                        label,
                        idx + 1,
                        chunk_index,
                        len(text_chunks),
                        len(text_chunk),
                    )
                payload = self._build_payload(
                    text=text_chunk,
                    reference_path=reference_path,
                    reference_text=reference_text,
                )
                audio_responses.append(self._post_generation(payload))
            synth_elapsed = time.perf_counter() - synth_start
            if len(audio_responses) == 1:
                rendered_path.write_bytes(audio_responses[0])
            else:
                self._write_concatenated_wavs(
                    audio_responses,
                    rendered_path,
                    silence_sec=self.chunk_silence_sec,
                )
            self._record_synthesis_timing(rendered_path, synth_elapsed)
            generated += 1

        self._flush_synthesis_timings()
        self.logger.info("[%s] Generated %d utterances.", label, generated)
