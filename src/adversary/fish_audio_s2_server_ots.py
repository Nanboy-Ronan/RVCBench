"""Fish Audio S2 local API-server adversary integration."""
from __future__ import annotations

import time
from pathlib import Path

import ormsgpack
import requests

from .base_adversary import BaseAdversary


class FishAudioS2ServerZeroShotAdversary(BaseAdversary):
    """Runs open-weight Fish Audio S2 cloning through Fish Speech's /v1/tts API."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.endpoint_url = str(
            self.config.get("endpoint_url", "http://localhost:8001/v1/tts")
        )
        self.timeout = float(self.config.get("timeout", 600))
        self.request_attempts = int(self.config.get("request_attempts", 3))
        self.retry_delay_sec = float(self.config.get("retry_delay_sec", 10))
        self.max_samples = self.config.get("max_samples")
        self.response_format = str(self.config.get("response_format", "wav"))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 2048))
        self.chunk_length = int(self.config.get("chunk_length", 300))
        self.top_p = float(self.config.get("top_p", 0.8))
        self.repetition_penalty = float(
            self.config.get("repetition_penalty", 1.1)
        )
        self.temperature = float(self.config.get("temperature", 0.8))
        self.normalize = bool(self.config.get("normalize", True))
        self.use_memory_cache = str(self.config.get("use_memory_cache", "off"))
        self.seed = self.config.get("seed")
        self.default_reference_text = str(
            self.config.get(
                "default_reference_text",
                "Here is a sample exhibiting the desired voice characteristics.",
            )
        )

        if self.response_format != "wav":
            raise ValueError("Benchmark output must use response_format='wav'.")
        if not 100 <= self.chunk_length <= 300:
            raise ValueError("Fish Audio S2 requires chunk_length between 100 and 300.")
        if self.use_memory_cache not in {"on", "off"}:
            raise ValueError("use_memory_cache must be 'on' or 'off'.")

    def _build_payload(self, text: str, reference_path: Path, reference_text: str) -> bytes:
        payload = {
            "text": text,
            "references": [
                {
                    "audio": reference_path.read_bytes(),
                    "text": reference_text,
                }
            ],
            "reference_id": None,
            "format": self.response_format,
            "max_new_tokens": self.max_new_tokens,
            "chunk_length": self.chunk_length,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "normalize": self.normalize,
            "streaming": False,
            "use_memory_cache": self.use_memory_cache,
            "seed": self.seed,
        }
        return ormsgpack.packb(payload)

    def _post_generation(self, payload: bytes) -> bytes:
        last_error = None
        for attempt in range(1, self.request_attempts + 1):
            try:
                response = requests.post(
                    self.endpoint_url,
                    params={"format": "msgpack"},
                    data=payload,
                    headers={"content-type": "application/msgpack"},
                    timeout=self.timeout,
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
                    "[FishAudioS2] Request attempt %d/%d failed (%s); retrying in %.1fs.",
                    attempt,
                    self.request_attempts,
                    last_error,
                    self.retry_delay_sec,
                )
                time.sleep(self.retry_delay_sec)

        raise RuntimeError(
            f"Fish Audio S2 request failed after {self.request_attempts} attempts: "
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
                    "[FishAudioS2] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Fish Audio S2.")

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan("FishAudioS2", samples, prompt_count)

        generated = 0
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                self.logger.warning(
                    "[FishAudioS2] Sample %d missing prompt audio; skipping.", idx
                )
                continue

            text = (sample.target_text or sample.prompt_text or "").strip()
            if not text:
                text = self.default_reference_text
            reference_text = (
                (sample.prompt_text or "").strip() or self.default_reference_text
            )

            self._log_clone_request(
                "FishAudioS2",
                idx,
                len(samples),
                str(sample.speaker_id),
                reference_path,
                text,
                prompt_transcript=reference_text,
            )

            synth_start = time.perf_counter()
            audio_bytes = self._post_generation(
                self._build_payload(text, reference_path, reference_text)
            )
            synth_elapsed = time.perf_counter() - synth_start

            speaker_dir = self._speaker_output_dir(output_dir, str(sample.speaker_id))
            rendered_path = speaker_dir / self._cloned_filename(sample, idx)
            rendered_path.write_bytes(audio_bytes)
            self._record_synthesis_timing(rendered_path, synth_elapsed)
            generated += 1

        self._flush_synthesis_timings()
        self.logger.info("[FishAudioS2] Generated %d utterances.", generated)
