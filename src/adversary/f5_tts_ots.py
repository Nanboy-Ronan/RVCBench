"""F5-TTS zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.f5_tts import F5TTSGenerator, F5TTSGeneratorConfig


class F5TTSZeroShotAdversary(BaseAdversary):
    """Runs F5-TTS for zero-shot voice cloning."""

    MODEL_NAME = "F5-TTS"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.model_name = str(self.config.get("model", "F5TTS_v1_Base"))
        self.ckpt_file = self._resolve_optional_path(self.config.get("ckpt_file", ""))
        self.vocab_file = self._resolve_optional_path(self.config.get("vocab_file", ""))
        self.vocoder_local_path = self._resolve_optional_path(self.config.get("vocoder_local_path"))
        self.hf_cache_dir = self._resolve_optional_path(self.config.get("hf_cache_dir"))
        self.ode_method = str(self.config.get("ode_method", "euler"))
        self.use_ema = bool(self.config.get("use_ema", True))
        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")

        self.default_prompt_text = str(
            self.config.get("default_prompt_text", "Here is a sample of the desired voice.")
        ).strip()
        self.auto_transcribe_missing_ref_text = bool(
            self.config.get("auto_transcribe_missing_ref_text", True)
        )
        transcription_language = self.config.get("transcription_language")
        self.transcription_language = (
            str(transcription_language).strip() if transcription_language not in (None, "") else None
        )

        infer_kwargs = {}
        for key in (
            "target_rms",
            "cross_fade_duration",
            "sway_sampling_coef",
            "cfg_strength",
            "nfe_step",
            "speed",
            "fix_duration",
            "remove_silence",
        ):
            value = self.config.get(key)
            if value is not None:
                infer_kwargs[key] = value

        self._generator: Optional[F5TTSGenerator] = None
        self._transcript_cache = {}
        self._infer_kwargs = infer_kwargs

    def _resolve_optional_path(self, value) -> Optional[str]:
        if value in (None, ""):
            return None
        raw = str(value)
        path = Path(raw).expanduser()
        if path.exists():
            return str(path.resolve())
        try:
            absolute_path = Path(to_absolute_path(raw))
        except Exception:
            absolute_path = path
        if absolute_path.exists():
            return str(absolute_path.resolve())
        return raw

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = F5TTSGeneratorConfig(
            model=self.model_name,
            ckpt_file=self.ckpt_file or "",
            vocab_file=self.vocab_file or "",
            ode_method=self.ode_method,
            use_ema=self.use_ema,
            vocoder_local_path=self.vocoder_local_path,
            hf_cache_dir=self.hf_cache_dir,
            infer_kwargs=dict(self._infer_kwargs),
        )
        self._generator = F5TTSGenerator(generator_config, self.device, self.logger)

    def _resolve_prompt_transcript(self, reference_path: Path, prompt_text: str) -> str:
        assert self._generator is not None

        if prompt_text:
            return prompt_text
        if not self.auto_transcribe_missing_ref_text:
            return self.default_prompt_text

        cache_key = str(reference_path.resolve())
        cached = self._transcript_cache.get(cache_key)
        if cached is not None:
            return cached

        transcript = self._generator.transcribe(
            str(reference_path.resolve()),
            language=self.transcription_language,
        )
        transcript = transcript.strip() or self.default_prompt_text
        self._transcript_cache[cache_key] = transcript
        return transcript

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path

        self._ensure_generator()
        assert self._generator is not None

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                if self.logger:
                    self.logger.warning(
                        "[%s] Invalid max_samples '%s'; processing full dataset.",
                        self.MODEL_NAME,
                        self.max_samples,
                    )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError(f"No zero-shot samples available for {self.MODEL_NAME} adversary.")

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan(self.MODEL_NAME, samples, prompt_count)

        completed = 0
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[%s] Sample %d missing prompt audio; skipping.",
                        self.MODEL_NAME,
                        idx,
                    )
                continue

            target_text = (sample.target_text or "").strip()
            prompt_text = (sample.prompt_text or "").strip()
            if not target_text:
                target_text = prompt_text or self.default_prompt_text

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                self.MODEL_NAME,
                idx,
                len(samples),
                speaker_id,
                reference_path,
                target_text,
                prompt_transcript=prompt_text,
            )

            try:
                ref_text = self._resolve_prompt_transcript(reference_path, prompt_text)
                sample_seed = None if self.seed is None else int(self.seed) + int(idx)
                synth_start = time.perf_counter()
                wav, sample_rate = self._generator.generate(
                    ref_audio=str(reference_path.resolve()),
                    ref_text=ref_text,
                    gen_text=target_text,
                    seed=sample_seed,
                )
                synth_elapsed = time.perf_counter() - synth_start
            except Exception as exc:
                if self.logger:
                    self.logger.error(
                        "[%s] Generation failed for sample %d (%s): %s",
                        self.MODEL_NAME,
                        idx,
                        speaker_id,
                        exc,
                    )
                continue

            wav = np.asarray(wav, dtype=np.float32)
            if not np.isfinite(wav).all():
                wav = np.nan_to_num(wav)
            wav = np.clip(wav, -1.0, 1.0)

            output_filename = self._cloned_filename(sample, idx)
            output_wav_path = speaker_dir / output_filename
            sf.write(str(output_wav_path), wav, sample_rate)
            self._record_synthesis_timing(output_wav_path, synth_elapsed)
            completed += 1

        self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, completed, len(samples))
