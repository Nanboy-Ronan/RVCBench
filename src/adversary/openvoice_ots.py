"""OpenVoice zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.openvoice import OpenVoiceGenerator, OpenVoiceGeneratorConfig


class OpenVoiceZeroShotAdversary(BaseAdversary):
    """Runs OpenVoice V2 with MeloTTS base speakers for zero-shot cloning."""

    MODEL_NAME = "OpenVoice"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.get("code_path", "checkpoints/OpenVoice"))).resolve()
        melo_path = self.config.get("melo_code_path") or "checkpoints/MeloTTS"
        self.melo_code_path = (
            Path(to_absolute_path(melo_path)).resolve() if melo_path else None
        )
        self.converter_config_path = Path(
            to_absolute_path(
                self.config.get(
                    "converter_config_path",
                    "checkpoints/OpenVoice/checkpoints_v2/converter/config.json",
                )
            )
        ).resolve()
        self.converter_checkpoint_path = Path(
            to_absolute_path(
                self.config.get(
                    "converter_checkpoint_path",
                    "checkpoints/OpenVoice/checkpoints_v2/converter/checkpoint.pth",
                )
            )
        ).resolve()
        self.base_speaker_dir = Path(
            to_absolute_path(
                self.config.get(
                    "base_speaker_dir",
                    "checkpoints/OpenVoice/checkpoints_v2/base_speakers/ses",
                )
            )
        ).resolve()
        self.speed = float(self.config.get("speed", 1.0))
        self.tau = float(self.config.get("tau", 0.3))
        self.enable_watermark = bool(self.config.get("enable_watermark", False))
        self.melo_use_hf = bool(self.config.get("melo_use_hf", True))
        self.default_language = str(self.config.get("default_language", "EN"))
        self.language_source = str(self.config.get("language_source", "target")).lower()
        self.source_speaker_key = self.config.get("source_speaker_key")
        self.max_samples = self.config.get("max_samples")

        self._generator: Optional[OpenVoiceGenerator] = None

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = OpenVoiceGeneratorConfig(
            code_path=self.code_path,
            converter_config_path=self.converter_config_path,
            converter_checkpoint_path=self.converter_checkpoint_path,
            base_speaker_dir=self.base_speaker_dir,
            melo_code_path=self.melo_code_path,
            device=str(self.device),
            speed=self.speed,
            tau=self.tau,
            enable_watermark=self.enable_watermark,
            melo_use_hf=self.melo_use_hf,
            default_melo_language=self.default_language,
            source_speaker_key=self.source_speaker_key,
        )
        self._generator = OpenVoiceGenerator(generator_config, self.device, self.logger)

    def _select_language(self, sample) -> str:
        if self.language_source == "prompt" and sample.prompt_language:
            return str(sample.prompt_language)
        if sample.target_language:
            return str(sample.target_language)
        if sample.prompt_language:
            return str(sample.prompt_language)
        return self.default_language

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
            if not target_text:
                target_text = (sample.prompt_text or "").strip()
            if not target_text:
                if self.logger:
                    self.logger.warning("[%s] Sample %d has empty target text; skipping.", self.MODEL_NAME, idx)
                continue

            speaker_id = str(sample.speaker_id)
            language = self._select_language(sample)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                self.MODEL_NAME,
                idx,
                len(samples),
                speaker_id,
                reference_path,
                target_text,
                prompt_transcript=sample.prompt_text,
            )

            try:
                synth_start = time.perf_counter()
                wav, sample_rate = self._generator.generate(
                    text=target_text,
                    reference_audio=reference_path,
                    language=language,
                    source_speaker_key=self.source_speaker_key,
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
