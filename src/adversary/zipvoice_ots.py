"""ZipVoice zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.zipvoice import ZipVoiceGenerator, ZipVoiceGeneratorConfig


class ZipVoiceZeroShotAdversary(BaseAdversary):
    """Runs ZipVoice for zero-shot voice cloning."""

    MODEL_NAME = "ZipVoice"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.get("code_path", "checkpoints/ZipVoice"))).resolve()
        model_dir_value = self.config.get("model_dir")
        self.model_dir = (
            Path(to_absolute_path(model_dir_value)).resolve()
            if model_dir_value
            else None
        )
        vocoder_value = self.config.get("vocoder_path")
        self.vocoder_path = (
            Path(to_absolute_path(vocoder_value)).resolve()
            if vocoder_value
            else None
        )
        trt_engine_value = self.config.get("trt_engine_path")
        self.trt_engine_path = (
            Path(to_absolute_path(trt_engine_value)).resolve()
            if trt_engine_value
            else None
        )
        runtime_python_value = self.config.get("runtime_python")
        self.runtime_python = (
            Path(to_absolute_path(str(runtime_python_value))).resolve()
            if runtime_python_value not in (None, "")
            else None
        )

        self.model_name = str(self.config.get("model_name", "zipvoice")).strip().lower()
        self.checkpoint_name = str(self.config.get("checkpoint_name", "model.pt"))
        self.tokenizer = str(self.config.get("tokenizer", "libritts"))
        self.default_lang = str(self.config.get("lang", "en-us"))
        self.use_sample_language = bool(self.config.get("use_sample_language", False))
        self.guidance_scale = self.config.get("guidance_scale")
        self.num_step = self.config.get("num_step")
        self.feat_scale = float(self.config.get("feat_scale", 0.1))
        self.speed = float(self.config.get("speed", 1.0))
        self.t_shift = float(self.config.get("t_shift", 0.5))
        self.target_rms = float(self.config.get("target_rms", 0.1))
        self.raw_evaluation = bool(self.config.get("raw_evaluation", False))
        self.max_duration = float(self.config.get("max_duration", 100.0))
        self.remove_long_sil = bool(self.config.get("remove_long_sil", False))
        self.num_thread = int(self.config.get("num_thread", 1))
        self.seed = self.config.get("seed")
        self.default_prompt_text = str(
            self.config.get(
                "default_prompt_text",
                "Here is a sample of the desired voice.",
            )
        )
        self.max_samples = self.config.get("max_samples")

        self._generator: Optional[ZipVoiceGenerator] = None

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        generator_config = ZipVoiceGeneratorConfig(
            code_path=self.code_path,
            model_name=self.model_name,
            model_dir=self.model_dir,
            checkpoint_name=self.checkpoint_name,
            vocoder_path=self.vocoder_path,
            tokenizer=self.tokenizer,
            lang=self.default_lang,
            guidance_scale=self.guidance_scale,
            num_step=self.num_step,
            feat_scale=self.feat_scale,
            speed=self.speed,
            t_shift=self.t_shift,
            target_rms=self.target_rms,
            raw_evaluation=self.raw_evaluation,
            max_duration=self.max_duration,
            remove_long_sil=self.remove_long_sil,
            trt_engine_path=self.trt_engine_path,
            num_thread=self.num_thread,
            seed=(int(self.seed) if self.seed is not None else None),
            runtime_python=self.runtime_python,
        )
        self._generator = ZipVoiceGenerator(generator_config, self.device, self.logger)

    def _select_language(self, sample) -> str:
        if not self.use_sample_language:
            return self.default_lang
        candidate = sample.target_language or sample.prompt_language
        if candidate:
            return str(candidate)
        return self.default_lang

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
                self.logger.warning(
                    "[%s] Sample %d missing prompt audio; skipping.",
                    self.MODEL_NAME,
                    idx,
                )
                continue

            target_text = (sample.target_text or "").strip()
            prompt_text = (sample.prompt_text or "").strip()
            if not target_text:
                target_text = prompt_text
            if not target_text:
                self.logger.warning(
                    "[%s] Sample %d has no target text; skipping.",
                    self.MODEL_NAME,
                    idx,
                )
                continue
            if not prompt_text:
                prompt_text = self.default_prompt_text

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
                synth_start = time.perf_counter()
                waveform, sample_rate = self._generator.generate(
                    text=target_text,
                    prompt_wav=reference_path,
                    prompt_text=prompt_text,
                    lang=self._select_language(sample),
                )
                synth_elapsed = time.perf_counter() - synth_start
            except Exception as exc:
                self.logger.error(
                    "[%s] Generation failed for sample %d (%s): %s",
                    self.MODEL_NAME,
                    idx,
                    speaker_id,
                    exc,
                )
                continue

            output_name = self._cloned_filename(sample, idx)
            output_wav_path = speaker_dir / output_name
            sf.write(str(output_wav_path), waveform, sample_rate)
            self._record_synthesis_timing(output_wav_path, synth_elapsed)
            completed += 1

        self._flush_synthesis_timings()
        self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, completed, len(samples))
