"""OpenVoice zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

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
        self.seed = self.config.get("seed")

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
            melo_models=(OmegaConf.to_container(self.config.melo_models, resolve=True)
                         if OmegaConf.is_config(self.config.get("melo_models"))
                         else self.config.get("melo_models")),
            text_models=(OmegaConf.to_container(self.config.text_models, resolve=True)
                         if OmegaConf.is_config(self.config.get("text_models"))
                         else self.config.get("text_models")),
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

    def generate_sample(self, sample, *, output_dir):
        self._ensure_generator()
        reference_path = self._resolve_prompt_path(sample)
        if reference_path is None:
            raise FileNotFoundError(f"Missing OpenVoice reference: {sample.prompt_path}")
        target_text = (sample.target_text or "").strip() or (sample.prompt_text or "").strip()
        if not target_text:
            raise ValueError("OpenVoice target text cannot be empty")
        language = self._select_language(sample)
        self._log_clone_request(self.MODEL_NAME, 0, 1, str(sample.speaker_id),
                                reference_path, target_text, prompt_transcript=sample.prompt_text)
        seed = self._sample_seed(sample)
        if seed is not None:
            from src.utils.seeding import configure_seeds
            configure_seeds(seed, logger=None)
        started = time.perf_counter()
        wav, sample_rate = self._generator.generate(
            text=target_text, reference_audio=reference_path, language=language,
            source_speaker_key=self.source_speaker_key)
        elapsed = time.perf_counter() - started
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
            raise ValueError("OpenVoice output must be nonempty finite mono audio")
        wav = np.clip(wav, -1.0, 1.0)
        path = self._speaker_output_dir(Path(output_dir).resolve(), str(sample.speaker_id)) / self._cloned_filename(sample, sample.index)
        sf.write(path, wav, sample_rate)
        return path, elapsed

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path
        self._ensure_generator()
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)
        samples = dataset.get_zero_shot_samples(max_samples=self.max_samples)
        if not samples:
            raise RuntimeError(f"No zero-shot samples available for {self.MODEL_NAME} adversary.")
        self._log_attack_plan(self.MODEL_NAME, samples, self._count_available_prompts(samples))
        try:
            for sample in samples:
                path, elapsed = self.generate_sample(sample, output_dir=output_dir)
                self._record_synthesis_timing(path, elapsed)
        finally:
            self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, len(samples), len(samples))
