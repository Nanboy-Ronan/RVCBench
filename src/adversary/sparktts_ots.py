from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.sparktts import SparkTTSGenerator, SparkTTSGeneratorConfig


class SparkTTSZeroShotAdversary(BaseAdversary):
    """Runs Spark-TTS zero-shot cloning pipeline."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        self.model_dir = str(self.config.model_dir)
        self.temperature = float(self.config.get("temperature", 0.8))
        self.top_k = int(self.config.get("top_k", 50))
        self.top_p = float(self.config.get("top_p", 0.95))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")

        self._generator: Optional[SparkTTSGenerator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = SparkTTSGeneratorConfig(
            code_path=self.code_path,
            model_dir=self.model_dir,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        self._generator = SparkTTSGenerator(generator_config, self.device, self.logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._ensure_generator()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[SparkTTS] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Spark-TTS adversary.")

        sample_prompt_texts: Dict[str, str] = {}
        for sample in samples:
            prompt_path = self._resolve_prompt_path(sample)
            if prompt_path is not None:
                sample_prompt_texts[str(prompt_path)] = sample.prompt_text or ""

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan("SparkTTS", samples, prompt_count)

        assert self._generator is not None
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[SparkTTS] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            lookup_key = str(reference_path)
            prompt_transcript = sample_prompt_texts.get(lookup_key, "").strip() or None
            desired_text = (sample.target_text or "").strip()
            if not desired_text:
                desired_text = (sample.prompt_text or "").strip()
            if not desired_text:
                desired_text = prompt_transcript or ""

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "SparkTTS",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                desired_text,
                prompt_transcript=prompt_transcript,
            )

            synth_start = time.perf_counter()
            wav, sr = self._generator.generate(
                text=desired_text,
                prompt_audio=reference_path,
                prompt_text=prompt_transcript,
                sample_index=idx,
            )
            synth_elapsed = time.perf_counter() - synth_start

            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, sr)
            self._record_synthesis_timing(output_path, synth_elapsed)

        self._flush_synthesis_timings()
        if self.logger is not None:
            self.logger.info("[SparkTTS] Generated %d utterances.", len(samples))
