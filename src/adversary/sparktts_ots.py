from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
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
        self.seed = self.config.get("seed")

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
            seed=self.seed,
            retry_without_prompt_text=self.config.get("retry_without_prompt_text", False),
        )
        self._generator = SparkTTSGenerator(generator_config, self.device, self.logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_sample(self, sample, *, output_dir):
        self._ensure_generator()
        reference_path = self._resolve_prompt_path(sample)
        if reference_path is None:
            raise FileNotFoundError(f"Missing SparkTTS reference: {sample.prompt_path}")
        prompt_text = (sample.prompt_text or "").strip() or None
        target_text = (sample.target_text or "").strip() or (sample.prompt_text or "").strip()
        if not target_text:
            raise ValueError("SparkTTS sample has no target or prompt text")
        self._log_clone_request("SparkTTS", 0, 1, str(sample.speaker_id),
                                reference_path, target_text, prompt_transcript=prompt_text)
        started = time.perf_counter()
        wav, sr = self._generator.generate(text=target_text, prompt_audio=reference_path,
                                          prompt_text=prompt_text, sample_index=sample.index)
        elapsed = time.perf_counter() - started
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
            raise ValueError("SparkTTS output must be nonempty finite mono audio")
        path = self._speaker_output_dir(Path(output_dir).resolve(), str(sample.speaker_id)) / self._cloned_filename(sample, sample.index)
        sf.write(path, wav, sr)
        return path, elapsed

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path
        self._ensure_generator()
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)
        samples = dataset.get_zero_shot_samples(max_samples=self.max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Spark-TTS adversary.")
        self._log_attack_plan("SparkTTS", samples, self._count_available_prompts(samples))
        try:
            for sample in samples:
                path, elapsed = self.generate_sample(sample, output_dir=output_dir)
                self._record_synthesis_timing(path, elapsed)
        finally:
            self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[SparkTTS] Generated %d utterances.", len(samples))
