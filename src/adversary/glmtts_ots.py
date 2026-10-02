"""GLM-TTS zero-shot adversary integration."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.glmtts.synthesizer import GLMTTSSynthesizer, GLMTTSSynthesizerConfig


class GLMTTSZeroShotAdversary(BaseAdversary):
    """Runs GLM-TTS for zero-shot voice cloning."""

    MODEL_NAME = "GLM-TTS"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        code_path = self.config.get("code_path", "checkpoints/GLM-TTS")
        self.code_path = Path(to_absolute_path(code_path)).resolve()
        ckpt_dir = self.config.get("ckpt_dir")
        self.ckpt_dir = Path(to_absolute_path(ckpt_dir)).resolve() if ckpt_dir else None
        frontend_dir = self.config.get("frontend_dir")
        self.frontend_dir = Path(to_absolute_path(frontend_dir)).resolve() if frontend_dir else None

        self.sample_rate = int(self.config.get("sample_rate", 24000))
        self.use_cache = bool(self.config.get("use_cache", True))
        self.use_phoneme = bool(self.config.get("use_phoneme", False))
        self.sample_method = str(self.config.get("sample_method", "ras"))
        self.seed = self.config.get("seed")
        self.max_samples = self.config.get("max_samples")
        self._synthesizer: Optional[GLMTTSSynthesizer] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_synthesizer(self) -> None:
        if self._synthesizer is not None:
            return

        synth_config = GLMTTSSynthesizerConfig(
            code_path=self.code_path,
            sample_rate=self.sample_rate,
            use_cache=self.use_cache,
            use_phoneme=self.use_phoneme,
            sample_method=self.sample_method,
            seed=self.seed,
            ckpt_dir=self.ckpt_dir,
            frontend_dir=self.frontend_dir,
        )
        self._synthesizer = GLMTTSSynthesizer(synth_config, self.device, self.logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_sample(self, sample, *, output_dir):
        reference_path = self._resolve_prompt_path(sample)
        if reference_path is None:
            raise FileNotFoundError(f"Missing GLM-TTS reference audio: {sample.prompt_path}")
        target_text = (sample.target_text or "").strip()
        prompt_text = (sample.prompt_text or "").strip()
        if not target_text:
            raise ValueError('GLM-TTS requires nonempty target text')
        if not prompt_text:
            raise ValueError('GLM-TTS requires the actual reference transcript')
        self._ensure_synthesizer()
        seed = self._sample_seed(sample)
        if seed is not None:
            from src.utils.seeding import configure_seeds
            configure_seeds(seed, logger=None)
        self._log_clone_request(self.MODEL_NAME, 0, 1, str(sample.speaker_id), reference_path,
                                target_text, prompt_transcript=prompt_text)
        started = time.perf_counter()
        waveform, sample_rate = self._synthesizer.generate(
            text=target_text, prompt_audio=reference_path, prompt_text=prompt_text, seed=seed)
        elapsed = time.perf_counter() - started
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError('GLM-TTS returned invalid mono audio')
        path = self._speaker_output_dir(Path(output_dir).resolve(), str(sample.speaker_id)) / self._cloned_filename(sample, sample.index)
        sf.write(path, waveform, sample_rate)
        return path, elapsed

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)
        samples = dataset.get_zero_shot_samples(max_samples=self.max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for GLM-TTS adversary.")
        self._log_attack_plan(self.MODEL_NAME, samples, self._count_available_prompts(samples))
        completed = 0
        try:
            for sample in samples:
                path, elapsed = self.generate_sample(sample, output_dir=output_dir)
                self._record_synthesis_timing(path, elapsed)
                completed += 1
        finally:
            self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, completed, len(samples))
