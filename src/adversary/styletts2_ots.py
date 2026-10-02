from pathlib import Path
import time
from typing import Optional

import numpy as np

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.styletts2 import StyleTTS2Synthesizer, StyleTTS2SynthesizerConfig


class StyleTTS2ZeroShotAdversary(BaseAdversary):
    """Runs StyleTTS2 LibriTTS demo pipeline for zero-shot cloning."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        self.config_path = Path(to_absolute_path(self.config.config_path)).resolve()
        self.checkpoint_path = Path(to_absolute_path(self.config.checkpoint_path)).resolve()

        self.alpha = float(self.config.get("alpha", 0.3))
        self.beta = float(self.config.get("beta", 0.7))
        self.diffusion_steps = int(self.config.get("diffusion_steps", 5))
        self.embedding_scale = float(self.config.get("embedding_scale", 1.0))
        self.sample_rate = int(self.config.get("sample_rate", 24000))
        self.tail_trim = int(self.config.get("tail_trim", 50))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")

        self._synthesizer: Optional[StyleTTS2Synthesizer] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_synthesizer(self) -> None:
        if self._synthesizer is not None:
            return

        synth_config = StyleTTS2SynthesizerConfig(
            code_path=self.code_path,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            alpha=self.alpha,
            beta=self.beta,
            diffusion_steps=self.diffusion_steps,
            embedding_scale=self.embedding_scale,
            sample_rate=self.sample_rate,
            tail_trim=self.tail_trim,
        )
        self._synthesizer = StyleTTS2Synthesizer(synth_config, self.device, self.logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_sample(self, sample, *, output_dir):
        self._ensure_synthesizer()
        reference_path = self._resolve_prompt_path(sample)
        if reference_path is None:
            raise FileNotFoundError(f"Missing StyleTTS2 reference: {sample.prompt_path}")
        text = (sample.target_text or "").strip() or (sample.prompt_text or "").strip()
        if not text:
            raise ValueError("StyleTTS2 target text cannot be empty")
        self._log_clone_request("StyleTTS2", 0, 1, str(sample.speaker_id), reference_path, text)
        seed = self._sample_seed(sample)
        if seed is not None:
            from src.utils.seeding import configure_seeds
            configure_seeds(seed, logger=None)
        started = time.perf_counter()
        wav = self._synthesizer.synthesize(text, reference_path)
        elapsed = time.perf_counter() - started
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
            raise ValueError("StyleTTS2 output must be nonempty finite mono audio")
        path = self._speaker_output_dir(Path(output_dir).resolve(), str(sample.speaker_id)) / self._cloned_filename(sample, sample.index)
        sf.write(path, wav, self.sample_rate)
        return path, elapsed

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path
        self._ensure_synthesizer()
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)
        samples = dataset.get_zero_shot_samples(max_samples=self.max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for StyleTTS2 adversary.")
        self._log_attack_plan("StyleTTS2", samples, self._count_available_prompts(samples))
        try:
            for sample in samples:
                path, elapsed = self.generate_sample(sample, output_dir=output_dir)
                self._record_synthesis_timing(path, elapsed)
        finally:
            self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[StyleTTS2] Generated %d/%d utterances.", len(samples), len(samples))
