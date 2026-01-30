from pathlib import Path
import time
from typing import List, Optional, Sequence

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
    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._ensure_synthesizer()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[StyleTTS2] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for StyleTTS2 adversary.")

        prompt_count = self._count_available_prompts(samples)

        self._log_attack_plan("StyleTTS2", samples, prompt_count)

        assert self._synthesizer is not None
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[StyleTTS2] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            self.logger.debug(
                "[StyleTTS2] Generating sample %d using reference %s for speaker %s",
                idx,
                reference_path.name,
                sample.speaker_id,
            )

            # Prefer the target transcript for generation; fall back to prompt text if missing.
            text = (sample.target_text or "").strip()
            if not text:
                text = (sample.prompt_text or "").strip()
            synth_start = time.perf_counter()
            wav = self._synthesizer.synthesize(text, reference_path)
            synth_elapsed = time.perf_counter() - synth_start
            if wav.size == 0:
                self.logger.warning(
                    "[StyleTTS2] Empty waveform for entry %s; skipping.", text[:30]
                )
                continue

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            self._log_clone_request(
                "StyleTTS2",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                text,
            )
            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, self.sample_rate)
            self._record_synthesis_timing(output_path, synth_elapsed)

        self._flush_synthesis_timings()
        self.logger.info("[StyleTTS2] Generated %d utterances.", len(samples))
