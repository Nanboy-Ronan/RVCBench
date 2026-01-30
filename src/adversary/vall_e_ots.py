from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.valle import VallEGenerator, VallEGeneratorConfig


class VallEZeroShotAdversary(BaseAdversary):
    """Runs the upstream VALL-E repository in zero-shot cloning mode."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        self.checkpoint_path = Path(to_absolute_path(self.config.checkpoint_path)).resolve()
        self.text_extractor = str(self.config.get("text_extractor", "espeak"))
        self.top_k = int(self.config.get("top_k", -100))
        self.temperature = float(self.config.get("temperature", 1.0))
        self.output_sample_rate = int(self.config.get("output_sample_rate", 24000))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")
        self.default_prompt_text = str(
            self.config.get(
                "default_prompt_text",
                "Here is a sample of the desired voice.",
            )
        )

        self._generator: Optional[VallEGenerator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = VallEGeneratorConfig(
            code_path=self.code_path,
            checkpoint_path=self.checkpoint_path,
            text_extractor=self.text_extractor,
            top_k=self.top_k,
            temperature=self.temperature,
            output_sample_rate=self.output_sample_rate,
        )
        self._generator = VallEGenerator(generator_config, self.device, self.logger)

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
                    "[VALL-E] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for VALL-E adversary.")

        sample_prompt_texts: Dict[str, str] = {}
        for sample in samples:
            prompt_path = self._resolve_prompt_path(sample)
            if prompt_path is not None:
                sample_prompt_texts[str(prompt_path.resolve())] = sample.prompt_text or ""

        prompt_count = self._count_available_prompts(samples)

        self._log_attack_plan("VALL-E", samples, prompt_count)

        assert self._generator is not None
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[VALL-E] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            prompt_text = sample_prompt_texts.get(
                str(reference_path.resolve()),
                self.default_prompt_text,
            )

            # Use the ground-truth target text for synthesis; fall back to the prompt/ default only if missing.
            text = (sample.target_text or "").strip()
            if not text:
                text = (sample.prompt_text or "").strip()
            if not text:
                text = prompt_text
            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "VALL-E",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                text,
                prompt_transcript=prompt_text,
            )

            synth_start = time.perf_counter()
            wav, sr = self._generator.generate(
                text=text,
                prompt_audio=reference_path,
                prompt_text=prompt_text,
                sample_index=idx,
            )
            synth_elapsed = time.perf_counter() - synth_start
            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, sr)
            self._record_synthesis_timing(output_path, synth_elapsed)

        self._flush_synthesis_timings()
        self.logger.info("[VALL-E] Generated %d utterances.", len(samples))
