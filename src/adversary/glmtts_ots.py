"""GLM-TTS zero-shot adversary integration."""
from __future__ import annotations

import random
from pathlib import Path
import time
from typing import Optional

import soundfile as sf
import torch
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
        self.default_prompt_text = str(
            self.config.get(
                "default_prompt_text",
                "Here is a sample of the desired voice.",
            )
        )

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

    def _set_seed(self, index: int) -> Optional[int]:
        if self.seed is None:
            return None
        adjusted_seed = int(self.seed) + int(index)
        random.seed(adjusted_seed)
        torch.manual_seed(adjusted_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(adjusted_seed)
        return adjusted_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path

        self._ensure_synthesizer()
        assert self._synthesizer is not None

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[%s] Invalid max_samples '%s'; processing full dataset.",
                    self.MODEL_NAME,
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for GLM-TTS adversary.")

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
                target_text = self.default_prompt_text
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

            seed = self._set_seed(idx)
            try:
                synth_start = time.perf_counter()
                waveform, sample_rate = self._synthesizer.generate(
                    text=target_text,
                    prompt_audio=reference_path,
                    prompt_text=prompt_text,
                    seed=seed,
                )
                synth_elapsed = time.perf_counter() - synth_start
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.MODEL_NAME}] Generation failed for sample {idx} ({speaker_id})."
                ) from exc

            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, waveform, sample_rate)
            self._record_synthesis_timing(output_path, synth_elapsed)
            completed += 1

        self._flush_synthesis_timings()
        self.logger.info(
            "[%s] Generated %d/%d utterances.",
            self.MODEL_NAME,
            completed,
            len(samples),
        )
