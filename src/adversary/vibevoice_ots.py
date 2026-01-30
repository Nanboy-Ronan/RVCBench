from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import soundfile as sf
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from .base_adversary import BaseAdversary
from src.models.vibevoice import VibeVoiceGenerator, VibeVoiceGeneratorConfig


class VibeVoiceZeroShotAdversary(BaseAdversary):
    """Runs the community VibeVoice inference stack for zero-shot cloning."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.get("code_path", "checkpoints/VibeVoice"))).resolve()
        self.model_path = self._maybe_resolve_resource(self.config.get("model_path", "vibevoice/VibeVoice-1.5B"))
        self.checkpoint_path = self._maybe_resolve_path(self.config.get("checkpoint_path"))

        self.cfg_scale = float(self.config.get("cfg_scale", 1.3))
        self.disable_prefill = bool(self.config.get("disable_prefill", False))
        self.torch_dtype = str(self.config.get("torch_dtype", "auto"))
        self.attn_implementation = str(self.config.get("attn_implementation", "flash_attention_2"))
        self.allow_attn_fallback = bool(self.config.get("allow_attn_fallback", True))
        self.num_inference_steps = self.config.get("num_inference_steps", 10)
        self.max_new_tokens = self.config.get("max_new_tokens" )
        self.seed = self.config.get("seed")
        self.verbose = bool(self.config.get("verbose", False))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")

        raw_generation = self.config.get("generation")
        if raw_generation is None:
            self.generation_kwargs = {}
        elif isinstance(raw_generation, dict):
            self.generation_kwargs = dict(raw_generation)
        else:
            self.generation_kwargs = dict(OmegaConf.to_container(raw_generation, resolve=True))

        self._generator: Optional[VibeVoiceGenerator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _maybe_resolve_path(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        candidate = Path(str(value)).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        abs_candidate = Path(to_absolute_path(str(value))).expanduser()
        if abs_candidate.exists():
            return str(abs_candidate.resolve())
        return str(value)

    def _maybe_resolve_resource(self, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("VibeVoice adversary requires a model_path entry.")
        candidate = Path(str(value)).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        abs_candidate = Path(to_absolute_path(str(value))).expanduser()
        if abs_candidate.exists():
            return str(abs_candidate.resolve())
        return str(value)

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        generator_config = VibeVoiceGeneratorConfig(
            code_path=self.code_path,
            model_path=self.model_path,
            checkpoint_path=self.checkpoint_path,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            allow_attn_fallback=self.allow_attn_fallback,
            cfg_scale=self.cfg_scale,
            disable_prefill=self.disable_prefill,
            num_inference_steps=self.num_inference_steps,
            max_new_tokens=self.max_new_tokens,
            generation_kwargs=self.generation_kwargs,
            seed=self.seed,
            verbose=self.verbose,
        )
        self._generator = VibeVoiceGenerator(generator_config, self.device, self.logger)

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
                    "[VibeVoice] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for VibeVoice adversary.")

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan("VibeVoice", samples, prompt_count)

        assert self._generator is not None
        completed = 0
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                self.logger.warning(
                    "[VibeVoice] Sample %d missing prompt audio; skipping.",
                    idx,
                )
                continue

            desired_text = (sample.target_text or "").strip()
            if not desired_text:
                desired_text = (sample.prompt_text or "").strip()
            if not desired_text:
                self.logger.warning(
                    "[VibeVoice] Sample %d has no target text; skipping.",
                    idx,
                )
                continue

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "VibeVoice",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                desired_text,
                prompt_transcript=sample.prompt_text,
            )

            try:
                # Format the desired_text to match the expected "Speaker X: text" format
                formatted_text = f"Speaker 1: {desired_text}"
                synth_start = time.perf_counter()
                waveform, sr = self._generator.generate(formatted_text, reference_path)
                synth_elapsed = time.perf_counter() - synth_start
            except Exception as exc:
                self.logger.error(
                    "[VibeVoice] Generation failed for sample %d (speaker %s): %s",
                    idx,
                    speaker_id,
                    exc,
                )
                continue

            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, waveform, sr)
            self._record_synthesis_timing(output_path, synth_elapsed)
            completed += 1

        self._flush_synthesis_timings()
        self.logger.info("[VibeVoice] Generated %d/%d utterances.", completed, len(samples))
