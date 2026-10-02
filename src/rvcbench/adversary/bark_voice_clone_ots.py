from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary


class BarkVoiceCloneZeroShotAdversary(BaseAdversary):
    """Runs the Bark voice cloning pipeline for zero-shot attacks."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        self.models_dir = self._resolve_optional_path(self.config.get("models_dir"))
        self.cache_dir = self._resolve_optional_path(self.config.get("cache_dir"))
        self.prompt_cache_dir = self._resolve_optional_path(self.config.get("prompt_cache_dir"))
        self.hubert_checkpoint = self._resolve_optional_path(self.config.get("hubert_checkpoint"))
        self.hubert_tokenizer = self._resolve_optional_path(self.config.get("hubert_tokenizer"))

        self.reference_assignment = str(
            self.config.get("reference_assignment", "round_robin")
        ).lower().strip()
        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")
        self.default_prompt_text = str(
            self.config.get(
                "default_prompt_text",
                "Here is a short sample of the desired voice.",
            )
        )

        self._generator: Optional[BarkVoiceCloneGenerator] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_optional_path(self, value) -> Optional[Path]:
        if value in (None, ""):
            return None
        resolved = Path(to_absolute_path(str(value)))
        return resolved.resolve()

    def _coerce_optional(self, key: str, caster):
        value = self.config.get(key)
        if value in (None, ""):
            return None
        try:
            return caster(value)
        except (TypeError, ValueError):
            if self.logger is not None:
                self.logger.warning(
                    "[BarkVC] Failed to cast '%s' value '%s'; ignoring.",
                    key,
                    value,
                )
            return None

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        from rvcbench.models.bark_voice_clone import BarkVoiceCloneGenerator, BarkVoiceCloneGeneratorConfig

        generator_config = BarkVoiceCloneGeneratorConfig(
            code_path=self.code_path,
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            prompt_cache_dir=self.prompt_cache_dir,
            hubert_checkpoint=self.hubert_checkpoint,
            hubert_tokenizer=self.hubert_tokenizer,
            text_tokenizer_path=self._resolve_optional_path(self.config.get('text_tokenizer_path')),
            text_temperature=float(self.config.get("text_temperature", 0.7)),
            text_top_k=self._coerce_optional("text_top_k", int),
            text_top_p=self._coerce_optional("text_top_p", float),
            coarse_temperature=float(self.config.get("coarse_temperature", 0.7)),
            coarse_top_k=self._coerce_optional("coarse_top_k", int),
            coarse_top_p=self._coerce_optional("coarse_top_p", float),
            fine_temperature=float(self.config.get("fine_temperature", 0.5)),
            semantic_use_kv_cache=bool(self.config.get("semantic_use_kv_cache", True)),
            coarse_use_kv_cache=bool(self.config.get("coarse_use_kv_cache", True)),
            silent=bool(self.config.get("silent", True)),
            force_reload_models=bool(self.config.get("force_reload_models", False)),
            max_prompt_seconds=self._coerce_optional("max_prompt_seconds", float),
            hubert_layer=int(self.config.get('hubert_layer', 9)),
        )
        self._generator = BarkVoiceCloneGenerator(generator_config, self.device, self.logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_sample(self, sample, *, output_dir):
        reference_path = self._resolve_prompt_path(sample)
        if reference_path is None:
            raise FileNotFoundError(f"Missing Bark reference audio: {sample.prompt_path}")
        text = (sample.target_text or "").strip()
        if not text:
            raise ValueError('Bark requires nonempty target text')
        self._ensure_generator()
        self._generator.ensure_model()
        seed = self._sample_seed(sample)
        if seed is not None:
            from rvcbench.utils.seeding import configure_seeds
            configure_seeds(seed, logger=None)
        self._log_clone_request("BarkVC", 0, 1, str(sample.speaker_id), reference_path,
                                text, prompt_transcript=sample.prompt_text or self.default_prompt_text)
        started = time.perf_counter()
        audio, sample_rate = self._generator.generate(
            text=text, prompt_audio=reference_path, sample_index=sample.index)
        elapsed = time.perf_counter() - started
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError('Bark returned invalid mono audio')
        path = self._speaker_output_dir(Path(output_dir).resolve(), str(sample.speaker_id)) / self._cloned_filename(sample, sample.index)
        sf.write(path, audio, sample_rate)
        return path, elapsed

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)
        samples = dataset.get_zero_shot_samples(max_samples=self.max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Bark voice cloning adversary.")
        self._log_attack_plan("BarkVC", samples, self._count_available_prompts(samples))
        completed = 0
        try:
            for sample in samples:
                path, elapsed = self.generate_sample(sample, output_dir=output_dir)
                self._record_synthesis_timing(path, elapsed)
                completed += 1
        finally:
            self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[BarkVC] Generated %d/%d utterances.", completed, len(samples))
