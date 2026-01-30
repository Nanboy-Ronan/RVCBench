from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
# from src.models.bark_voice_clone import (
#     BarkVoiceCloneGenerator,
#     BarkVoiceCloneGeneratorConfig,
# )


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

        generator_config = BarkVoiceCloneGeneratorConfig(
            code_path=self.code_path,
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            prompt_cache_dir=self.prompt_cache_dir,
            hubert_checkpoint=self.hubert_checkpoint,
            hubert_tokenizer=self.hubert_tokenizer,
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
            torchaudio_hubert_bundle=str(
                self.config.get("torchaudio_hubert_bundle", "HUBERT_BASE")
            ),
            torchaudio_hubert_layer=int(
                self.config.get("torchaudio_hubert_layer", -1)
            ),
        )
        self._generator = BarkVoiceCloneGenerator(generator_config, self.device, self.logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._ensure_generator()
        assert self._generator is not None

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                if self.logger is not None:
                    self.logger.warning(
                        "[BarkVC] Invalid max_samples '%s'; processing full dataset.",
                        self.max_samples,
                    )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Bark voice cloning adversary.")

        prompt_count = self._count_available_prompts(samples)

        self._log_attack_plan("BarkVC", samples, prompt_count)

        sample_prompt_texts: Dict[str, str] = {}
        for sample in samples:
            prompt_path = self._resolve_prompt_path(sample)
            if prompt_path is not None:
                sample_prompt_texts[str(prompt_path)] = sample.prompt_text or ""

        self._generator.ensure_model()

        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[BarkVC] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            lookup_key = str(reference_path)
            prompt_transcript = sample_prompt_texts.get(lookup_key, "") or self.default_prompt_text
            utterance_text = (sample.target_text or "").strip()
            if not utterance_text:
                utterance_text = (sample.prompt_text or "").strip()
            if not utterance_text:
                utterance_text = prompt_transcript

            synth_start = time.perf_counter()
            audio_np, sample_rate = self._generator.generate(
                text=utterance_text,
                prompt_audio=reference_path,
                sample_index=idx,
            )
            synth_elapsed = time.perf_counter() - synth_start

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "BarkVC",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                utterance_text,
                prompt_transcript=prompt_transcript,
            )
            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, audio_np, sample_rate)
            self._record_synthesis_timing(output_path, synth_elapsed)

            if self.logger is not None:
                self.logger.debug(
                    "[BarkVC] Generated sample %d for %s using reference %s",
                    idx,
                    speaker_id,
                    reference_path.name,
                )

        self._flush_synthesis_timings()
        if self.logger is not None:
            self.logger.info(
                "[BarkVC] Generated %d utterances using %d prompts.",
                len(samples),
                prompt_count,
            )
