"""Qwen3-TTS zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.qwen3_tts import Qwen3TTSGenerator, Qwen3TTSGeneratorConfig


_LANGUAGE_MAP = {
    "auto": "Auto",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
}


class Qwen3TTSZeroShotAdversary(BaseAdversary):
    """Runs Qwen3-TTS base checkpoints for zero-shot voice cloning."""

    MODEL_NAME = "Qwen3-TTS"
    DEFAULT_CHECKPOINT_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        checkpoint_value = self.config.get("checkpoint_path", self.DEFAULT_CHECKPOINT_PATH)
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_value)
        configured_device_map = self.config.get("device_map")
        if configured_device_map in (None, "", "auto"):
            self.device_map = str(self.device)
        else:
            self.device_map = configured_device_map
        self.torch_dtype = self.config.get("torch_dtype", "auto")
        self.use_flash_attn2 = bool(self.config.get("use_flash_attn2", False))
        self.attn_implementation = self.config.get("attn_implementation")
        self.seed = self.config.get("seed")
        self.max_samples = self.config.get("max_samples")

        self.default_language = self._normalise_language(self.config.get("language", "Auto"))
        self.language_source = str(self.config.get("language_source", "target")).strip().lower()
        self.default_prompt_text = str(
            self.config.get("default_prompt_text", "Here is a sample of the desired voice.")
        ).strip()
        self.x_vector_only_mode = bool(self.config.get("x_vector_only_mode", False))
        self.fallback_to_x_vector_only_mode_on_missing_ref_text = bool(
            self.config.get("fallback_to_x_vector_only_mode_on_missing_ref_text", True)
        )
        self.reuse_voice_clone_prompt = bool(self.config.get("reuse_voice_clone_prompt", True))

        generation_kwargs = {}
        for key in (
            "max_new_tokens",
            "top_k",
            "top_p",
            "temperature",
            "do_sample",
            "repetition_penalty",
        ):
            value = self.config.get(key)
            if value is not None:
                generation_kwargs[key] = value

        self._generator: Optional[Qwen3TTSGenerator] = None
        self._prompt_cache: Dict[Tuple[str, str, bool], Any] = {}
        self._generation_kwargs = generation_kwargs

    def _resolve_checkpoint_path(self, value) -> str:
        raw = str(value)
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        try:
            absolute_candidate = Path(to_absolute_path(raw))
        except Exception:
            absolute_candidate = candidate
        if absolute_candidate.exists():
            return str(absolute_candidate.resolve())
        return raw

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = Qwen3TTSGeneratorConfig(
            checkpoint_path=self.checkpoint_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            use_flash_attn2=self.use_flash_attn2,
            attn_implementation=self.attn_implementation,
            seed=self.seed,
            generation_kwargs=dict(self._generation_kwargs),
        )
        self._generator = Qwen3TTSGenerator(generator_config, self.device, self.logger)

    def _normalise_language(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        token = str(value).strip()
        if not token:
            return None
        mapped = _LANGUAGE_MAP.get(token.lower())
        if mapped is not None:
            return mapped
        return token

    def _select_language(self, sample) -> Optional[str]:
        choices = {
            "target": sample.target_language,
            "prompt": sample.prompt_language,
        }

        raw = choices.get(self.language_source)
        if raw:
            return self._normalise_language(raw)

        if sample.target_language:
            return self._normalise_language(sample.target_language)
        if sample.prompt_language:
            return self._normalise_language(sample.prompt_language)
        return self.default_language

    def _resolve_prompt_features(
        self,
        reference_path: Path,
        prompt_text: str,
        use_x_vector_only_mode: bool,
    ) -> Any:
        assert self._generator is not None

        if not self.reuse_voice_clone_prompt:
            return None

        cache_key = (
            str(reference_path.resolve()),
            prompt_text,
            bool(use_x_vector_only_mode),
        )
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        prompt = self._generator.create_voice_clone_prompt(
            ref_audio=str(reference_path.resolve()),
            ref_text=prompt_text if prompt_text else None,
            x_vector_only_mode=use_x_vector_only_mode,
        )
        self._prompt_cache[cache_key] = prompt
        return prompt

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path

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
                if self.logger:
                    self.logger.warning(
                        "[%s] Invalid max_samples '%s'; processing full dataset.",
                        self.MODEL_NAME,
                        self.max_samples,
                    )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError(f"No zero-shot samples available for {self.MODEL_NAME} adversary.")

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan(self.MODEL_NAME, samples, prompt_count)

        completed = 0
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[%s] Sample %d missing prompt audio; skipping.",
                        self.MODEL_NAME,
                        idx,
                    )
                continue

            target_text = (sample.target_text or "").strip()
            prompt_text = (sample.prompt_text or "").strip()
            if not target_text:
                target_text = prompt_text or self.default_prompt_text

            use_x_vector_only_mode = self.x_vector_only_mode
            if not prompt_text and not use_x_vector_only_mode:
                if self.fallback_to_x_vector_only_mode_on_missing_ref_text:
                    use_x_vector_only_mode = True
                    if self.logger:
                        self.logger.warning(
                            "[%s] Sample %d has no prompt transcript; falling back to x_vector_only_mode.",
                            self.MODEL_NAME,
                            idx,
                        )
                else:
                    if self.logger:
                        self.logger.warning(
                            "[%s] Sample %d has no prompt transcript and x_vector_only_mode is disabled; skipping.",
                            self.MODEL_NAME,
                            idx,
                        )
                    continue

            language = self._select_language(sample)
            speaker_id = str(sample.speaker_id)

            self._log_clone_request(
                self.MODEL_NAME,
                idx,
                len(samples),
                speaker_id,
                reference_path,
                target_text,
                prompt_transcript=prompt_text,
            )

            prompt_features = self._resolve_prompt_features(
                reference_path,
                prompt_text,
                use_x_vector_only_mode,
            )

            try:
                synth_start = time.perf_counter()
                wav, sample_rate = self._generator.generate(
                    text=target_text,
                    language=language,
                    ref_audio=str(reference_path.resolve()),
                    ref_text=prompt_text if prompt_text else None,
                    voice_clone_prompt=prompt_features,
                    x_vector_only_mode=use_x_vector_only_mode,
                    sample_index=idx,
                )
                synth_elapsed = time.perf_counter() - synth_start
            except Exception as exc:
                if self.logger:
                    self.logger.error(
                        "[%s] Generation failed for sample %d (%s): %s",
                        self.MODEL_NAME,
                        idx,
                        speaker_id,
                        exc,
                    )
                continue

            wav = np.asarray(wav, dtype=np.float32)
            if not np.isfinite(wav).all():
                wav = np.nan_to_num(wav)
            wav = np.clip(wav, -1.0, 1.0)

            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            output_filename = self._cloned_filename(sample, idx)
            resolved_output_path = speaker_dir / output_filename
            sf.write(str(resolved_output_path), wav, sample_rate)
            self._record_synthesis_timing(resolved_output_path, synth_elapsed)
            completed += 1

        self._flush_synthesis_timings()
        if self.logger:
            self.logger.info(
                "[%s] Generated %d/%d utterances.",
                self.MODEL_NAME,
                completed,
                len(samples),
            )
