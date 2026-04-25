"""XTTS-v2 zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.xtts import XttsGenerator, XttsGeneratorConfig


_LANGUAGE_MAP = {
    "ar": "ar",
    "arabic": "ar",
    "cs": "cs",
    "czech": "cs",
    "de": "de",
    "german": "de",
    "en": "en",
    "english": "en",
    "es": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
    "hi": "hi",
    "hindi": "hi",
    "hu": "hu",
    "hungarian": "hu",
    "it": "it",
    "italian": "it",
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kr": "ko",
    "korean": "ko",
    "nl": "nl",
    "dutch": "nl",
    "pl": "pl",
    "polish": "pl",
    "pt": "pt",
    "portuguese": "pt",
    "pt-br": "pt",
    "ru": "ru",
    "russian": "ru",
    "tr": "tr",
    "turkish": "tr",
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zh-tw": "zh-cn",
    "cn": "zh-cn",
    "chinese": "zh-cn",
}


class XttsZeroShotAdversary(BaseAdversary):
    """Runs Coqui XTTS-v2 for zero-shot voice cloning."""

    MODEL_NAME = "XTTS-v2"
    DEFAULT_CHECKPOINT = "coqui/XTTS-v2"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        cache_dir_value = self.config.get("cache_dir")
        self.cache_dir = (
            Path(to_absolute_path(str(cache_dir_value))).resolve()
            if cache_dir_value not in (None, "")
            else None
        )
        checkpoint_path_value = self.config.get("checkpoint_path")
        self.checkpoint_path = (
            Path(to_absolute_path(str(checkpoint_path_value))).resolve()
            if checkpoint_path_value not in (None, "")
            else None
        )
        config_path_value = self.config.get("config_path")
        self.config_path = (
            Path(to_absolute_path(str(config_path_value))).resolve()
            if config_path_value not in (None, "")
            else None
        )
        vocab_path_value = self.config.get("vocab_path")
        self.vocab_path = (
            Path(to_absolute_path(str(vocab_path_value))).resolve()
            if vocab_path_value not in (None, "")
            else None
        )
        speaker_file_value = self.config.get("speaker_file_path")
        self.speaker_file_path = (
            Path(to_absolute_path(str(speaker_file_value))).resolve()
            if speaker_file_value not in (None, "")
            else None
        )

        checkpoint_value = self.config.get("checkpoint", self.DEFAULT_CHECKPOINT)
        self.checkpoint = self._resolve_checkpoint_value(checkpoint_value)
        self.use_deepspeed = bool(self.config.get("use_deepspeed", False))
        self.strict_checkpoint = bool(self.config.get("strict_checkpoint", True))
        self.seed = self.config.get("seed")
        self.max_samples = self.config.get("max_samples")
        self.default_language = self._normalise_language(self.config.get("language", "en")) or "en"
        self.language_source = str(self.config.get("language_source", "target")).strip().lower()
        self.default_prompt_text = str(
            self.config.get("default_prompt_text", "Here is a sample of the desired voice.")
        ).strip()

        self.temperature = float(self.config.get("temperature", 0.75))
        self.length_penalty = float(self.config.get("length_penalty", 1.0))
        self.repetition_penalty = float(self.config.get("repetition_penalty", 10.0))
        self.top_k = int(self.config.get("top_k", 50))
        self.top_p = float(self.config.get("top_p", 0.85))
        self.do_sample = bool(self.config.get("do_sample", True))
        self.num_beams = int(self.config.get("num_beams", 1))
        self.speed = float(self.config.get("speed", 1.0))
        self.enable_text_splitting = bool(self.config.get("enable_text_splitting", False))
        self.gpt_cond_len = int(self.config.get("gpt_cond_len", 30))
        self.gpt_cond_chunk_len = int(self.config.get("gpt_cond_chunk_len", 6))
        self.max_ref_len = int(self.config.get("max_ref_len", 10))
        self.sound_norm_refs = bool(self.config.get("sound_norm_refs", False))

        self._generator: Optional[XttsGenerator] = None

    def _resolve_checkpoint_value(self, value) -> str:
        raw = str(value).strip()
        if not raw:
            return self.DEFAULT_CHECKPOINT
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

    def _normalise_language(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        token = str(value).strip()
        if not token:
            return None
        lowered = token.lower()
        if lowered in _LANGUAGE_MAP:
            return _LANGUAGE_MAP[lowered]
        return token

    def _select_language(self, sample) -> str:
        choices = {
            "target": sample.target_language,
            "prompt": sample.prompt_language,
        }
        raw = choices.get(self.language_source)
        if raw:
            normalised = self._normalise_language(raw)
            if normalised:
                return normalised

        for candidate in (sample.target_language, sample.prompt_language, self.default_language):
            normalised = self._normalise_language(candidate)
            if normalised:
                return normalised
        return "en"

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = XttsGeneratorConfig(
            checkpoint=self.checkpoint,
            cache_dir=self.cache_dir,
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            vocab_path=self.vocab_path,
            speaker_file_path=self.speaker_file_path,
            use_deepspeed=self.use_deepspeed,
            strict_checkpoint=self.strict_checkpoint,
            temperature=self.temperature,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            speed=self.speed,
            enable_text_splitting=self.enable_text_splitting,
            gpt_cond_len=self.gpt_cond_len,
            gpt_cond_chunk_len=self.gpt_cond_chunk_len,
            max_ref_len=self.max_ref_len,
            sound_norm_refs=self.sound_norm_refs,
        )
        self._generator = XttsGenerator(generator_config, self.device, self.logger)

    def _set_seed(self, index: int) -> None:
        if self.seed is None:
            return
        adjusted_seed = int(self.seed) + int(index)
        random.seed(adjusted_seed)
        np.random.seed(adjusted_seed)
        torch.manual_seed(adjusted_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(adjusted_seed)

    def _release_generator(self) -> None:
        generator = self._generator
        if generator is None:
            return
        try:
            model = getattr(generator, "model", None)
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
        finally:
            self._generator = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            if not target_text:
                target_text = (sample.prompt_text or "").strip()
            if not target_text:
                target_text = self.default_prompt_text

            speaker_id = str(sample.speaker_id)
            language = self._select_language(sample)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                self.MODEL_NAME,
                idx,
                len(samples),
                speaker_id,
                reference_path,
                target_text,
                prompt_transcript=sample.prompt_text,
            )

            try:
                self._set_seed(idx)
                synth_start = time.perf_counter()
                wav, sample_rate = self._generator.generate(
                    text=target_text,
                    reference_audio=reference_path,
                    language=language,
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

            output_filename = self._cloned_filename(sample, idx)
            output_wav_path = speaker_dir / output_filename
            sf.write(str(output_wav_path), wav, sample_rate)
            self._record_synthesis_timing(output_wav_path, synth_elapsed)
            completed += 1

        self._flush_synthesis_timings()
        self._release_generator()
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, completed, len(samples))
