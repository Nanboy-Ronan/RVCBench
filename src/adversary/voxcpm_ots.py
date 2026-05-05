"""VoxCPM zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.voxcpm import VoxCPMGenerator, VoxCPMGeneratorConfig


class VoxCPMZeroShotAdversary(BaseAdversary):
    """Runs VoxCPM/VoxCPM2 for zero-shot voice cloning."""

    MODEL_NAME = "VoxCPM"
    DEFAULT_MODEL_PATH = "openbmb/VoxCPM2"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.model_path = self._resolve_model_path(self.config.get("model_path", self.DEFAULT_MODEL_PATH))
        self.code_path = self._resolve_optional_path(self.config.get("code_path"))
        self.cache_dir = self._resolve_optional_path(self.config.get("cache_dir"))
        self.local_files_only = bool(self.config.get("local_files_only", False))
        self.optimize = bool(self.config.get("optimize", False))
        self.load_denoiser = bool(self.config.get("load_denoiser", False))
        self.zipenhancer_model_id = str(
            self.config.get("zipenhancer_model_id", "iic/speech_zipenhancer_ans_multiloss_16k_base")
        )
        self.max_samples = self.config.get("max_samples")
        self.default_prompt_text = str(
            self.config.get("default_prompt_text", "Here is a sample of the desired voice.")
        ).strip()
        self.use_prompt_text = bool(self.config.get("use_prompt_text", True))
        self.include_reference_audio = bool(self.config.get("include_reference_audio", True))

        self._generator: Optional[VoxCPMGenerator] = None

    def _resolve_optional_path(self, value) -> Optional[str]:
        if value in (None, ""):
            return None
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

    def _resolve_model_path(self, value) -> str:
        resolved = self._resolve_optional_path(value)
        return resolved or self.DEFAULT_MODEL_PATH

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = VoxCPMGeneratorConfig(
            model_path=self.model_path,
            code_path=self.code_path,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            optimize=self.optimize,
            load_denoiser=self.load_denoiser,
            zipenhancer_model_id=self.zipenhancer_model_id,
            cfg_value=float(self.config.get("cfg_value", 2.0)),
            inference_timesteps=int(self.config.get("inference_timesteps", 10)),
            min_len=int(self.config.get("min_len", 2)),
            max_len=int(self.config.get("max_len", 4096)),
            normalize=bool(self.config.get("normalize", False)),
            denoise=bool(self.config.get("denoise", False)),
            retry_badcase=bool(self.config.get("retry_badcase", True)),
            retry_badcase_max_times=int(self.config.get("retry_badcase_max_times", 3)),
            retry_badcase_ratio_threshold=float(self.config.get("retry_badcase_ratio_threshold", 6.0)),
        )
        self._generator = VoxCPMGenerator(generator_config, self.device, self.logger)

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

            generation_kwargs = {
                "text": target_text,
                "reference_wav_path": str(reference_path.resolve()) if self.include_reference_audio else None,
            }
            if self.use_prompt_text and prompt_text:
                generation_kwargs["prompt_wav_path"] = str(reference_path.resolve())
                generation_kwargs["prompt_text"] = prompt_text

            try:
                synth_start = time.perf_counter()
                wav, sample_rate = self._generator.generate(**generation_kwargs)
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
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, completed, len(samples))
