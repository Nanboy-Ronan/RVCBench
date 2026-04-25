"""MaskGCT zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, Optional

from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.maskgct import MaskGCTGenerator, MaskGCTGeneratorConfig


class MaskGCTZeroShotAdversary(BaseAdversary):
    """Runs Amphion MaskGCT for zero-shot voice cloning."""

    MODEL_NAME = "MaskGCT"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(
            to_absolute_path(self.config.get("code_path", "checkpoints/Amphion-maskgct"))
        ).resolve()
        self.config_path = Path(
            to_absolute_path(
                self.config.get(
                    "config_path",
                    "checkpoints/Amphion-maskgct/models/tts/maskgct/config/maskgct.json",
                )
            )
        ).resolve()
        runtime_python_value = self.config.get("runtime_python")
        self.runtime_python = (
            Path(to_absolute_path(str(runtime_python_value))).resolve()
            if runtime_python_value not in (None, "")
            else None
        )
        worker_script_value = self.config.get("worker_script_path", "scripts/maskgct_worker.py")
        self.worker_script_path = Path(to_absolute_path(str(worker_script_value))).resolve()
        self.repo_id = str(self.config.get("repo_id", "amphion/MaskGCT"))
        self.max_samples = self.config.get("max_samples")
        self.default_prompt_text = str(
            self.config.get("default_prompt_text", "Here is a sample of the desired voice.")
        ).strip()
        self.default_language = str(self.config.get("default_language", "en")).strip() or "en"
        self.target_len = self.config.get("target_len")
        self.verbose = bool(self.config.get("verbose", False))

        generation_kwargs: Dict[str, object] = {}
        for key in (
            "n_timesteps",
            "cfg",
            "rescale_cfg",
            "n_timesteps_s2a",
            "cfg_s2a",
            "rescale_cfg_s2a",
        ):
            value = self.config.get(key)
            if value is not None:
                generation_kwargs[key] = value

        self._generator: Optional[MaskGCTGenerator] = None
        self._generation_kwargs = generation_kwargs

    def _normalise_language(self, value: Optional[str]) -> str:
        token = str(value or "").strip().lower()
        if not token:
            return self.default_language
        mapping = {
            "en-us": "en",
            "en-gb": "en",
            "english": "en",
            "zh-cn": "zh",
            "zh": "zh",
            "chinese": "zh",
            "ja-jp": "ja",
            "jp": "ja",
            "japanese": "ja",
            "ko-kr": "ko",
            "korean": "ko",
            "fr-fr": "fr",
            "french": "fr",
            "de-de": "de",
            "german": "de",
        }
        return mapping.get(token, token)

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        generator_config = MaskGCTGeneratorConfig(
            code_path=self.code_path,
            config_path=self.config_path,
            runtime_python=self.runtime_python,
            worker_script_path=self.worker_script_path,
            repo_id=self.repo_id,
            verbose=self.verbose,
            generation_kwargs=dict(self._generation_kwargs),
        )
        self._generator = MaskGCTGenerator(generator_config, self.device, self.logger)

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

            prompt_text = (sample.prompt_text or "").strip() or self.default_prompt_text
            target_text = (sample.target_text or "").strip() or prompt_text
            prompt_language = self._normalise_language(sample.prompt_language)
            target_language = self._normalise_language(sample.target_language) or prompt_language

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            output_filename = self._cloned_filename(sample, idx)
            output_wav_path = speaker_dir / output_filename

            self._log_clone_request(
                self.MODEL_NAME,
                idx,
                len(samples),
                speaker_id,
                reference_path,
                target_text,
                prompt_transcript=prompt_text,
            )

            try:
                synth_start = time.perf_counter()
                self._generator.generate(
                    prompt_speech_path=reference_path,
                    prompt_text=prompt_text,
                    target_text=target_text,
                    output_path=output_wav_path,
                    prompt_language=prompt_language,
                    target_language=target_language,
                    target_len=self.target_len,
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

            self._record_synthesis_timing(output_wav_path, synth_elapsed)
            completed += 1

        self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, completed, len(samples))
