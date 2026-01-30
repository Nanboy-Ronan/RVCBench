"""CosyVoice zero-shot adversary integration."""

from __future__ import annotations

import random
from pathlib import Path
import time
from typing import Optional

import soundfile as sf
import torch
import torchaudio
from torchaudio import functional as taF
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.cosyvoice import CosyVoiceGenerator, CosyVoiceGeneratorConfig


class CosyVoiceZeroShotAdversary(BaseAdversary):
    """Runs CosyVoice/CosyVoice2 for zero-shot voice cloning."""

    MODEL_NAME = "CosyVoice"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.get("code_path", "checkpoints/CosyVoice"))).resolve()
        model_dir_value = self.config.get("model_dir")
        if not model_dir_value:
            raise ValueError("CosyVoice adversary requires 'model_dir' in the config block.")
        self.model_dir = Path(to_absolute_path(model_dir_value)).resolve()

        self.variant = str(self.config.get("variant", "cosyvoice2")).strip()
        self.stream = bool(self.config.get("stream", False))
        self.speed = float(self.config.get("speed", 1.0))
        self.prompt_sample_rate = int(self.config.get("prompt_sample_rate", 16000))
        self.text_frontend = bool(self.config.get("text_frontend", False))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")
        self.default_prompt_text = str(
            self.config.get(
                "default_prompt_text",
                "Here is a sample of the desired voice.",
            )
        )
        self.seed = self.config.get("seed")
        self.zero_shot_speaker_id = str(self.config.get("zero_shot_spk_id", "") or "")
        self.load_jit = bool(self.config.get("load_jit", False))
        self.load_trt = bool(self.config.get("load_trt", False))
        self.load_vllm = bool(self.config.get("load_vllm", False))
        self.fp16 = bool(self.config.get("fp16", False))
        self.trt_concurrent = int(self.config.get("trt_concurrent", 1))

        self._generator: Optional[CosyVoiceGenerator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        generator_config = CosyVoiceGeneratorConfig(
            code_path=self.code_path,
            model_dir=self.model_dir,
            variant=self.variant,
            stream=self.stream,
            speed=self.speed,
            text_frontend=self.text_frontend,
            load_jit=self.load_jit,
            load_trt=self.load_trt,
            load_vllm=self.load_vllm,
            fp16=self.fp16,
            trt_concurrent=self.trt_concurrent,
        )
        self._generator = CosyVoiceGenerator(generator_config, self.device, self.logger)

    def _load_prompt_audio(self, reference_path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(reference_path))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(torch.float32)
        if sample_rate != self.prompt_sample_rate:
            waveform = taF.resample(waveform, sample_rate, self.prompt_sample_rate)
        return waveform

    def _set_seed(self, index: int) -> None:
        if self.seed is None:
            return
        adjusted_seed = int(self.seed) + int(index)
        random.seed(adjusted_seed)
        torch.manual_seed(adjusted_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(adjusted_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
                self.logger.warning(
                    "[%s] Sample %d missing prompt audio; skipping.",
                    self.MODEL_NAME,
                    idx,
                )
                continue

            try:
                prompt_waveform = self._load_prompt_audio(reference_path)
            except Exception as exc:
                self.logger.warning(
                    "[%s] Failed to load prompt %s: %s",
                    self.MODEL_NAME,
                    reference_path,
                    exc,
                )
                continue

            target_text = (sample.target_text or "").strip()
            prompt_text = (sample.prompt_text or "").strip()
            if not target_text:
                target_text = prompt_text
            if not target_text:
                target_text = self.default_prompt_text
            prompt_text_for_model = prompt_text or self.default_prompt_text

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

            self._set_seed(idx)
            try:
                synth_start = time.perf_counter()
                waveform, sample_rate = self._generator.generate(
                    text=target_text,
                    prompt_audio_16k=prompt_waveform,
                    prompt_text=prompt_text_for_model,
                    zero_shot_speaker_id=self.zero_shot_speaker_id,
                )
                synth_elapsed = time.perf_counter() - synth_start
            except Exception as exc:
                self.logger.error(
                    "[%s] Generation failed for sample %d (%s): %s",
                    self.MODEL_NAME,
                    idx,
                    speaker_id,
                    exc,
                )
                continue

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
