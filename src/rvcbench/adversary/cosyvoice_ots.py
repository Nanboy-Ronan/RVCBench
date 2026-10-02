"""CosyVoice zero-shot adversary integration."""

from __future__ import annotations

import random
from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio import functional as taF
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from rvcbench.models.cosyvoice import CosyVoiceGenerator, CosyVoiceGeneratorConfig


class CosyVoiceZeroShotAdversary(BaseAdversary):
    """Runs CosyVoice/CosyVoice2 for zero-shot voice cloning."""

    MODEL_NAME = "CosyVoice"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.get("code_path", "checkpoints/CosyVoice"))).resolve()
        matcha_path = self.config.get('matcha_code_path')
        self.matcha_code_path = Path(to_absolute_path(str(matcha_path))).resolve() if matcha_path else None
        model_dir_value = self.config.get("model_dir")
        if not model_dir_value:
            raise ValueError("CosyVoice adversary requires 'model_dir' in the config block.")
        self.model_dir = Path(to_absolute_path(model_dir_value)).resolve()

        self.variant = str(self.config.get("variant", "cosyvoice2")).strip()
        self.stream = bool(self.config.get("stream", False))
        self.speed = float(self.config.get("speed", 1.0))
        self.prompt_sample_rate = self.config.get("prompt_sample_rate", 16000)
        if isinstance(self.prompt_sample_rate, bool) or not isinstance(self.prompt_sample_rate, int) or self.prompt_sample_rate != 16000:
            raise ValueError("CosyVoice prompt_sample_rate must be the integer 16000")
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
            matcha_code_path=self.matcha_code_path,
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
        if waveform.ndim != 2 or not waveform.numel() or not torch.isfinite(waveform).all():
            raise ValueError("CosyVoice reference must be nonempty finite audio")
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(torch.float32)
        if sample_rate != self.prompt_sample_rate:
            waveform = taF.resample(waveform, sample_rate, self.prompt_sample_rate)
        if waveform.ndim != 2 or waveform.shape[0] != 1 or not waveform.numel() or not torch.isfinite(waveform).all():
            raise ValueError("CosyVoice reference must be nonempty finite mono audio")
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
    def generate_sample(self, sample, *, output_dir):
        self._ensure_generator()
        reference_path = self._resolve_prompt_path(sample)
        if reference_path is None:
            raise FileNotFoundError(f"Missing CosyVoice reference: {sample.prompt_path}")
        prompt_waveform = self._load_prompt_audio(reference_path)
        target_text = (sample.target_text or "").strip()
        prompt_text = (sample.prompt_text or "").strip()
        if not target_text:
            target_text = prompt_text or self.default_prompt_text
        prompt_text_for_model = prompt_text or self.default_prompt_text
        self._log_clone_request(self.MODEL_NAME, 0, 1, str(sample.speaker_id),
                                reference_path, target_text, prompt_transcript=prompt_text)
        self._set_seed(sample.index)
        started = time.perf_counter()
        waveform, sample_rate = self._generator.generate(
            text=target_text, prompt_audio_16k=prompt_waveform,
            prompt_text=prompt_text_for_model, zero_shot_speaker_id=self.zero_shot_speaker_id)
        elapsed = time.perf_counter() - started
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError("CosyVoice output must be nonempty finite mono audio")
        path = self._speaker_output_dir(Path(output_dir).resolve(), str(sample.speaker_id)) / self._cloned_filename(sample, sample.index)
        sf.write(path, waveform, sample_rate)
        return path, elapsed

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path
        self._ensure_generator()
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)
        samples = dataset.get_zero_shot_samples(max_samples=self.max_samples)
        if not samples:
            raise RuntimeError(f"No zero-shot samples available for {self.MODEL_NAME} adversary.")
        self._log_attack_plan(self.MODEL_NAME, samples, self._count_available_prompts(samples))
        try:
            for sample in samples:
                path, elapsed = self.generate_sample(sample, output_dir=output_dir)
                self._record_synthesis_timing(path, elapsed)
        finally:
            self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, len(samples), len(samples))
