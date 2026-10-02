"""FireRedTTS2 zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from rvcbench.models.fireredtts2 import FireRedTTS2Generator, FireRedTTS2GeneratorConfig


class FireRedTTS2ZeroShotAdversary(BaseAdversary):
    """Runs FireRedTTS2 monologue mode for zero-shot voice cloning."""

    MODEL_NAME = "FireRedTTS2"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = self._resolve_required_path(self.config.get("code_path"))
        self.pretrained_dir = self._resolve_required_path(self.config.get("pretrained_dir"))
        self.gen_type = str(self.config.get("gen_type", "monologue"))
        self.use_bf16 = bool(self.config.get("use_bf16", False))
        self.temperature = float(self.config.get("temperature", 0.75))
        self.topk = int(self.config.get("topk", 20))
        self.min_token_frames = int(self.config.get('min_token_frames', 18))
        self.max_prompt_retries = int(self.config.get('max_prompt_retries', 3))
        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")
        self.default_prompt_text = str(
            self.config.get("default_prompt_text", "Here is a sample of the desired voice.")
        ).strip()
        self.require_prompt_text = bool(self.config.get("require_prompt_text", False))

        self._generator: Optional[FireRedTTS2Generator] = None

    def _resolve_required_path(self, value) -> str:
        if value in (None, ""):
            raise ValueError(f"{self.MODEL_NAME} adversary requires this path to be configured.")
        raw = str(value)
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        absolute_candidate = Path(to_absolute_path(raw))
        if absolute_candidate.exists():
            return str(absolute_candidate.resolve())
        return raw

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = FireRedTTS2GeneratorConfig(
            pretrained_dir=self.pretrained_dir,
            code_path=self.code_path,
            gen_type=self.gen_type,
            use_bf16=self.use_bf16,
            temperature=self.temperature,
            topk=self.topk,
            min_token_frames=self.min_token_frames,
            max_prompt_retries=self.max_prompt_retries,
        )
        self._generator = FireRedTTS2Generator(generator_config, self.device, self.logger)

    def generate_sample(self, sample, *, output_dir):
        reference_path = self._resolve_prompt_path(sample)
        if reference_path is None:
            raise ValueError('FireRedTTS2 requires reference audio')
        target_text = (sample.target_text or "").strip()
        prompt_text = (sample.prompt_text or "").strip()
        if not target_text:
            raise ValueError('FireRedTTS2 requires target text; reference text is not a substitute')
        if not prompt_text:
            raise ValueError('FireRedTTS2 requires the actual reference transcript')
        self._ensure_generator()
        self._log_clone_request(self.MODEL_NAME, 0, 1, str(sample.speaker_id),
                                reference_path, target_text, prompt_transcript=prompt_text)
        seed = self._sample_seed(sample)
        if seed is not None:
            from rvcbench.utils.seeding import configure_seeds
            configure_seeds(seed, logger=None)
        started = time.perf_counter()
        wav, sample_rate = self._generator.generate(
            text=target_text, prompt_wav=str(reference_path.resolve()), prompt_text=prompt_text)
        elapsed = time.perf_counter() - started
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or not np.isfinite(wav).all() or not wav.size:
            raise ValueError('FireRedTTS2 returned invalid mono audio')
        path = self._speaker_output_dir(Path(output_dir).resolve(), str(sample.speaker_id)) / self._cloned_filename(sample, sample.index)
        sf.write(path, wav, sample_rate)
        return path, elapsed

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)
        samples = dataset.get_zero_shot_samples(max_samples=self.max_samples)
        if not samples:
            raise RuntimeError(f"No zero-shot samples available for {self.MODEL_NAME} adversary.")
        self._log_attack_plan(self.MODEL_NAME, samples, self._count_available_prompts(samples))
        completed = 0
        try:
            for sample in samples:
                path, elapsed = self.generate_sample(sample, output_dir=output_dir)
                self._record_synthesis_timing(path, elapsed)
                completed += 1
        finally:
            self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, completed, len(samples))
