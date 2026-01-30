"""MGM-Omni off-the-shelf adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import soundfile as sf

from .base_adversary import BaseAdversary
from src.models.mgm_omni import MGMOmniGenerator, MGMOmniGeneratorConfig


class MGMOmniZeroShotAdversary(BaseAdversary):
    """Runs the MGM-Omni pipeline for zero-shot voice cloning."""

    MODEL_NAME = "MGM-Omni"
    DEFAULT_REPO_ROOT = "checkpoints/MGM-Omni"
    DEFAULT_MODEL_PATH = "wcy1122/MGM-Omni-TTS-2B-0927"
    DEFAULT_SPEECHLM_PATH = "wcy1122/MGM-Omni-TTS-2B-0927"
    DEFAULT_COSYVOICE_PATH = None

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.repo_root = self._resolve_path(self.config.get("repo_root", self.DEFAULT_REPO_ROOT))
        self.checkpoint_path = self._resolve_path(self.config.get("checkpoint_path", self.DEFAULT_MODEL_PATH))
        self.speechlm_path = self._resolve_path(
            self.config.get("speechlm_checkpoint_path", self.DEFAULT_SPEECHLM_PATH)
        )
        self.cosyvoice_path = self._resolve_path(self.config.get("cosyvoice_path", self.DEFAULT_COSYVOICE_PATH))

        self.device_map = self.config.get("device_map", "auto")
        self.load_8bit = bool(self.config.get("load_8bit", False))
        self.load_4bit = bool(self.config.get("load_4bit", False))
        self.use_flash_attn = bool(self.config.get("use_flash_attn", True))

        self.do_sample = bool(self.config.get("do_sample", True))
        self.temperature = float(self.config.get("generation_temperature", 0.3))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 4096))

        self.enable_vision_tower = bool(self.config.get("enable_vision_tower", False))
        self.enable_speech_tower = bool(self.config.get("enable_speech_tower", False))

        self.instruction_template = str(
            self.config.get(
                "instruction_template",
                "Please read the following text using the same voice as the provided audio sample: {text}",
            )
        )
        self.include_prompt_transcript = bool(self.config.get("include_prompt_transcript", True))
        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")

        self._generator: Optional[MGMOmniGenerator] = None

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _resolve_path(self, raw_value) -> Path:
        raw = str(raw_value)
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return candidate.resolve()
        # If the path does not exist, return the raw candidate without forcing an absolute
        # resolution so that HuggingFace repo ids like "namespace/model" remain intact.
        return candidate

    def _prepare_instruction(self, text: str) -> str:
        # MGM-Omni works best when we feed the exact phrase we want spoken.
        cleaned = (text or "").strip()
        if cleaned:
            return cleaned
        return self.instruction_template.replace("{text}", "").strip()

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = MGMOmniGeneratorConfig(
            repo_root=str(self.repo_root),
            checkpoint_path=str(self.checkpoint_path),
            speechlm_checkpoint_path=str(self.speechlm_path),
            cosyvoice_path=str(self.cosyvoice_path),
            device_map=self.device_map,
            load_8bit=self.load_8bit,
            load_4bit=self.load_4bit,
            use_flash_attn=self.use_flash_attn,
            do_sample=self.do_sample,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            enable_vision_tower=self.enable_vision_tower,
            enable_speech_tower=self.enable_speech_tower,
        )
        self._generator = MGMOmniGenerator(generator_config, self.device, self.logger)

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

            prompt_transcript = (sample.prompt_text or "").strip()
            target_text = (sample.target_text or "").strip()
            if not target_text:
                target_text = prompt_transcript

            instruction = self._prepare_instruction(target_text)
            reference_transcript = prompt_transcript if self.include_prompt_transcript else None

            synth_start = time.perf_counter()
            wav, sample_rate = self._generator.generate(
                instruction_text=instruction,
                reference_audio_path=reference_path,
                reference_transcript=reference_transcript,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            synth_elapsed = time.perf_counter() - synth_start

            wav = np.asarray(wav, dtype=np.float32)
            wav = np.clip(np.nan_to_num(wav), -1.0, 1.0)

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            output_filename = self._cloned_filename(sample, idx)
            clone_path = speaker_dir / output_filename
            sf.write(str(clone_path), wav, sample_rate)
            self._record_synthesis_timing(clone_path, synth_elapsed)

            self._log_clone_request(
                self.MODEL_NAME,
                idx,
                len(samples),
                speaker_id,
                reference_path,
                target_text,
                prompt_transcript=prompt_transcript,
            )

        self._flush_synthesis_timings()
