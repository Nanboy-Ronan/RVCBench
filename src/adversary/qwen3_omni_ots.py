"""Qwen-style Omni adversary integrations."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.qwen3_omni import Qwen3OmniGenerator, Qwen3OmniGeneratorConfig


class QwenOmniBaseZeroShotAdversary(BaseAdversary):
    """Shared implementation for Qwen-style Omni adversaries."""

    MODEL_NAME = "Qwen3-Omni"
    DEFAULT_CHECKPOINT_PATH = "checkpoints/Qwen3-Omni"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        checkpoint_value = self.config.get("checkpoint_path", self.DEFAULT_CHECKPOINT_PATH)
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_value)

        self.device_map = self.config.get("device_map", "auto")
        self.torch_dtype = self.config.get("torch_dtype", "auto")
        self.use_flash_attn2 = bool(self.config.get("use_flash_attn2", False))
        self.attn_implementation = self.config.get("attn_implementation")

        self.thinker_temperature = float(self.config.get("thinker_temperature", 0.7))
        self.thinker_top_p = float(self.config.get("thinker_top_p", 0.8))
        self.thinker_top_k = int(self.config.get("thinker_top_k", 20))
        self.thinker_max_new_tokens = int(self.config.get("thinker_max_new_tokens", 2048))
        self.thinker_do_sample = bool(self.config.get("thinker_do_sample", True))

        self.return_audio = bool(self.config.get("return_audio", True))
        self.use_audio_in_video = bool(self.config.get("use_audio_in_video", True))
        self.clean_up_tokenization_spaces = bool(
            self.config.get("clean_up_tokenization_spaces", False)
        )

        self.speaker = self.config.get("speaker")
        self.system_prompt = str(
            self.config.get("system_prompt", "") or ""
        ).strip() or None
        self.instruction_template = str(
            self.config.get(
                "instruction_template",
                "Please read the following text using the same voice as the provided audio sample: {text}",
            )
        )
        self.include_prompt_transcript = bool(self.config.get("include_prompt_transcript", False))
        self.additional_messages: List[Dict[str, Any]] = list(self.config.get("additional_messages", []))

        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")

        self._generator: Optional[Qwen3OmniGenerator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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

        generator_config = Qwen3OmniGeneratorConfig(
            checkpoint_path=self.checkpoint_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            use_flash_attn2=self.use_flash_attn2,
            attn_implementation=self.attn_implementation,
            return_audio=self.return_audio,
            use_audio_in_video=self.use_audio_in_video,
            thinker_max_new_tokens=self.thinker_max_new_tokens,
            thinker_temperature=self.thinker_temperature,
            thinker_top_p=self.thinker_top_p,
            thinker_top_k=self.thinker_top_k,
            thinker_do_sample=self.thinker_do_sample,
            speaker=self.speaker,
            system_prompt=self.system_prompt,
            seed=self.seed,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
        )
        self._generator = Qwen3OmniGenerator(generator_config, self.device, self.logger)

    def _prepare_instruction(self, text: str) -> str:
        cleaned = text.strip()
        template = self.instruction_template
        if "{text}" in template:
            return template.replace("{text}", cleaned)
        if cleaned:
            return f"{template.strip()} {cleaned}".strip()
        return template.strip()

    def _build_messages(
        self,
        reference_path: Path,
        prompt_transcript: str,
        target_text: str,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                        }
                    ],
                }
            )

        for entry in self.additional_messages:
            if isinstance(entry, dict):
                messages.append(dict(entry))

        user_content: List[Dict[str, Any]] = []

        if reference_path is not None:
            user_content.append(
                {
                    "type": "audio",
                    "audio": str(reference_path.resolve()),
                }
            )

        if self.include_prompt_transcript and prompt_transcript:
            user_content.append(
                {
                    "type": "text",
                    "text": prompt_transcript.strip(),
                }
            )

        instruction = self._prepare_instruction(target_text)
        if instruction:
            user_content.append(
                {
                    "type": "text",
                    "text": instruction,
                }
            )

        messages.append({"role": "user", "content": user_content})
        return messages

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

            messages = self._build_messages(reference_path, prompt_transcript, target_text)

            synth_start = time.perf_counter()
            wav, sample_rate = self._generator.generate(messages, sample_index=idx)
            synth_elapsed = time.perf_counter() - synth_start

            wav = np.asarray(wav, dtype=np.float32)
            if not np.isfinite(wav).all():
                wav = np.nan_to_num(wav)
            wav = np.clip(wav, -1.0, 1.0)

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            output_filename = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_filename

            sf.write(str(output_path), wav, sample_rate)
            self._record_synthesis_timing(output_path, synth_elapsed)

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


class Qwen3OmniZeroShotAdversary(QwenOmniBaseZeroShotAdversary):
    """Runs the Qwen3-Omni multimodal model for zero-shot voice cloning."""

    MODEL_NAME = "Qwen3-Omni"
    DEFAULT_CHECKPOINT_PATH = "checkpoints/Qwen3-Omni"
