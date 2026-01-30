from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.kimi_audio import KimiAudioGenerator, KimiAudioGeneratorConfig


class KimiAudioZeroShotAdversary(BaseAdversary):
    """Runs Kimi Audio's zero-shot cloning workflow."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        self.model_path = str(self.config.get("model_path", "moonshotai/Kimi-Audio-7B-Instruct"))
        self.load_detokenizer = bool(self.config.get("load_detokenizer", True))
        self.audio_temperature = float(self.config.get("audio_temperature", 0.8))
        self.audio_top_k = int(self.config.get("audio_top_k", 10))
        self.text_temperature = float(self.config.get("text_temperature", 0.0))
        self.text_top_k = int(self.config.get("text_top_k", 5))
        self.audio_repetition_penalty = float(self.config.get("audio_repetition_penalty", 1.0))
        self.audio_repetition_window_size = int(self.config.get("audio_repetition_window_size", 64))
        self.text_repetition_penalty = float(self.config.get("text_repetition_penalty", 1.0))
        self.text_repetition_window_size = int(self.config.get("text_repetition_window_size", 16))
        self.max_new_tokens = self.config.get("max_new_tokens", -1)
        self.sample_rate = int(self.config.get("sample_rate", 24000))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")
        self.instruction_template = str(
            self.config.get(
                "instruction_template",
                "Please perform voice clone with the following text sentence using the same voice as the provided audio sample: {text}",
            )
        )
        self.additional_messages: List[Dict[str, Any]] = list(self.config.get("additional_messages", []))

        self._generator: Optional[KimiAudioGenerator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = KimiAudioGeneratorConfig(
            code_path=self.code_path,
            model_path=self.model_path,
            load_detokenizer=self.load_detokenizer,
            audio_temperature=self.audio_temperature,
            audio_top_k=self.audio_top_k,
            text_temperature=self.text_temperature,
            text_top_k=self.text_top_k,
            audio_repetition_penalty=self.audio_repetition_penalty,
            audio_repetition_window_size=self.audio_repetition_window_size,
            text_repetition_penalty=self.text_repetition_penalty,
            text_repetition_window_size=self.text_repetition_window_size,
            max_new_tokens=self.max_new_tokens,
            sample_rate=self.sample_rate,
        )
        self._generator = KimiAudioGenerator(generator_config, self.device, self.logger)

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
        for entry in self.additional_messages:
            if not isinstance(entry, dict):
                continue
            messages.append(dict(entry))

        ref_path_str = str(reference_path.resolve())
        transcript = prompt_transcript.strip()

        messages.append(
            {
                "role": "assistant",
                "message_type": "audio",
                "content": ref_path_str,
            }
        )

        if transcript:
            messages.append(
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": transcript,
                }
            )

        instruction = self._prepare_instruction(target_text)
        if instruction:
            messages.append(
                {
                    "role": "user",
                    "message_type": "text",
                    "content": instruction, # prompt
                }
            )
        return messages

    def _set_seed(self, index: int) -> None:
        if self.seed is None:
            return
        adjusted_seed = int(self.seed) + int(index)
        random.seed(adjusted_seed)
        np.random.seed(adjusted_seed % (2**32 - 1))
        torch.manual_seed(adjusted_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(adjusted_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._ensure_generator()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[KimiAudio] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Kimi Audio adversary.")

        sample_prompt_targets: Dict[str, str] = {}
        sample_prompt_transcripts: Dict[str, str] = {}
        for sample in samples:
            prompt_path = self._resolve_prompt_path(sample)
            if prompt_path is not None:
                key = str(prompt_path)
                text_value = sample.prompt_text or ""
                sample_prompt_targets[key] = text_value
                sample_prompt_transcripts[key] = text_value

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan("KimiAudio", samples, prompt_count)

        assert self._generator is not None
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[KimiAudio] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            self._set_seed(idx)

            lookup_key = str(reference_path)
            prompt_transcript = sample_prompt_transcripts.get(lookup_key, "")
            target_override = sample_prompt_targets.get(lookup_key, "")
            target_text = (sample.target_text or "").strip()
            if not target_text:
                target_text = (sample.prompt_text or "").strip()
            if not target_text:
                target_text = target_override or prompt_transcript or ""
            messages = self._build_messages(reference_path, prompt_transcript, target_text)
            synth_start = time.perf_counter()
            wav, sr = self._generator.generate(messages, sample_index=idx)
            synth_elapsed = time.perf_counter() - synth_start
            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "KimiAudio",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                target_text,
                prompt_transcript=prompt_transcript,
            )
            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, sr)
            self._record_synthesis_timing(output_path, synth_elapsed)

            if self.logger is not None:
                generated_text = getattr(self._generator, "last_generated_text", None)
                self.logger.debug(
                    "[KimiAudio] Sample %d generated for speaker %s using %s. Model response text: %s",
                    idx,
                    sample.speaker_id,
                    reference_path.name,
                    generated_text or "<empty>",
                )

        self._flush_synthesis_timings()
        if self.logger is not None:
            self.logger.info("[KimiAudio] Generated %d utterances.", len(samples))
