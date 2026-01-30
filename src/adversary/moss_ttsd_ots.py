from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.moss_ttsd import MossTTSDGenerator, MossTTSDGeneratorConfig


class MossTTSDZeroShotAdversary(BaseAdversary):
    """Runs the open-source MOSS-TTSD model for zero-shot voice cloning."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        self.model_path = str(self.config.get("model_path", "fnlp/MOSS-TTSD-v0.5"))
        self.spt_config_path = Path(
            to_absolute_path(self.config.spt_config_path)
        ).resolve()
        self.spt_checkpoint_path = Path(
            to_absolute_path(self.config.spt_checkpoint_path)
        ).resolve()
        self.system_prompt = str(
            self.config.get(
                "system_prompt",
                "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
            )
        )
        self.torch_dtype = self.config.get("torch_dtype", "bf16")
        self.attn_implementation = str(
            self.config.get("attn_implementation", "sdpa")
        )  # flash_attention_2
        self.use_normalize = bool(self.config.get("use_normalize", True))
        self.silence_duration = float(self.config.get("silence_duration", 0.0))
        self.reference_assignment = (
            str(self.config.get("reference_assignment", "round_robin")).lower().strip()
        )
        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")
        self.default_prompt_text = str(
            self.config.get(
                "default_prompt_text",
                "Here is a short sample of the desired voice.",
            )
        )
        self.use_prompt_transcript = bool(
            self.config.get("use_prompt_transcript", False)
        )

        self._generator: Optional[MossTTSDGenerator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = MossTTSDGeneratorConfig(
            code_path=self.code_path,
            model_path=self.model_path,
            spt_config_path=self.spt_config_path,
            spt_checkpoint_path=self.spt_checkpoint_path,
            system_prompt=self.system_prompt,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            use_normalize=self.use_normalize,
            silence_duration=self.silence_duration,
            seed=self.seed,
        )
        self._generator = MossTTSDGenerator(generator_config, self.device, self.logger)

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
                    "[MOSS-TTSD] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError(
                "No zero-shot samples available for MOSS-TTSD adversary."
            )

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan("MOSS-TTSD", samples, prompt_count)

        assert self._generator is not None
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[MOSS-TTSD] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            prompt_transcript = (
                sample.prompt_text or ""
            ).strip() or self.default_prompt_text
            prompt_text_tagged = ""
            if self.use_prompt_transcript:
                effective_prompt = (
                    prompt_transcript or ""
                ).strip() or self.default_prompt_text
                prompt_text_tagged = f"{effective_prompt}"
            script_text = (sample.target_text or "").strip()
            if not script_text:
                script_text = (sample.prompt_text or "").strip()
            if not script_text:
                script_text = self.default_prompt_text
            dialogue = f"{script_text}"
            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "MOSS-TTSD",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                script_text,
                prompt_transcript=(
                    prompt_transcript if self.use_prompt_transcript else None
                ),
            )

            synth_start = time.perf_counter()
            wav, sr = self._generator.generate(
                text=dialogue,
                prompt_audio=reference_path,
                prompt_text=prompt_text_tagged,
                sample_index=idx,
            )
            synth_elapsed = time.perf_counter() - synth_start

            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, sr)
            self._record_synthesis_timing(output_path, synth_elapsed)

            if self.logger is not None:
                self.logger.debug(
                    "[MOSS-TTSD] Generated sample %d for speaker %s using reference %s",
                    idx,
                    sample.speaker_id,
                    reference_path.name,
                )

        self._flush_synthesis_timings()
        if self.logger is not None:
            self.logger.info("[MOSS-TTSD] Generated %d utterances.", len(samples))
