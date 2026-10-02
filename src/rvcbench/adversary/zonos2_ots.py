from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, Optional

import soundfile as sf

from .base_adversary import BaseAdversary
from rvcbench.models.zonos2 import Zonos2Generator, Zonos2GeneratorConfig


_ZONOS_LANGUAGE_BY_DATASET_CODE = {
    "EN": "en_us",
    "ZH": "cmn",
    "FR": "fr_fr",
}


def resolve_zonos_language(configured_language, target_language):
    if configured_language:
        return str(configured_language)
    code = str(target_language or "").upper()
    return _ZONOS_LANGUAGE_BY_DATASET_CODE.get(code, "en_us")


class Zonos2ZeroShotAdversary(BaseAdversary):
    """Runs ZONOS2 zero-shot cloning pipeline."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.checkpoint = str(self.config.get("checkpoint", "Zyphra/ZONOS2"))
        self.seed = self.config.get("seed", 42)
        self.clean_speaker_background = bool(self.config.get("clean_speaker_background", False))
        self.accurate_mode = bool(self.config.get("accurate_mode", True))
        self.max_tokens = self.config.get("max_tokens")
        self.language = self.config.get("language")
        self.max_samples = self.config.get("max_samples")

        self._generator: Optional[Zonos2Generator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = Zonos2GeneratorConfig(
            checkpoint=self.checkpoint,
            seed=self.seed,
            native_seed_policy=str(self.config.get('native_seed_policy', 'source_index')),
            clean_speaker_background=self.clean_speaker_background,
            accurate_mode=self.accurate_mode,
            max_tokens=self.max_tokens,
            code_path=self.config.get('code_path'),
            memory_ratio=float(self.config.get('memory_ratio', 0.9)),
            max_running_req=int(self.config.get('max_running_req', 256)),
            distributed_port=self.config.get('distributed_port'),
            vocoder_path=self.config.get('vocoder_path'),
            speaker_file_path=self.config.get('speaker_file_path'),
        )
        self._generator = Zonos2Generator(generator_config, self.device, self.logger)

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
                    "[ZONOS2] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for ZONOS2 adversary.")

        sample_prompt_texts: Dict[str, str] = {}
        for sample in samples:
            prompt_path = self._resolve_prompt_path(sample)
            if prompt_path is not None:
                sample_prompt_texts[str(prompt_path)] = sample.prompt_text or ""

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan("ZONOS2", samples, prompt_count)

        assert self._generator is not None
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[ZONOS2] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            lookup_key = str(reference_path)
            prompt_transcript = sample_prompt_texts.get(lookup_key, "").strip() or None
            desired_text = (sample.target_text or "").strip()
            if not desired_text:
                desired_text = (sample.prompt_text or "").strip()
            if not desired_text:
                desired_text = prompt_transcript or ""

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "ZONOS2",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                desired_text,
                prompt_transcript=prompt_transcript,
            )

            synth_start = time.perf_counter()
            wav, sr = self._generator.generate(
                text=desired_text,
                prompt_audio=reference_path,
                prompt_text=prompt_transcript,
                sample_index=sample.index,
                language=resolve_zonos_language(self.language, sample.target_language),
            )
            synth_elapsed = time.perf_counter() - synth_start

            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, sr)
            self._record_synthesis_timing(output_path, synth_elapsed)

        self._flush_synthesis_timings()
        if self.logger is not None:
            self.logger.info("[ZONOS2] Generated %d utterances.", len(samples))
