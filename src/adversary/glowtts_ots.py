import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.glowtts import GlowTTSSynthesizer, GlowTTSSynthesizerConfig


class GlowTTSZeroShotAdversary(BaseAdversary):
    """Runs Glow-TTS in zero-shot mode using protected audio for speaker selection."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        self.config_path = Path(to_absolute_path(self.config.config_path)).resolve()
        self.checkpoint_path = Path(to_absolute_path(self.config.checkpoint_path)).resolve()

        self.noise_scale = float(self.config.get("noise_scale", 0.667))
        self.length_scale = float(self.config.get("length_scale", 1.0))
        self.vocoder_type = str(self.config.get("vocoder", "waveglow")).lower()
        self.waveglow_sigma = float(self.config.get("waveglow_sigma", 0.666))
        self.sample_rate = int(self.config.get("sample_rate", 22050))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")
        self.griffinlim_iters = int(self.config.get("griffinlim_iters", 60))

        waveglow_checkpoint = self.config.get("waveglow_checkpoint")
        self.waveglow_checkpoint = (
            Path(to_absolute_path(waveglow_checkpoint)).resolve()
            if waveglow_checkpoint
            else None
        )
        waveglow_code = self.config.get("waveglow_code_path")
        self.waveglow_code_path = (
            Path(to_absolute_path(waveglow_code)).resolve()
            if waveglow_code
            else None
        )

        speaker_map = self.config.get("speaker_map_path")
        self.speaker_map_path = (
            Path(to_absolute_path(speaker_map)).resolve()
            if speaker_map
            else None
        )

        self._synthesizer: Optional[GlowTTSSynthesizer] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_speaker_mapping(self) -> Optional[Dict[str, int]]:
        if self.speaker_map_path is None:
            return None
        if not self.speaker_map_path.exists():
            raise FileNotFoundError(f"Speaker map file not found: {self.speaker_map_path}")
        with open(self.speaker_map_path, "r", encoding="utf-8") as handle:
            try:
                raw_map = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "speaker_map_path must point to a JSON file mapping speaker keys to integer IDs."
                ) from exc
        sanitized = {str(key): int(value) for key, value in raw_map.items()}
        return sanitized

    def _ensure_synthesizer(self) -> None:
        if self._synthesizer is not None:
            return

        speaker_map = self._load_speaker_mapping()
        synth_config = GlowTTSSynthesizerConfig(
            code_path=self.code_path,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            noise_scale=self.noise_scale,
            length_scale=self.length_scale,
            vocoder_type=self.vocoder_type,
            waveglow_sigma=self.waveglow_sigma,
            waveglow_checkpoint=self.waveglow_checkpoint,
            waveglow_code_path=self.waveglow_code_path,
            griffinlim_iters=self.griffinlim_iters,
            sample_rate=self.sample_rate,
            speaker_map=speaker_map,
        )
        self._synthesizer = GlowTTSSynthesizer(synth_config, self.device, self.logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._ensure_synthesizer()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[GlowTTS] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Glow-TTS adversary.")

        prompt_count = self._count_available_prompts(samples)

        self._log_attack_plan("GlowTTS", samples, prompt_count)

        assert self._synthesizer is not None
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[GlowTTS] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            text = (sample.prompt_text or "").strip()
            if not text:
                text = (sample.target_text or "").strip()
            if not text:
                text = "This is a synthesized sample."

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            self._log_clone_request(
                "GlowTTS",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                text,
            )
            self.logger.debug(
                "[GlowTTS] Generating sample %d for speaker '%s' using reference %s",
                idx,
                speaker_id,
                reference_path.name,
            )

            synth_start = time.perf_counter()
            wav = self._synthesizer.synthesize(text, speaker_id)
            synth_elapsed = time.perf_counter() - synth_start
            if wav.size == 0:
                self.logger.warning("[GlowTTS] Empty waveform for text '%s'. Skipping.", text[:32])
                continue

            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, self.sample_rate)
            self._record_synthesis_timing(output_path, synth_elapsed)

        self._flush_synthesis_timings()
        self.logger.info("[GlowTTS] Generated %d utterances.", len(samples))
