from __future__ import annotations

import random
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.playdiffusion import PlayDiffusionGenerator, PlayDiffusionGeneratorConfig


class PlayDiffusionZeroShotAdversary(BaseAdversary):
    """Runs the PlayDiffusion inference stack for zero-shot voice cloning."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        preset_dir = self.config.get("preset_dir")
        self.preset_dir = (
            Path(to_absolute_path(str(preset_dir))).resolve()
            if preset_dir is not None
            else None
        )
        cache_dir = self.config.get("cache_dir")
        self.cache_dir = (
            Path(to_absolute_path(str(cache_dir))).resolve()
            if cache_dir is not None
            else None
        )

        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower().strip()
        self.max_samples = self.config.get("max_samples")
        self.seed = self.config.get("seed")
        self.default_prompt_text = str(
            self.config.get(
                "default_prompt_text",
                "Here is a short sample of the desired voice.",
            )
        )

        self._rng: Optional[random.Random] = None
        if self.seed is not None:
            try:
                seed_value = int(self.seed)
                self._rng = random.Random(seed_value)
            except (TypeError, ValueError):
                self.logger.warning("[PlayDiffusion] Invalid seed '%s'; falling back to random module.", self.seed)

        generator_config = PlayDiffusionGeneratorConfig(
            code_path=self.code_path,
            preset_dir=self.preset_dir,
            cache_dir=self.cache_dir,
            num_steps=int(self.config.get("num_steps", 30)),
            init_temp=float(self.config.get("init_temp", 1.0)),
            init_diversity=float(self.config.get("init_diversity", 1.0)),
            guidance=float(self.config.get("guidance", 0.5)),
            rescale=float(self.config.get("rescale", 0.7)),
            top_k=int(self.config.get("top_k", 25)),
            audio_token_syllable_ratio=self.config.get("audio_token_syllable_ratio"),
            vocoder_checkpoint=str(self.config.get("vocoder_checkpoint", "v090_g_01105000")),
            tokenizer_file=str(self.config.get("tokenizer_file", "tokenizer-multi_bpe16384_merged_extended_58M.json")),
            speech_tokenizer_checkpoint=str(self.config.get("speech_tokenizer_checkpoint", "xlsr2_1b_v2_custom.pt")),
            kmeans_layer_checkpoint=str(self.config.get("kmeans_layer_checkpoint", "kmeans_10k.npy")),
            voice_encoder_checkpoint=str(self.config.get("voice_encoder_checkpoint", "voice_encoder_1992000.pt")),
            inpainter_checkpoint=str(self.config.get("inpainter_checkpoint", "last_250k_fixed.pkl")),
            speech_tokenizer_sample_rate=int(self.config.get("speech_tokenizer_sample_rate", 16000)),
        )
        self._generator = PlayDiffusionGenerator(generator_config, device, logger)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def attack(self, *, output_path, dataset, protected_audio_path=None):
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[PlayDiffusion] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for PlayDiffusion adversary.")

        dataset_prompt_texts: Dict[str, str] = {}
        for sample in samples:
            prompt_path = self._resolve_prompt_path(sample)
            if prompt_path is not None:
                dataset_prompt_texts[str(prompt_path)] = sample.prompt_text or ""

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan("PlayDiffusion", samples, prompt_count)

        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[PlayDiffusion] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            lookup_key = str(reference_path)
            prompt_transcript = dataset_prompt_texts.get(lookup_key, "").strip() or self.default_prompt_text
            script_text = (sample.target_text or "").strip()
            if not script_text:
                script_text = (sample.prompt_text or "").strip()
            if not script_text:
                script_text = prompt_transcript or ""
            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "PlayDiffusion",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                script_text,
                prompt_transcript=prompt_transcript,
            )

            synth_start = time.perf_counter()
            sample_rate, wav = self._generator.generate(
                text=script_text,
                prompt_audio=reference_path,
                prompt_text=prompt_transcript,
                sample_index=idx,
            )
            synth_elapsed = time.perf_counter() - synth_start

            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, sample_rate)
            self._record_synthesis_timing(output_path, synth_elapsed)

            if self.logger is not None:
                self.logger.debug(
                    "[PlayDiffusion] Generated sample %d for speaker %s using reference %s",
                    idx,
                    sample.speaker_id,
                    reference_path.name,
                )

        self._flush_synthesis_timings()
        if self.logger is not None:
            self.logger.info("[PlayDiffusion] Generated %d utterances.", len(samples))
