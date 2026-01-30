from pathlib import Path
import time
from typing import List, Optional, Sequence

import soundfile as sf
from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.higgs_audio import HiggsAudioGenerator, HiggsAudioGeneratorConfig


class HiggsAudioZeroShotAdversary(BaseAdversary):
    """Runs Higgs Audio's example generation pipeline for zero-shot cloning."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        if "model_path" not in self.config:
            raise ValueError("Higgs Audio adversary config must specify 'model_path'.")
        self.model_path = str(self.config.model_path)
        self.audio_tokenizer_path = str(
            self.config.get("audio_tokenizer_path", "bosonai/higgs-audio-v2-tokenizer")
        )
        self.scene_prompt_path = self.config.get("scene_prompt_path")
        # Inline scene/system prompt text (optional). If provided, overrides file path.
        self.scene_prompt_text = self.config.get("scene_prompt_text")
        self.temperature = float(self.config.get("temperature", 1.0))
        self.top_k = int(self.config.get("top_k", 50))
        self.top_p = float(self.config.get("top_p", 0.95))
        self.ras_win_len = int(self.config.get("ras_win_len", 7))
        self.ras_win_max_num_repeat = int(self.config.get("ras_win_max_num_repeat", 2))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 2048))
        self.chunk_method = self.config.get("chunk_method")
        self.chunk_max_word_num = int(self.config.get("chunk_max_word_num", 200))
        self.chunk_max_num_turns = int(self.config.get("chunk_max_num_turns", 1))
        self.generation_chunk_buffer_size = self.config.get("generation_chunk_buffer_size")
        self.seed = self.config.get("seed")
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")
        self.reference_prompt_text = str(
            self.config.get(
                "reference_prompt_text",
                "Here is a sample of the desired voice.",
            )
        )
        self.use_static_kv_cache = bool(self.config.get("use_static_kv_cache", False))
        # New message-format options
        self.reference_role = str(self.config.get("reference_role", "assistant")).lower()
        self.reference_audio_first = bool(self.config.get("reference_audio_first", False))

        self._generator: Optional[HiggsAudioGenerator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = HiggsAudioGeneratorConfig(
            code_path=self.code_path,
            model_path=self.model_path,
            audio_tokenizer_path=self.audio_tokenizer_path,
            scene_prompt_path=self.scene_prompt_path,
            scene_prompt_text=self.scene_prompt_text,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            ras_win_len=self.ras_win_len,
            ras_win_max_num_repeat=self.ras_win_max_num_repeat,
            max_new_tokens=self.max_new_tokens,
            chunk_method=self.chunk_method,
            chunk_max_word_num=self.chunk_max_word_num,
            chunk_max_num_turns=self.chunk_max_num_turns,
            generation_chunk_buffer_size=self.generation_chunk_buffer_size,
            seed=self.seed,
            reference_prompt_text=self.reference_prompt_text,
            use_static_kv_cache=self.use_static_kv_cache,
            reference_role=self.reference_role,
            reference_audio_first=self.reference_audio_first,
        )
        self._generator = HiggsAudioGenerator(generator_config, self.device, self.logger)

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
                    "[HiggsAudio] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for Higgs Audio adversary.")

        prompt_count = self._count_available_prompts(samples)
        self._log_attack_plan("HiggsAudio", samples, prompt_count)

        assert self._generator is not None
        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[HiggsAudio] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            # Use target_text (ground truth text) for voice conversion
            text = (sample.target_text or "").strip()
            if not text:
                text = (sample.prompt_text or "").strip()
            if not text:
                text = self.reference_prompt_text
            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)

            self._log_clone_request(
                "HiggsAudio",
                idx,
                len(samples),
                speaker_id,
                reference_path,
                text,
            )

            synth_start = time.perf_counter()
            wav, sr = self._generator.generate(text, speaker_id, reference_path, idx)
            synth_elapsed = time.perf_counter() - synth_start
            output_name = self._cloned_filename(sample, idx)
            output_path = speaker_dir / output_name
            sf.write(output_path, wav, sr)
            self._record_synthesis_timing(output_path, synth_elapsed)

        self._flush_synthesis_timings()
        self.logger.info("[HiggsAudio] Generated %d utterances.", len(samples))
