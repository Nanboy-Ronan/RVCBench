"""IndexTTS zero-shot adversary integration."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.index_tts import IndexTTSGenerator, IndexTTSGeneratorConfig


class IndexTTSZeroShotAdversary(BaseAdversary):
    """Runs IndexTTS2 for zero-shot voice cloning."""

    MODEL_NAME = "IndexTTS"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.get("code_path", "checkpoints/index-tts"))).resolve()
        self.model_dir = Path(
            to_absolute_path(self.config.get("model_dir", "checkpoints/index-tts/checkpoints"))
        ).resolve()
        self.config_path = Path(
            to_absolute_path(self.config.get("config_path", "checkpoints/index-tts/checkpoints/config.yaml"))
        ).resolve()
        runtime_python_value = self.config.get("runtime_python")
        self.runtime_python = (
            Path(to_absolute_path(str(runtime_python_value))).resolve()
            if runtime_python_value not in (None, "")
            else None
        )
        worker_script_value = self.config.get("worker_script_path", "scripts/index_tts_worker.py")
        self.worker_script_path = Path(to_absolute_path(str(worker_script_value))).resolve()

        self.use_fp16 = bool(self.config.get("use_fp16", False))
        self.use_cuda_kernel = bool(self.config.get("use_cuda_kernel", False))
        self.use_deepspeed = bool(self.config.get("use_deepspeed", False))
        self.use_accel = bool(self.config.get("use_accel", False))
        self.use_torch_compile = bool(self.config.get("use_torch_compile", False))
        self.interval_silence = int(self.config.get("interval_silence", 200))
        self.max_text_tokens_per_segment = int(self.config.get("max_text_tokens_per_segment", 120))
        self.verbose = bool(self.config.get("verbose", False))
        self.default_prompt_text = str(
            self.config.get("default_prompt_text", "Here is a sample of the desired voice.")
        ).strip()
        self.max_samples = self.config.get("max_samples")

        self.emo_alpha = float(self.config.get("emo_alpha", 1.0))
        self.use_prompt_as_emo_audio = bool(self.config.get("use_prompt_as_emo_audio", False))
        self.explicit_emo_audio_prompt = self.config.get("emo_audio_prompt")

        generation_kwargs = {}
        for key in (
            "top_p",
            "top_k",
            "temperature",
            "length_penalty",
            "num_beams",
            "repetition_penalty",
            "max_mel_tokens",
            "do_sample",
        ):
            value = self.config.get(key)
            if value is not None:
                generation_kwargs[key] = value

        self._generator: Optional[IndexTTSGenerator] = None
        self._generation_kwargs = generation_kwargs

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return

        generator_config = IndexTTSGeneratorConfig(
            code_path=self.code_path,
            model_dir=self.model_dir,
            config_path=self.config_path,
            runtime_python=self.runtime_python,
            worker_script_path=self.worker_script_path,
            use_fp16=self.use_fp16,
            use_cuda_kernel=self.use_cuda_kernel,
            use_deepspeed=self.use_deepspeed,
            use_accel=self.use_accel,
            use_torch_compile=self.use_torch_compile,
            interval_silence=self.interval_silence,
            max_text_tokens_per_segment=self.max_text_tokens_per_segment,
            verbose=self.verbose,
            generation_kwargs=dict(self._generation_kwargs),
        )
        self._generator = IndexTTSGenerator(generator_config, self.device, self.logger)

    def _resolve_emo_prompt(self, reference_path: Path) -> Optional[Path]:
        if self.explicit_emo_audio_prompt not in (None, ""):
            return Path(to_absolute_path(str(self.explicit_emo_audio_prompt))).resolve()
        if self.use_prompt_as_emo_audio:
            return reference_path
        return None

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

        completed = 0
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

            target_text = (sample.target_text or "").strip()
            if not target_text:
                target_text = (sample.prompt_text or "").strip()
            if not target_text:
                target_text = self.default_prompt_text

            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            output_filename = self._cloned_filename(sample, idx)
            output_wav_path = speaker_dir / output_filename
            emo_prompt = self._resolve_emo_prompt(reference_path)

            self._log_clone_request(
                self.MODEL_NAME,
                idx,
                len(samples),
                speaker_id,
                reference_path,
                target_text,
                prompt_transcript=sample.prompt_text,
            )

            try:
                synth_start = time.perf_counter()
                self._generator.generate(
                    ref_audio=reference_path,
                    text=target_text,
                    output_path=output_wav_path,
                    emo_audio_prompt=emo_prompt,
                    emo_alpha=self.emo_alpha,
                )
                synth_elapsed = time.perf_counter() - synth_start
                synth_elapsed = max(
                    0.0,
                    synth_elapsed - float(getattr(self._generator, "last_output_wait_sec", 0.0)),
                )
            except Exception as exc:
                if self.logger:
                    self.logger.error(
                        "[%s] Generation failed for sample %d (%s): %s",
                        self.MODEL_NAME,
                        idx,
                        speaker_id,
                        exc,
                    )
                continue

            self._record_synthesis_timing(output_wav_path, synth_elapsed)
            completed += 1

        self._flush_synthesis_timings()
        if self.logger:
            self.logger.info("[%s] Generated %d/%d utterances.", self.MODEL_NAME, completed, len(samples))
