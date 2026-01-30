import re
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

from hydra.utils import to_absolute_path

from .base_adversary import BaseAdversary
from src.models.ozspeech import OzSpeechSynthesizer


class OzSpeechZeroShotAdversary(BaseAdversary):
    """Wrapper around OZSpeech's off-the-shelf synthesis script."""

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        ckpt_key = "checkpoint_path" if "checkpoint_path" in self.config else "ckpt_path"
        cfg_key = "config_path" if "config_path" in self.config else "cfg_path"
        if ckpt_key not in self.config or cfg_key not in self.config:
            raise ValueError("OZSpeech adversary config must provide 'checkpoint_path'/'ckpt_path' and 'config_path'/'cfg_path'.")

        self.checkpoint_path = Path(to_absolute_path(self.config[ckpt_key])).resolve()
        self.config_path = Path(to_absolute_path(self.config[cfg_key])).resolve()

        self.code_path = (
            Path(to_absolute_path(self.config["code_path"])).resolve()
            if "code_path" in self.config
            else None
        )

        encoder_key = next((k for k in ("codec_encoder_path", "facodec_encoder_path") if k in self.config), None)
        decoder_key = next((k for k in ("codec_decoder_path", "facodec_decoder_path") if k in self.config), None)
        codec_encoder_value = self.config.get(encoder_key) if encoder_key is not None else None
        codec_decoder_value = self.config.get(decoder_key) if decoder_key is not None else None

        self.codec_encoder_path = (
            Path(to_absolute_path(codec_encoder_value)).resolve()
            if codec_encoder_value not in (None, "")
            else None
        )
        self.codec_decoder_path = (
            Path(to_absolute_path(codec_decoder_value)).resolve()
            if codec_decoder_value not in (None, "")
            else None
        )

        self.temperature = float(self.config.get("temperature", 0.01))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")

        self._imports_loaded = False
        self._device_str = None
        self._synthesizer: Optional[OzSpeechSynthesizer] = None

    def _ensure_imports(self):
        if self._imports_loaded:
            return

        if self.code_path is not None and not self.code_path.exists():
            raise FileNotFoundError(f"OZSpeech code path not found: {self.code_path}")

        if self.device.type == "cuda":
            index = 0 if self.device.index is None else self.device.index
            self._device_str = f"cuda:{index}"
        else:
            self._device_str = "cpu"

        self._synthesizer = OzSpeechSynthesizer(
            cfg_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device=self._device_str,
            temperature=self.temperature,
            logger=self.logger,
            code_path=self.code_path,
            codec_encoder_path=self.codec_encoder_path,
            codec_decoder_path=self.codec_decoder_path,
        )

        self._imports_loaded = True

    def _build_manifest(
        self,
        manifest_path: Path,
        prompt_files: Sequence[Path],
        samples: Sequence["ZeroShotSample"],
    ) -> None:
        total = len(samples)
        with open(manifest_path, "w", encoding="utf-8") as manifest:
            for idx, (sample, prompt_path) in enumerate(zip(samples, prompt_files)):
                target_stub = self._target_stub_for_sample(sample, idx)
                # Prefer the ground-truth text so synthesized speech matches evaluation targets.
                raw_text = sample.target_text
                text = raw_text.replace("|", " ").replace("\n", " ")
                if not text:
                    text = "This is a synthesized example."
                self._log_clone_request(
                    "OZSpeech",
                    idx,
                    total,
                    str(sample.speaker_id),
                    prompt_path,
                    raw_text,
                )
                line = f"{target_stub}|{prompt_path.name}|{text}|||0.0\n"
                manifest.write(line)

    def _target_stub_for_sample(self, sample, index: int) -> str:
        base_stub = sample.target_stub or f"sample_{index}"
        base_stub = re.sub(r'[^a-zA-Z0-9_-]', '_', base_stub)
        speaker_slug = self._speaker_slug(sample.speaker_id)
        return f"{base_stub}_{speaker_slug}_{index}"

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._ensure_imports()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[OZSpeech] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for OZSpeech adversary.")

        usable_samples: List["ZeroShotSample"] = []
        prompt_paths: List[Path] = []
        for sample in samples:
            prompt_path = self._resolve_prompt_path(sample)
            if prompt_path is None:
                if self.logger:
                    self.logger.warning(
                        "[OZSpeech] Sample %d missing prompt audio; skipping.",
                        sample.index,
                    )
                continue
            usable_samples.append(sample)
            prompt_paths.append(prompt_path)

        if not usable_samples:
            raise FileNotFoundError("No prompt audio found for OZSpeech adversary.")

        samples = usable_samples

        self._log_attack_plan("OZSpeech", samples, len(prompt_paths))

        prompt_dir = output_dir / "ozspeech_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        alias_map: dict[Path, Path] = {}
        cached_files: List[Path] = []
        for ref in prompt_paths:
            resolved = ref.resolve()
            if resolved not in alias_map:
                alias_name = f"prompt_{len(alias_map):04d}_{resolved.name}"
                cached_path = prompt_dir / alias_name
                if not cached_path.exists():
                    shutil.copy2(resolved, cached_path)
                alias_map[resolved] = cached_path
            cached_files.append(alias_map[resolved])

        manifest_path = output_dir / "ozspeech_manifest.txt"
        self._build_manifest(manifest_path, cached_files, samples)

        assert self._synthesizer is not None
        generated = self._synthesizer.synthesize(
            manifest_path=manifest_path,
            prompt_dir=prompt_dir,
            output_dir=output_dir,
        )

        synth_dir = output_dir / "synth"
        if not synth_dir.exists():
            raise FileNotFoundError("OZSpeech synthesis completed but no 'synth' directory was created.")

        if not generated:
            generated = list(synth_dir.glob("*.wav"))
        if not generated:
            raise FileNotFoundError("OZSpeech synthesis produced no audio files.")

        copied = []
        for idx, sample in enumerate(samples):
            target_stub = self._target_stub_for_sample(sample, idx)
            src = synth_dir / f"{target_stub}.wav"
            if not src.exists():
                raise FileNotFoundError(f"OZSpeech missing expected output file: {src}")
            speaker_id = str(sample.speaker_id)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_id)
            dest = speaker_dir / self._cloned_filename(sample, idx)
            shutil.copy2(src, dest)
            copied.append(dest)

        try:
            manifest_path.unlink(missing_ok=True)
        except TypeError:
            # Python <3.8 fallback
            if manifest_path.exists():
                manifest_path.unlink()

        self.logger.info("[OZSpeech] Generated %d utterances.", len(copied))
