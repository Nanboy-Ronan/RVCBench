import importlib
import os
import sys
from pathlib import Path
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from scipy.signal import resample_poly

from .base_adversary import BaseAdversary


class BertVits2ZeroShotAdversary(BaseAdversary):
    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger

        self.code_path = Path(to_absolute_path(self.config.code_path)).resolve()
        self.config_path = Path(to_absolute_path(self.config.config_path)).resolve()
        self.generator_checkpoint = Path(to_absolute_path(self.config.generator_checkpoint)).resolve()
        default_yml = self.code_path / "config.yml"
        self.yml_config_path = Path(
            to_absolute_path(self.config.get("yml_config_path", str(default_yml)))
        ).resolve()

        self.language = str(self.config.get("language", "EN")).upper()
        self.emotion = int(self.config.get("emotion", -1))
        self.sdp_ratio = float(self.config.get("sdp_ratio", 0.2))
        self.noise_scale = float(self.config.get("noise_scale", 0.2))
        self.noise_scale_w = float(self.config.get("noise_scale_w", 0.9))
        self.length_scale = float(self.config.get("length_scale", 1.0))
        self.reference_sr = int(self.config.get("reference_audio_sr", 48000))
        self.reference_assignment = str(self.config.get("reference_assignment", "round_robin")).lower()
        self.max_samples = self.config.get("max_samples")
        self.default_target_text = str(
            self.config.get(
                "default_target_text",
                "This is a synthesized sample for evaluation.",
            )
        )

        self._imports_loaded = False
        self._infer_fn = None
        self._hps = None
        self._net_g = None
        self._device_str = None

    def _ensure_imports(self):
        if self._imports_loaded:
            return

        prev_cwd = Path.cwd()
        try:
            if str(self.code_path) not in sys.path:
                sys.path.insert(0, str(self.code_path))

            config_flag = "--yml_config"
            config_value = str(self.yml_config_path)
            if config_flag not in sys.argv:
                sys.argv.extend([config_flag, config_value])

            os.chdir(self.code_path)

            utils_mod = importlib.import_module("utils")
            infer_mod = importlib.import_module("infer")
        finally:
            os.chdir(prev_cwd)

        self._infer_fn = infer_mod.infer
        latest_version = getattr(infer_mod, "latest_version", "2.3")
        get_net_g = infer_mod.get_net_g

        self._hps = utils_mod.get_hparams_from_file(str(self.config_path))

        class _SpeakerLookup(dict):
            def __getitem__(self, key):
                key_str = str(key)
                if key_str not in self:
                    try:
                        value = int(key)
                    except (TypeError, ValueError):
                        value = int(key_str)
                    self[key_str] = value
                return dict.__getitem__(self, key_str)

        speaker_map = getattr(self._hps.data, "spk2id", None)
        if speaker_map is None:
            self.logger.warning("hps.data.spk2id missing; defaulting to identity mapping for speaker ids")
            self._hps.data.spk2id = _SpeakerLookup()
        else:
            base_map = {}
            if isinstance(speaker_map, dict):
                base_map = speaker_map
            else:
                try:
                    base_map = dict(speaker_map)
                except TypeError:
                    try:
                        base_map = OmegaConf.to_container(speaker_map, resolve=True)
                    except Exception:  # pragma: no cover - defensive fallback
                        base_map = {}
            cleaned_map = {}
            for key, value in base_map.items():
                try:
                    cleaned_map[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
            self._hps.data.spk2id = _SpeakerLookup(cleaned_map)

        version = getattr(self._hps, "version", latest_version)

        if self.device.type == "cuda":
            index = 0 if self.device.index is None else self.device.index
            self._device_str = f"cuda:{index}"
        else:
            self._device_str = "cpu"

        self._net_g = get_net_g(str(self.generator_checkpoint), version, self._device_str, self._hps)
        self._net_g.eval()

        self._imports_loaded = True

    def _load_reference_audio(self, path: Path) -> np.ndarray:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.reference_sr:
            audio = resample_poly(audio, self.reference_sr, sr)
        return np.asarray(audio, dtype=np.float32)

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._ensure_imports()

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = output_dir.resolve()
        self._init_synthesis_timings(output_dir)

        max_samples = None
        if self.max_samples is not None:
            try:
                max_samples = int(self.max_samples)
            except (TypeError, ValueError):
                self.logger.warning(
                    "[BERT-VITS2] Invalid max_samples '%s'; processing full dataset.",
                    self.max_samples,
                )

        samples = dataset.get_zero_shot_samples(max_samples=max_samples)
        if not samples:
            raise RuntimeError("No zero-shot samples available for BERT-VITS2 adversary.")

        prompt_count = self._count_available_prompts(samples)

        self._log_attack_plan("BertVITS2", samples, prompt_count)

        sampling_rate = getattr(self._hps.data, "sampling_rate", 24000)

        for idx, sample in enumerate(samples):
            reference_path = self._resolve_prompt_path(sample)
            if reference_path is None:
                if self.logger:
                    self.logger.warning(
                        "[BERT-VITS2] Sample %d missing prompt audio; skipping.",
                        idx,
                    )
                continue
            reference_audio = self._load_reference_audio(reference_path)
            target_path = sample.target_path
            speaker_identifier = sample.speaker_id
            if target_path is not None and target_path.parent.name:
                speaker_identifier = target_path.parent.name
            speaker_identifier = str(speaker_identifier)
            speaker_dir = self._speaker_output_dir(output_dir, speaker_identifier)

            text = (sample.prompt_text or "").strip()
            if not text:
                text = (sample.target_text or "").strip()
            if not text:
                text = self.default_target_text

            self._log_clone_request(
                "BertVITS2",
                idx,
                len(samples),
                speaker_identifier,
                reference_path,
                text,
            )

            prev_cwd = Path.cwd()
            try:
                os.chdir(self.code_path)
                synth_start = time.perf_counter()
                audio = self._infer_fn(
                    text,
                    emotion=self.emotion,
                    sdp_ratio=self.sdp_ratio,
                    noise_scale=self.noise_scale,
                    noise_scale_w=self.noise_scale_w,
                    length_scale=self.length_scale,
                    sid=int(sample.speaker_id),
                    language=self.language,
                    hps=self._hps,
                    net_g=self._net_g,
                    device=self._device_str,
                    reference_audio=reference_audio,
                )
                synth_elapsed = time.perf_counter() - synth_start
            finally:
                os.chdir(prev_cwd)

            output_name = self._cloned_filename(sample, idx)
            dest_path = speaker_dir / output_name
            sf.write(dest_path, np.asarray(audio, dtype=np.float32), sampling_rate)
            self._record_synthesis_timing(dest_path, synth_elapsed)
            self.logger.debug(
                "Generated zero-shot audio %s using reference %s", dest_path.name, reference_path.name
            )

        self._flush_synthesis_timings()
