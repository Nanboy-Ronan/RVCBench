"""MGM-Omni generator wrapper for voice cloning."""

from __future__ import annotations

import sys
import contextlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class MGMOmniGeneratorConfig:
    """Configuration options for running MGM-Omni voice cloning."""

    repo_root: str
    checkpoint_path: str
    speechlm_checkpoint_path: Optional[str] = None
    cosyvoice_path: Optional[str] = None
    device_map: str = "auto"
    load_8bit: bool = False
    load_4bit: bool = False
    use_flash_attn: bool = True
    do_sample: bool = True
    temperature: float = 0.3
    max_new_tokens: int = 4096
    enable_vision_tower: bool = False
    enable_speech_tower: bool = False


class MGMOmniGenerator(BaseModel):
    """Thin wrapper around MGM-Omni TTS inference utilities."""

    def __init__(
        self,
        config: MGMOmniGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        materialised_config = replace(config)
        super().__init__(
            model_name_or_path=str(materialised_config.checkpoint_path),
            device=device,
            logger=logger,
        )
        self.config = materialised_config

        self.repo_root = Path(str(self.config.repo_root)).expanduser()
        if self.repo_root.exists():
            self.repo_root = self.repo_root.resolve()
        self._tokenizer = None
        self._model = None
        self._whisper_model = None

        self._validate_repo_root()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        self._ensure_repo_on_path()

        from mgm.model.builder import load_pretrained_model
        from mgm.utils import disable_torch_init

        disable_torch_init()

        device_str = self._device_string()
        model_path = str(self.config.checkpoint_path)
        speechlm_path = self.config.speechlm_checkpoint_path
        cosyvoice_path = self._resolve_cosyvoice_path(self.config.cosyvoice_path)

        self.logger.info("[MGM-Omni] Loading model from %s", model_path)

        with self._working_dir():
            loaded = load_pretrained_model(
                model_path,
                load_8bit=bool(self.config.load_8bit),
                load_4bit=bool(self.config.load_4bit),
                device_map=self.config.device_map,
                device=device_str,
                use_flash_attn=bool(self.config.use_flash_attn),
                speechlm_path=speechlm_path,
                cosyvoice_path=cosyvoice_path,
            )

        if isinstance(loaded, tuple) and len(loaded) == 2:
            self._tokenizer, self._model = loaded
        else:
            raise RuntimeError(
                "MGM-Omni voice cloning expects an MGM-Omni-TTS checkpoint. "
                "Point checkpoint_path at MGM-Omni-TTS-2B-0927 (or equivalent)."
            )

        self.model = self._model
        self.logger.info("[MGM-Omni] Model ready on %s", device_str)

    def generate(
        self,
        *,
        instruction_text: str,
        reference_audio_path: Path,
        reference_transcript: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        self.ensure_model()
        assert self._model is not None and self._tokenizer is not None

        from mgm.constants import AUDIO_END, AUDIO_START, DEFAULT_SPEECH_TOKEN
        from mgm.conversation import conv_templates
        from mgm.mm_utils import tokenizer_speech_token
        from mgm.serve.utils import whispers_asr

        instruction_text = (instruction_text or "").strip()
        if not instruction_text:
            raise ValueError("MGM-Omni requires non-empty instruction text.")

        reference_audio_path = Path(reference_audio_path)
        if not reference_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

        model_device = getattr(self._model, "device", self.device)

        audio_refer = self._load_reference_audio(reference_audio_path, device=model_device)
        transcript = (reference_transcript or "").strip()
        if not transcript:
            transcript = self._transcribe_reference_audio(reference_audio_path, whispers_asr)

        input_ids_refer = torch.tensor(
            self._tokenizer(transcript)["input_ids"], dtype=torch.long
        ).unsqueeze(0).to(model_device)

        pre_prompt_cn = "使用参考音频中听到的语气回答。"
        pre_prompt_en = "Respond with the tone of the reference audio clip."
        has_chinese = any("\u4e00" <= char <= "\u9fff" for char in transcript)
        pre_prompt = pre_prompt_cn if has_chinese else pre_prompt_en

        conv = conv_templates["qwen2vl"].copy()
        roles = conv.roles
        prompt_inp = pre_prompt + AUDIO_START + DEFAULT_SPEECH_TOKEN + AUDIO_END + "\n"
        prompt_out = AUDIO_START + instruction_text
        conv.append_message(roles[0], prompt_inp)
        conv.append_message(roles[1], prompt_out)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_speech_token(prompt, self._tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .to(model_device)
        )

        effective_temperature = float(temperature) if temperature is not None else float(self.config.temperature)
        effective_max_tokens = int(max_new_tokens) if max_new_tokens is not None else int(self.config.max_new_tokens)
        do_sample = bool(self.config.do_sample) and effective_temperature > 0

        with self._working_dir():
            with torch.inference_mode():
                outputs = self._model.generate(
                    input_ids.clone(),
                    input_ids_refer=input_ids_refer,
                    audio_refer=audio_refer,
                    do_sample=do_sample,
                    temperature=effective_temperature,
                    max_new_tokens=effective_max_tokens,
                    bos_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=[self._tokenizer.eos_token_id],
                    pad_token_id=self._tokenizer.pad_token_id,
                    tokenizer=self._tokenizer,
                    use_cache=True,
                )

        speech_ids, audio = self._unpack_generate_outputs(outputs)
        del speech_ids

        audio = self._to_numpy_audio(audio)
        return audio, 24000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_repo_root(self) -> None:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"MGM-Omni repo_root not found: {self.repo_root}")

    def _ensure_repo_on_path(self) -> None:
        candidate_paths = [self.repo_root]
        third_party_root = self.repo_root / "third_party"
        if third_party_root.exists():
            candidate_paths.append(third_party_root)
            matcha_root = third_party_root / "Matcha-TTS"
            if matcha_root.exists():
                candidate_paths.append(matcha_root)

        for path in candidate_paths:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    @contextlib.contextmanager
    def _working_dir(self):
        original = Path.cwd()
        try:
            if self.repo_root.exists():
                os.chdir(self.repo_root)
            yield
        finally:
            os.chdir(original)

    def _resolve_cosyvoice_path(self, raw_value: Optional[str]) -> Optional[str]:
        if raw_value is None:
            return None
        raw = str(raw_value).strip()
        if not raw or raw.lower() in {"none", "null"}:
            return None
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return str(candidate)
        cosyvoice_yaml = candidate / "cosyvoice2.yaml"
        if cosyvoice_yaml.exists():
            return str(candidate)
        self.logger.warning(
            "[MGM-Omni] cosyvoice_path '%s' not found; falling back to MGM-Omni defaults.",
            raw,
        )
        return None

    def _device_string(self) -> str:
        if isinstance(self.device, torch.device):
            return str(self.device)
        return str(self.device)

    def _load_reference_audio(self, path: Path, device: torch.device) -> torch.Tensor:
        import librosa

        audio, _ = librosa.load(str(path), sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        return audio_tensor

    def _transcribe_reference_audio(self, path: Path, whispers_asr) -> str:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "whisper is required to transcribe reference audio when no transcript is provided."
            ) from exc

        if self._whisper_model is None:
            self._whisper_model = whisper.load_model("large-v3")

        transcript = whispers_asr(self._whisper_model, str(path))
        transcript = (transcript or "").strip()
        if not transcript:
            raise RuntimeError("Whisper failed to return a transcript for the reference audio.")
        return transcript

    def _unpack_generate_outputs(self, outputs):
        if isinstance(outputs, tuple):
            if len(outputs) >= 2:
                return outputs[0], outputs[1]
        raise RuntimeError("MGM-Omni did not return speech tokens and audio.")

    def _to_numpy_audio(self, audio) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio)
        audio = np.clip(audio, -1.0, 1.0)
        return audio
