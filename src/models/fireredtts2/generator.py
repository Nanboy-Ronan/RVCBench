"""FireRedTTS2 generator wrapper."""

from __future__ import annotations

import gc
import importlib
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class FireRedTTS2GeneratorConfig:
    """Configuration options for FireRedTTS2 voice cloning."""

    pretrained_dir: str
    code_path: Optional[str] = None
    gen_type: str = "monologue"
    use_bf16: bool = False
    temperature: float = 0.75
    topk: int = 20
    min_token_frames: int = 18
    max_prompt_retries: int = 3


class FireRedTTS2Generator(BaseModel):
    """Thin wrapper around the upstream FireRedTTS2 Python API."""

    def __init__(self, config: FireRedTTS2GeneratorConfig, device: torch.device, logger) -> None:
        materialised_config = replace(config)
        super().__init__(
            model_name_or_path=str(materialised_config.pretrained_dir),
            device=device,
            logger=logger,
        )
        self.config = materialised_config
        self.device = torch.device(device)
        if self.config.gen_type != 'monologue':
            raise ValueError('FireRedTTS2 zero-shot generator requires monologue mode')
        if self.config.max_prompt_retries <= 0 or self.config.min_token_frames < 2:
            raise ValueError('FireRedTTS2 requires positive retries and at least two token frames')
        self._generator = None
        self._spliter_module = None

    def load_model(self) -> None:
        if self._generator is not None:
            return

        self._ensure_pythonpath()
        try:
            module = importlib.import_module("fireredtts2.fireredtts2")
        except ImportError as exc:
            raise ImportError(
                "Missing FireRedTTS2 runtime. Set adversary.code_path to a local FireRedTTS2 checkout "
                "or install the upstream package and dependencies."
            ) from exc

        generator_cls = getattr(module, "FireRedTTS2", None)
        if generator_cls is None:
            raise ImportError("fireredtts2.fireredtts2 does not expose FireRedTTS2.")
        if self.config.code_path:
            root = Path(self.config.code_path).expanduser().resolve()
            if not getattr(module, '__file__', None) or not Path(module.__file__).resolve().is_relative_to(root):
                raise ImportError('Imported FireRedTTS2 runtime is outside configured code_path')

        with self._device_scope(), self._cpu_safe_torch_load():
            self._generator = generator_cls(
                pretrained_dir=str(self.config.pretrained_dir),
                gen_type=str(self.config.gen_type),
                device=str(self.device),
                use_bf16=bool(self.config.use_bf16),
            )
        self.model = getattr(self._generator, "_model", self._generator)
        self._spliter_module = importlib.import_module("fireredtts2.utils.spliter")

    def generate(
        self,
        *,
        text: str,
        prompt_wav: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        if not str(text).strip():
            raise ValueError('FireRedTTS2 requires nonempty target text')
        if bool(prompt_wav) != bool(prompt_text and str(prompt_text).strip()):
            raise ValueError('FireRedTTS2 reference audio requires its actual transcript')
        self.ensure_model()
        assert self._generator is not None

        kwargs = {
            "text": str(text),
            "temperature": float(self.config.temperature),
            "topk": int(self.config.topk),
        }
        if prompt_wav:
            kwargs["prompt_wav"] = str(prompt_wav)
        if prompt_text:
            kwargs["prompt_text"] = str(prompt_text)

        with self._device_scope(), torch.inference_mode():
            if prompt_wav and prompt_text:
                waveform = self._generate_prompted_monologue(
                    text=str(text),
                    prompt_wav=str(prompt_wav),
                    prompt_text=str(prompt_text),
                )
            else:
                waveform = self._generator.generate_monologue(**kwargs)
        if isinstance(waveform, torch.Tensor):
            audio = waveform.detach().cpu().numpy()
        else:
            audio = np.asarray(waveform)
        del waveform
        self._cleanup_after_generation()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError('FireRedTTS2 returned an empty, nonfinite or multiple waveform output')
        # Upstream sample_rate=16000 describes prompt input, not codec output.
        # Official monologue examples save decoded audio at 24000 Hz.
        return audio, 24000

    def close(self) -> None:
        self._generator = None
        self.model = None
        self._spliter_module = None
        self._model_ready = False

    def _device_scope(self):
        return torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()

    def _ensure_pythonpath(self) -> None:
        raw = self.config.code_path
        if raw in (None, ""):
            return

        root = Path(str(raw)).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"FireRedTTS2 code_path not found: {root}")
        root = root.resolve()

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    def _generate_prompted_monologue(self, *, text: str, prompt_wav: str, prompt_text: str):
        assert self._generator is not None
        assert self._spliter_module is not None

        clean_text = getattr(self._spliter_module, "clean_text")
        split_text = getattr(self._spliter_module, "split_text")

        prompt_text = clean_text(text=prompt_text)
        text = clean_text(text=text)
        text_list = split_text(text=text, length=400)
        audio_list = []

        for chunk in text_list:
            chunk = clean_text(text=chunk)
            input_text = prompt_text[:-1] + "," + chunk if prompt_text else chunk
            prompt_segment = self._generator.prepare_prompt(
                text=input_text,
                speaker="[S1]",
                audio_path=prompt_wav,
            )
            context = [prompt_segment]

            gen_tokens = None
            for _ in range(max(1, int(self.config.max_prompt_retries))):
                candidate = self._generator.generate_single(
                    context=context,
                    temperature=float(self.config.temperature),
                    topk=int(self.config.topk),
                )
                if candidate.shape[2] > int(self.config.min_token_frames):
                    gen_tokens = candidate
                    break
                gen_tokens = candidate

            assert gen_tokens is not None
            if gen_tokens.shape[2] <= int(self.config.min_token_frames):
                raise ValueError('FireRedTTS2 exhausted prompt retries without sufficient token frames')
            cut = 2 if gen_tokens.shape[2] > 2 else 0
            trimmed_tokens = gen_tokens[:, :, cut:]
            audio = self._generator._audio_tokenizer.decode(trimmed_tokens).squeeze(0).squeeze(0)
            audio_list.append(audio.unsqueeze(0))
            del prompt_segment, context, gen_tokens, trimmed_tokens, audio

        return torch.cat(audio_list, dim=1)

    def _cleanup_after_generation(self) -> None:
        gc.collect()

    @contextmanager
    def _cpu_safe_torch_load(self):
        if str(self.device) != "cpu":
            yield
            return

        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device("cpu"))
            return original_torch_load(*args, **kwargs)

        torch.load = patched_torch_load
        try:
            yield
        finally:
            torch.load = original_torch_load
