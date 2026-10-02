"""Qwen3-Omni generator wrapper."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from rvcbench.models.model import BaseModel


DEFAULT_SYSTEM_PROMPT = (
    "You are a virtual voice assistant with no gender or age.\n"
    "You are communicating with the user.\n"
    "In user messages, 'I/me/my/we/our' refer to the user and 'you/your' refer to the assistant.\n"
    "In your replies, address the user as 'you/your' and yourself as 'I/me/my'; never mirror the user's pronouns.\n"
    "Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\n"
    "Interact with users using short (no more than 50 words), brief, straightforward language, maintaining a natural tone.\n"
    "Never use formal phrasing, mechanical expressions, bullet points, or overly structured language.\n"
    "Your output must consist only of the spoken content you want the user to hear.\n"
    "Do not include any descriptions of actions, emotions, sounds, or voice changes.\n"
    "Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions.\n"
    "You must answer users' audio or text questions and should communicate in the same language as the user unless they request otherwise.\n"
    "When uncertain, use brief questions to guide the user to continue the conversation.\n"
    "Keep replies concise and conversational, as if talking face-to-face."
)


@dataclass
class Qwen3OmniGeneratorConfig:
    """Configuration options for running the Qwen3-Omni generator."""

    checkpoint_path: str
    torch_dtype: Optional[str] = "auto"
    device_map: Optional[str] = "auto"
    use_flash_attn2: bool = False
    attn_implementation: Optional[str] = None
    return_audio: bool = True
    use_audio_in_video: bool = True
    thinker_max_new_tokens: int = 2048
    thinker_temperature: float = 0.7
    thinker_top_p: float = 0.8
    thinker_top_k: int = 20
    thinker_do_sample: bool = True
    speaker: Optional[str] = None
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    seed: Optional[int] = None
    sample_rate: Optional[int] = None
    clean_up_tokenization_spaces: bool = False


class Qwen3OmniGenerator(BaseModel):
    """Single-request Transformers API with explicit waveform and device contracts."""

    def __init__(self, config, device, logger):
        super().__init__(model_name_or_path=config.checkpoint_path, device=device, logger=logger)
        self.config = replace(config)
        if not str(config.checkpoint_path).strip():
            raise ValueError('Qwen3-Omni checkpoint_path must be provided')
        if not config.return_audio:
            raise ValueError('The voice benchmark requires return_audio=True')
        self._model = self._processor = self._process_mm_info = None
        self.last_generated_text = None

    def load_model(self):
        if self._model is not None:
            return
        try:
            transformers = importlib.import_module('transformers')
            model_cls = transformers.Qwen3OmniMoeForConditionalGeneration
            processor_cls = transformers.Qwen3OmniMoeProcessor
            utilities = importlib.import_module('qwen_omni_utils')
        except (ImportError, AttributeError) as exc:
            raise ImportError('Qwen3-Omni requires Transformers with Qwen3OmniMoe support '
                              'and qwen-omni-utils') from exc
        kwargs = {}
        dtype = str(self.config.torch_dtype or '').lower()
        dtypes = {'fp32': torch.float32, 'float32': torch.float32, 'float': torch.float32,
                  'single': torch.float32, 'fp16': torch.float16, 'float16': torch.float16,
                  'half': torch.float16, 'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16}
        if dtype not in ('', 'none', 'auto') and dtype not in dtypes:
            raise ValueError(f'Unsupported torch_dtype {self.config.torch_dtype!r}')
        if dtype == 'auto':
            kwargs['torch_dtype'] = 'auto'
        elif dtype in dtypes:
            kwargs['torch_dtype'] = dtypes[dtype]
        if self.config.device_map not in (None, ''):
            kwargs['device_map'] = self.config.device_map
        attention = self.config.attn_implementation
        if attention is None and self.config.use_flash_attn2:
            attention = 'flash_attention_2'
        if attention:
            kwargs['attn_implementation'] = attention
        try:
            self._model = model_cls.from_pretrained(self.config.checkpoint_path, **kwargs)
            if 'device_map' not in kwargs:
                self._model.to(torch.device(self.device))
            self._model.eval()
            self._processor = processor_cls.from_pretrained(self.config.checkpoint_path)
            self._process_mm_info = utilities.process_mm_info
            self.model = self._model
        except BaseException:
            self.close()
            raise

    def generate(self, messages: List[dict[str, Any]], sample_index: int = 0) -> Tuple[np.ndarray, int]:
        self.ensure_model()
        if self.config.seed is not None:
            from rvcbench.utils.seeding import configure_seeds
            configure_seeds(int(self.config.seed) + int(sample_index),
                            deterministic=False, disable_benchmark=False)
        messages = list(messages)
        if self.config.system_prompt and not any(m.get('role') == 'system' for m in messages):
            messages.insert(0, {'role': 'system', 'content': [
                {'type': 'text', 'text': self.config.system_prompt}]})
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._process_mm_info(
            messages, use_audio_in_video=self.config.use_audio_in_video)
        inputs = self._processor(text=text, audio=audios, images=images, videos=videos,
                                 return_tensors='pt', padding=True,
                                 use_audio_in_video=self.config.use_audio_in_video)
        # BatchFeature.to(dtype) casts floating tensors only, preserving token IDs.
        inputs = inputs.to(getattr(self._model, 'device', torch.device(self.device)))
        dtype = getattr(self._model, 'dtype', None)
        if dtype is not None:
            inputs = inputs.to(dtype)
        kwargs = dict(thinker_return_dict_in_generate=True,
                      thinker_max_new_tokens=int(self.config.thinker_max_new_tokens),
                      thinker_do_sample=bool(self.config.thinker_do_sample),
                      thinker_temperature=float(self.config.thinker_temperature),
                      thinker_top_p=float(self.config.thinker_top_p),
                      thinker_top_k=int(self.config.thinker_top_k),
                      use_audio_in_video=bool(self.config.use_audio_in_video), return_audio=True)
        if self.config.speaker:
            kwargs['speaker'] = self.config.speaker
        with torch.inference_mode():
            text_ids, audio = self._model.generate(**inputs, **kwargs)
        sequences = getattr(text_ids, 'sequences', text_ids)
        decoded = self._processor.batch_decode(
            sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=self.config.clean_up_tokenization_spaces)
        self.last_generated_text = decoded[0] if decoded else ''
        if audio is None:
            raise RuntimeError('Qwen3-Omni returned no audio')
        if isinstance(audio, (list, tuple)):
            if len(audio) != 1:
                raise RuntimeError('Expected one waveform for the single benchmark request')
            audio = audio[0]
        waveform = torch.as_tensor(audio).detach().cpu().float().reshape(-1).numpy()
        if not waveform.size or not np.isfinite(waveform).all():
            raise RuntimeError('Qwen3-Omni returned empty or nonfinite audio')
        # The processor's feature extractor rate is for INPUT audio (16 kHz).
        # Qwen3-Omni Code2Wav produces OUTPUT audio at 24 kHz.
        rate = int(self.config.sample_rate if self.config.sample_rate is not None else 24000)
        if rate <= 0:
            raise ValueError('Qwen3-Omni output sample_rate must be positive')
        return waveform, rate

    def close(self):
        self._model = self.model = self._processor = self._process_mm_info = None
        self._model_ready = False
        self.last_generated_text = None
