"""Scoped ownership of the official Kimi-Audio inference runtime."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from src.models.model import BaseModel
from src.utils.seeding import configure_seeds


@dataclass
class KimiAudioGeneratorConfig:
    code_path: Path
    model_path: str
    load_detokenizer: bool = True
    audio_temperature: float = 0.8
    audio_top_k: int = 10
    text_temperature: float = 0.0
    text_top_k: int = 5
    audio_repetition_penalty: float = 1.0
    audio_repetition_window_size: int = 64
    text_repetition_penalty: float = 1.0
    text_repetition_window_size: int = 16
    max_new_tokens: int | None = -1
    sample_rate: int = 24000
    seed: int | None = None
    audio_tokenizer_path: str | None = None


class KimiAudioGenerator(BaseModel):
    def __init__(self, config, device, logger):
        super().__init__(model_name_or_path=config.model_path, device=device, logger=logger)
        self.device = torch.device(device)
        self.config = replace(config, code_path=Path(config.code_path).expanduser().resolve())
        self._model = self._kimi_cls = None
        self.last_generated_text = None

    def _ensure_imports(self):
        if self._kimi_cls is not None:
            return
        if not self.config.code_path.is_dir():
            raise FileNotFoundError(f'Kimi Audio code path not found: {self.config.code_path}')
        if str(self.config.code_path) not in sys.path:
            sys.path.insert(0, str(self.config.code_path))
        try:
            module = importlib.import_module('kimia_infer.api.kimia')
        except ImportError as exc:
            raise ImportError('Kimi Audio runtime requires its recursive git submodule and matching Python dependencies') from exc
        if not Path(module.__file__).resolve().is_relative_to(self.config.code_path):
            raise RuntimeError('Kimi Audio runtime was imported from outside configured code_path')
        self._kimi_cls = module.KimiAudio

    @contextmanager
    def _tokenizer_scope(self):
        """Override the constructor's hardcoded auxiliary model only during load."""
        path = self.config.audio_tokenizer_path
        if path is None:
            yield
            return
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'Kimi audio tokenizer path not found: {path}')
        module = importlib.import_module('kimia_infer.api.prompt_manager')
        original = module.Glm4Tokenizer
        def local_tokenizer(_default):
            return original(str(path))
        module.Glm4Tokenizer = local_tokenizer
        try:
            yield
        finally:
            module.Glm4Tokenizer = original

    def load_model(self):
        if self._model is not None:
            return
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Kimi Audio requires a CUDA-capable device')
        if not self.config.load_detokenizer:
            raise ValueError('Kimi Audio audio generation requires load_detokenizer=true')
        if self.config.sample_rate != 24000:
            raise ValueError('Kimi Audio native output sample rate is 24000')
        if self.config.max_new_tokens not in (None, -1):
            raise ValueError('Kimi Audio overrides max_new_tokens for audio output; only the native default -1 is supported')
        self._ensure_imports()
        candidate = Path(self.config.model_path).expanduser()
        nested = self.config.code_path / candidate
        model_path = str(candidate.resolve()) if candidate.exists() else (
            str(nested.resolve()) if nested.exists() else self.config.model_path)
        try:
            with torch.cuda.device(self.device), self._tokenizer_scope():
                self._model = self._kimi_cls(model_path=model_path, load_detokenizer=True)
                self.model = self._model.alm
        except BaseException:
            self.close()
            raise
        self.logger.info('[KimiAudio] Loaded official runtime on %s; native audio token budget is upstream-controlled', self.device)

    def generate(self, messages, sample_index=0):
        self.ensure_model()
        self.last_generated_text = None
        if self.config.seed is not None:
            configure_seeds(int(self.config.seed) + int(sample_index))
        keys = ('audio_temperature', 'audio_top_k', 'text_temperature', 'text_top_k',
                'audio_repetition_penalty', 'audio_repetition_window_size',
                'text_repetition_penalty', 'text_repetition_window_size')
        with torch.cuda.device(self.device), torch.inference_mode():
            wave, text = self._model.generate(messages, output_type='both', max_new_tokens=-1,
                **{key: getattr(self.config, key) for key in keys})
        if wave is None:
            raise RuntimeError('Kimi Audio returned no waveform')
        wave = wave.detach().cpu().float().numpy() if isinstance(wave, torch.Tensor) else np.asarray(wave)
        if wave.ndim > 2 or (wave.ndim == 2 and 1 not in wave.shape):
            raise RuntimeError('Kimi Audio must return one mono waveform')
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if not wave.size or not np.isfinite(wave).all():
            raise RuntimeError('Kimi Audio returned empty or nonfinite waveform')
        self.last_generated_text = text
        return wave, 24000

    def close(self):
        self._model = self.model = self._kimi_cls = None
        self._model_ready = False
        self.last_generated_text = None
