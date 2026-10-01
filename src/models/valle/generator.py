"""Strict bridge to lifeiteng/vall-e; Amphion and VALL-E-X are separate runtimes."""
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class VallEGeneratorConfig:
    code_path: Path
    checkpoint_path: Path
    text_extractor: str = 'espeak'
    top_k: int = -100
    temperature: float = 1.0
    output_sample_rate: int = 24000
    text_tokens_path: str | None = None


class VallEGenerator(BaseModel):
    def __init__(self, config, device, logger):
        super().__init__(model_name_or_path=str(config.checkpoint_path), device=device, logger=logger)
        self.config = config
        self.device = torch.device(device)
        self._model = self._audio_tokenizer = self._text_tokenizer = self._text_collater = None
        self._data = None
        if config.output_sample_rate != 24000:
            raise ValueError('lifeiteng VALL-E uses the 24kHz EnCodec codec')

    def _scope(self):
        return torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()

    def load_model(self):
        root = Path(self.config.code_path).expanduser().resolve()
        checkpoint_path = Path(self.config.checkpoint_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'VALL-E code_path not found: {root}')
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'lifeiteng VALL-E checkpoint not found: {checkpoint_path}; '
                                    'Amphion and VALL-E-X checkpoints are not compatible replacements')
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            data = importlib.import_module('valle.data')
            models = importlib.import_module('valle.models')
            for module in (data, models):
                if not Path(module.__file__).resolve().is_relative_to(root):
                    raise ImportError('Imported VALL-E runtime is outside configured code_path')
            from icefall.utils import AttributeDict
            with self._scope():
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if not isinstance(checkpoint, dict) or 'model' not in checkpoint or 'text_tokens' not in checkpoint:
                    raise ValueError('Expected lifeiteng checkpoint with model state and text_tokens metadata')
                args = AttributeDict(checkpoint)
                tokens = Path(self.config.text_tokens_path or args.text_tokens).expanduser()
                if not tokens.is_absolute():
                    candidates = (root / tokens, checkpoint_path.parent / tokens, Path.cwd() / tokens)
                    tokens = next((p for p in candidates if p.is_file()), candidates[0])
                if not tokens.is_file():
                    raise FileNotFoundError(f'VALL-E text token vocabulary not found: {tokens}')
                model = models.get_model(args)
                model.load_state_dict(checkpoint['model'], strict=True)
                model.to(self.device).eval()
                codec = data.AudioTokenizer(device=self.device)
                if int(codec.sample_rate) != 24000:
                    raise ValueError('Unexpected VALL-E codec output sample rate')
                tokenizer = data.TextTokenizer(backend=self.config.text_extractor)
                collater = data.get_text_token_collater(str(tokens.resolve()))
            self._model = self.model = model
            self._audio_tokenizer, self._text_tokenizer, self._text_collater = codec, tokenizer, collater
            self._data = data
        except Exception:
            self.close()
            raise

    def generate(self, *, text, prompt_audio, prompt_text, sample_index=0):
        # The benchmark seeds global RNGs with run seed + original source index
        # before this serial request. Upstream inference exposes no seed argument.
        del sample_index
        if not str(text or '').strip() or not str(prompt_text or '').strip():
            raise ValueError('VALL-E requires target text and the actual reference transcript')
        prompt_audio = Path(prompt_audio)
        if not prompt_audio.is_file():
            raise FileNotFoundError(f'VALL-E reference audio not found: {prompt_audio}')
        self.ensure_model()
        with self._scope(), torch.inference_mode():
            text_tokens, lengths = self._text_collater([self._data.tokenize_text(
                self._text_tokenizer, text=f'{prompt_text.strip()} {text.strip()}')])
            _, enroll = self._text_collater([self._data.tokenize_text(self._text_tokenizer, text=prompt_text.strip())])
            prompt = self._data.tokenize_audio(self._audio_tokenizer, str(prompt_audio))[0][0].transpose(2, 1)
            codes = self._model.inference(text_tokens.to(self.device), lengths.to(self.device),
                prompt.to(self.device), enroll_x_lens=enroll.to(self.device),
                top_k=int(self.config.top_k), temperature=float(self.config.temperature))
            if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] == 0:
                raise ValueError('VALL-E returned empty or malformed codec tokens')
            audio = self._audio_tokenizer.decode([(codes.transpose(2, 1), None)])
        if not isinstance(audio, torch.Tensor):
            raise TypeError('VALL-E codec returned an unexpected waveform type')
        audio = audio.detach().cpu().numpy()
        if audio.ndim == 3 and audio.shape[:2] == (1, 1):
            audio = audio[0, 0]
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError('VALL-E returned an empty, nonfinite or multiple waveform output')
        return audio.astype(np.float32), 24000

    def close(self):
        self._model = self.model = self._audio_tokenizer = self._text_tokenizer = self._text_collater = None
        self._data = None
        self._model_ready = False
