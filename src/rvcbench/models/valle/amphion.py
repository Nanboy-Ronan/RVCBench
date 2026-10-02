"""Explicit Amphion VALLE implementation using its released config and symbols."""
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from .generator import VallEGenerator, VallEGeneratorConfig


@dataclass
class AmphionVallEGeneratorConfig(VallEGeneratorConfig):
    config_path: str | None = None


def read_config(path, root, merge, active=()):
    import json5
    path = Path(path).resolve()
    if path in active:
        raise ValueError('Cyclic Amphion base_config references')
    config = json5.loads(path.read_text())
    if config.get('base_config'):
        base = Path(config['base_config'])
        if not base.is_absolute():
            base = Path(root) / base
        config = merge(read_config(base, root, merge, (*active, path)), config)
    return config


class AmphionVallEGenerator(VallEGenerator):
    def load_model(self):
        root = Path(self.config.code_path).expanduser().resolve()
        weights = Path(self.config.checkpoint_path).expanduser().resolve()
        config_path = Path(self.config.config_path or '').expanduser()
        symbols = Path(self.config.text_tokens_path or '').expanduser()
        for path in (weights, config_path, symbols):
            if not path.is_file():
                raise FileNotFoundError(f'Amphion VALLE requires checkpoint, config and symbols files: {path}')
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            modules = {name: importlib.import_module(name) for name in (
                'utils.util', 'utils.tokenizer', 'models.tts.valle.valle',
                'processors.phone_extractor', 'text.text_token_collation')}
            for module in modules.values():
                if not Path(module.__file__).resolve().is_relative_to(root):
                    raise ImportError('Imported Amphion runtime is outside configured code_path')
            util = modules['utils.util']
            cfg = util.JsonHParams(**read_config(config_path, root, util.override_config))
            with self._scope():
                model = modules['models.tts.valle.valle'].VALLE(cfg.model)
                # Released pytorch_model.bin is an unwrapped tensor state dict.
                state = torch.load(weights, map_location='cpu', weights_only=True)
                model.load_state_dict(state, strict=True)
                model.to(self.device).eval()
                codec = modules['utils.tokenizer'].AudioTokenizer(device=self.device)
                if int(codec.sample_rate) != 24000:
                    raise ValueError('Amphion VALLE requires 24kHz EnCodec')
                phone = modules['processors.phone_extractor'].phoneExtractor(cfg)
                collater = modules['text.text_token_collation'].phoneIDCollation(
                    cfg, symbols_dict_file=str(symbols.resolve()))
            def collate(texts):
                ids = collater.get_phone_id_sequence(cfg, texts[0])
                return torch.from_numpy(np.array([ids])), torch.IntTensor([len(ids)])
            self.model = self._model = model
            self._audio_tokenizer, self._text_tokenizer, self._text_collater = codec, phone, collate
            self._data = SimpleNamespace(
                tokenize_text=lambda tokenizer, *, text: tokenizer.extract_phone(text),
                tokenize_audio=modules['utils.tokenizer'].tokenize_audio)
        except Exception:
            self.close()
            raise
