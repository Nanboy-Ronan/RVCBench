"""Explicit local Transformer assets for Melo's eagerly imported text modules."""
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch


def text_files(entry):
    directory = entry.get('path')
    if not isinstance(directory, str) or not directory:
        raise ValueError('OpenVoice text model requires a resolved local path')
    root = Path(directory).expanduser().resolve()
    load_model = entry.get('load_model', False)
    if not isinstance(load_model, bool):
        raise ValueError('OpenVoice text load_model must be boolean')
    names = ['tokenizer_config.json', 'vocab.txt']
    names += [name for name in ('config.json', 'tokenizer.json', 'special_tokens_map.json', 'added_tokens.json')
              if (root / name).exists()]
    if load_model:
        if 'config.json' not in names:
            names.append('config.json')
        weights = next((name for name in ('model.safetensors', 'pytorch_model.bin')
                        if (root / name).exists()), None)
        if weights is None:
            raise FileNotFoundError(f'OpenVoice text model weights are missing: {root}')
        names.append(weights)
    files = []
    for name in names:
        path = root / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f'OpenVoice text asset is missing or empty: {path}')
        files.append(path)
    return files


@contextmanager
def pinned_text_loading(resources):
    if resources is None:
        yield
        return
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer_loader = AutoTokenizer.from_pretrained
    model_loader = AutoModelForMaskedLM.from_pretrained

    def load(original, identifier, args, kwargs, model=False):
        entry = resources.get(str(identifier))
        if entry is None:
            raise ValueError(f'OpenVoice text asset is not configured: {identifier}')
        text_files(entry)
        if model and not entry.get('load_model', False):
            raise ValueError(f'OpenVoice text weights are not enabled for {identifier}')
        kwargs = dict(kwargs)
        kwargs.pop('revision', None)
        kwargs.update(local_files_only=True, trust_remote_code=False)
        return original(str(Path(entry['path']).expanduser().resolve()), *args, **kwargs)

    def tokenizer(identifier, *args, **kwargs):
        return load(tokenizer_loader, identifier, args, kwargs)

    def model(identifier, *args, **kwargs):
        return load(model_loader, identifier, args, kwargs, model=True)

    with patch.object(AutoTokenizer, 'from_pretrained', side_effect=tokenizer), \
         patch.object(AutoModelForMaskedLM, 'from_pretrained', side_effect=model):
        yield


@contextmanager
def capture_melo_text_modules(owned):
    """Track module objects imported by this serial generator operation."""
    before = {name: module for name, module in sys.modules.items()
              if name.startswith('melo.text.')}
    try:
        yield
    finally:
        for name, module in list(sys.modules.items()):
            if name.startswith('melo.text.') and module is not before.get(name):
                owned[name] = module


def release_melo_text_models(owned):
    """Release native global model caches on the owned module objects only."""
    import torch
    for module in owned.values():
        if isinstance(getattr(module, 'model', None), torch.nn.Module):
            module.model = None
        models = getattr(module, 'models', None)
        if isinstance(models, dict):
            for name, model in list(models.items()):
                if isinstance(model, torch.nn.Module):
                    del models[name]
    owned.clear()
