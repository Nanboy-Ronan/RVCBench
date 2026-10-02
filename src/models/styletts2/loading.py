"""Validate the complete state used by native StyleTTS2 inference."""
from collections.abc import Mapping

import torch


def load_model_state(model, checkpoint):
    """Normalize DataParallel keys and require complete, finite module state.

    A training checkpoint can contain modules absent from the constructed
    inference model. Record those names; they cannot supply missing inference
    weights. Every constructed module must be present and load strictly.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError('StyleTTS2 checkpoint net must be a module mapping')
    prepared = {}
    for name, module in model.items():
        if name not in checkpoint:
            raise ValueError(f'StyleTTS2 checkpoint is missing module: {name}')
        state = checkpoint[name]
        if not isinstance(state, Mapping):
            raise TypeError(f'StyleTTS2 checkpoint module must be a state mapping: {name}')
        normalized = {}
        for key, value in state.items():
            if not isinstance(key, str) or not torch.is_tensor(value):
                raise TypeError(f'StyleTTS2 checkpoint has invalid tensor state: {name}')
            clean = key.removeprefix('module.')
            if clean in normalized:
                raise ValueError(f'StyleTTS2 checkpoint has duplicate normalized key: {name}.{clean}')
            if not torch.isfinite(value).all():
                raise ValueError(f'StyleTTS2 checkpoint has nonfinite tensor: {name}.{clean}')
            normalized[clean] = value
        expected = module.state_dict()
        missing = sorted(set(expected) - set(normalized))
        unexpected = sorted(set(normalized) - set(expected))
        if missing or unexpected:
            raise ValueError(f'StyleTTS2 incompatible module {name}: missing={missing}, unexpected={unexpected}')
        for key, value in normalized.items():
            if value.shape != expected[key].shape:
                raise ValueError(f'StyleTTS2 checkpoint shape mismatch: {name}.{key}')
        prepared[name] = normalized
    for name, state in prepared.items():
        model[name].load_state_dict(state, strict=True)
    return {
        'strict': True,
        'finite_tensors': True,
        'modules': {name: {'checkpoint_keys': len(state), 'missing_keys': [], 'unexpected_keys': []}
                    for name, state in prepared.items()},
        'unused_checkpoint_modules': sorted(set(checkpoint) - set(model)),
    }
