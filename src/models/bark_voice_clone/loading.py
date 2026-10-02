"""Complete Bark checkpoint loading with validated derived attention masks."""
import hashlib
from collections.abc import Mapping

import torch


def load_bark_state(model, state, **kwargs):
    if not isinstance(state, Mapping):
        raise TypeError('Bark checkpoint state must be a mapping')
    expected = model.state_dict()
    parameters = dict(model.named_parameters())
    derived = {}

    def check_mask(key, value):
        if not key.endswith('.attn.bias') or key in parameters:
            return False
        try:
            attention = model.get_submodule(key.removesuffix('.bias'))
        except AttributeError:
            return False
        if type(attention).__name__ not in ('CausalSelfAttention', 'NonCausalSelfAttention'):
            return False
        size = model.config.block_size
        mask = torch.tril(torch.ones(size, size, dtype=value.dtype, device=value.device)).reshape(1, 1, size, size)
        if value.shape != mask.shape or not torch.equal(value, mask):
            raise ValueError(f'Bark checkpoint mismatch: invalid derived causal mask {key}')
        derived[key] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        return True

    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError('Bark checkpoint state must contain named tensors')
        if not torch.isfinite(value).all():
            raise ValueError(f'Bark checkpoint mismatch: nonfinite tensor {key}')
        if key not in expected and check_mask(key, value):
            continue
        if key.endswith('.attn.bias'):
            check_mask(key, value)
        normalized[key] = value
    for key in set(expected) - set(normalized):
        if check_mask(key, expected[key]):
            normalized[key] = expected[key]
    missing = sorted(set(expected) - set(normalized))
    unexpected = sorted(set(normalized) - set(expected))
    if missing or unexpected:
        raise ValueError(f'Bark checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    for key, value in normalized.items():
        if value.shape != expected[key].shape:
            raise ValueError(f'Bark checkpoint mismatch: shape differs for {key}')
    result = torch.nn.Module.load_state_dict(model, normalized, strict=True, **kwargs)
    model._rvcbench_checkpoint_receipt = {
        'strict': True, 'finite_tensors': True, 'checkpoint_keys': len(state),
        'loaded_keys': len(normalized), 'missing_keys': [], 'unexpected_keys': [],
        'validated_derived_masks': derived,
    }
    return result
