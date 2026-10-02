"""Check the native converter's checkpoint loading without changing upstream code."""
from contextlib import contextmanager
from unittest.mock import patch

import torch


@contextmanager
def checked_converter_loading(model, receipts):
    if not isinstance(model, torch.nn.Module):
        raise TypeError('OpenVoice converter must expose a Torch model')
    original = model.load_state_dict

    def checked(state, *args, **kwargs):
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(f'OpenVoice converter checkpoint has a nonfinite tensor: {name}')
        # Override native strict=False while preserving other Torch arguments.
        if args:
            args = (True, *args[1:])
            kwargs.pop('strict', None)
        else:
            kwargs['strict'] = True
        result = original(state, *args, **kwargs)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError('OpenVoice converter checkpoint contains incompatible state')
        receipts.append({'checkpoint_keys': len(state), 'strict': True,
                         'finite_tensors': True, 'missing_keys': [], 'unexpected_keys': []})
        return result

    with patch.object(model, 'load_state_dict', checked):
        yield
