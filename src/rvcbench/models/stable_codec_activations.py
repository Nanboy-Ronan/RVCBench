"""Owned codec activation replacement with unchanged trained parameters."""
import torch
from torch import nn


class EagerSnake1d(nn.Module):
    """DAC's exact Snake expression without profiling-dependent JIT fusion."""
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


def stabilize_codec_snake(model, *, snake_type=None):
    """Replace owned Snake modules, preserving Parameter objects and state keys.

    The module hierarchy belongs to the caller; no shared class or function is
    patched. Explicit opt-in defines a distinct numerical execution variant.
    """
    if snake_type is None:
        from dac.nn.layers import Snake1d
        snake_type = Snake1d
    replaced = 0
    for name, child in list(model.named_children()):
        if isinstance(child, snake_type):
            if (set(child.state_dict()) != {'alpha'} or
                    not isinstance(child.alpha, nn.Parameter)):
                raise ValueError('Unsupported Snake module state; refusing to discard codec weights')
            replacement = EagerSnake1d(child.alpha)
            replacement.train(child.training)
            setattr(model, name, replacement)
            replaced += 1
        else:
            replaced += stabilize_codec_snake(child, snake_type=snake_type)
    return replaced
