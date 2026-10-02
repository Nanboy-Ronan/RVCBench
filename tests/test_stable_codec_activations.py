import torch
import pytest
from torch import nn
from rvcbench.models.stable_codec_activations import EagerSnake1d, stabilize_codec_snake


class SourceSnake(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([[[.7], [1.2]]]))


def test_codec_activation_conversion_preserves_parameters_state_and_other_models():
    codec = nn.Sequential(nn.Sequential(SourceSnake()), nn.Identity()).eval()
    other = nn.Sequential(SourceSnake())
    parameter = codec[0][0].alpha
    state = {k: v.clone() for k, v in codec.state_dict().items()}
    assert stabilize_codec_snake(codec, snake_type=SourceSnake) == 1
    assert isinstance(codec[0][0], EagerSnake1d)
    assert isinstance(other[0], SourceSnake)
    assert codec[0][0].alpha is parameter
    assert not codec[0][0].training
    assert set(codec.state_dict()) == set(state)
    for key, value in codec.state_dict().items():
        assert torch.equal(value, state[key])
    x = torch.linspace(-2, 2, 40).reshape(1, 2, 20)
    expected = x + (parameter + 1e-9).reciprocal() * torch.sin(parameter * x).pow(2)
    assert torch.equal(codec(x), expected)
    assert stabilize_codec_snake(codec, snake_type=SourceSnake) == 0


def test_codec_activation_conversion_rejects_additional_trained_state():
    source = SourceSnake()
    source.bias = nn.Parameter(torch.zeros(2))
    codec = nn.Sequential(source)
    with pytest.raises(ValueError, match='refusing to discard codec weights'):
        stabilize_codec_snake(codec, snake_type=SourceSnake)
    assert codec[0] is source
