import torch
from torch import nn

class WienerFilter(nn.Module):
    def __init__(self, noise_est_frames: int = 10, eps: float = 1e-8):
        super(WienerFilter, self).__init__()
        self.noise_est_frames = noise_est_frames
        self.eps = eps

    def forward(self, noisy_stft: torch.Tensor) -> torch.Tensor:
        """
        Apply Wiener filter to the noisy STFT
        """
        mag = torch.abs(noisy_stft)
        phase = torch.angle(noisy_stft)

        noise_mag_est = mag[:, :, :self.noise_est_frames].mean(dim=2, keepdim=True)
        noise_mag_est = noise_mag_est.expand_as(mag)

        gain = mag ** 2 / (mag ** 2 + noise_mag_est ** 2 + self.eps)

        enhanced_mag = gain * mag

        real = enhanced_mag * torch.cos(phase)
        imag = enhanced_mag * torch.sin(phase)

        return torch.complex(real, imag)