"""Audio file input and output for evaluation, through soundfile.

torchaudio 2.9 moved ``load``, ``save`` and ``info`` to the optional ``torchcodec`` package.
soundfile reads WAV, FLAC and OGG with every supported torch version and returns the same
float32 samples as ``torchaudio.load`` (integer PCM scaled to [-1, 1]).
"""
from pathlib import Path

import soundfile as sf


def load_audio(path, *, frame_offset=0, num_frames=-1):
    """Return ``(waveform, sample_rate)`` with a float32 tensor shaped (channels, frames), like ``torchaudio.load``."""
    import torch
    data, rate = sf.read(str(Path(path)), start=frame_offset, frames=num_frames, dtype='float32', always_2d=True)
    return torch.from_numpy(data.T.copy()), rate


def save_audio(path, waveform, sample_rate):
    """Write a (channels, frames) tensor as 32-bit float, as ``torchaudio.save`` does for float tensors."""
    sf.write(str(Path(path)), waveform.detach().cpu().numpy().T, int(sample_rate), subtype='FLOAT')


def audio_info(path):
    """Return ``(sample_rate, num_frames)`` of an audio file."""
    info = sf.info(str(Path(path)))
    return info.samplerate, info.frames
