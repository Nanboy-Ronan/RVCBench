"""Model-only DNS64 operations, importable without benchmark/Hydra dependencies."""
import hashlib
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_dns64(weights, device):
    import torch
    from denoiser import pretrained
    model = pretrained.dns64(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    sources = {str(p.relative_to(Path(pretrained.__file__).parent)): sha256(p)
               for p in sorted(Path(pretrained.__file__).parent.rglob('*.py'))}
    return model.eval().to(device), sources


def enhance_reference(model, source_path, destination, device, dry, dataset_rate):
    import torch
    import torchaudio
    if model.sample_rate != 16000:
        raise ValueError('DNS64 model sample rate differs from the declared architecture')
    audio, rate = torchaudio.load(str(source_path))
    if not torch.isfinite(audio).all():
        raise ValueError('Nonfinite reference input')
    if rate != dataset_rate:
        audio = torchaudio.transforms.Resample(rate, dataset_rate)(audio)
    length = audio.shape[-1]
    source = (torchaudio.transforms.Resample(dataset_rate, 16000)(audio)
              if dataset_rate != 16000 else audio).to(device)
    with torch.no_grad():
        estimate = model(source.unsqueeze(0)).squeeze(0)
    if estimate.shape != source.shape or not torch.isfinite(estimate).all():
        raise ValueError('Invalid DNS64 output shape or values')
    if dry:
        estimate = (1 - dry) * estimate + dry * source
    estimate = estimate.cpu()
    if dataset_rate != 16000:
        estimate = torchaudio.transforms.Resample(16000, dataset_rate)(estimate)
    estimate = torch.nn.functional.pad(estimate[..., :length],
        (0, max(0, length - estimate.shape[-1]))).clamp(-1, 1)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(destination), estimate, dataset_rate)
    return length
