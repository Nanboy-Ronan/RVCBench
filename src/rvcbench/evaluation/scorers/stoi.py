class STOIScorer:
    """Short-time objective intelligibility of the generated audio against its reference recording."""
    version = 'torch_stoi_16k_trim_to_shorter_v1'
    dependencies = ('torch-stoi', 'torch', 'torchaudio', 'soundfile', 'numpy')
    sample_rate = 16000

    def __init__(self, device, logger):
        self.model = None

    def prepare(self):
        from torch_stoi import NegSTOILoss
        self.model = NegSTOILoss(sample_rate=self.sample_rate)
        self.model_provenance = {'implementation': 'torch_stoi.NegSTOILoss', 'sample_rate': self.sample_rate,
                                 'alignment': 'mono, resampled, trimmed to the shorter signal, padded to one second'}

    def _load(self, path):
        import soundfile as sf
        import torch
        import torchaudio.functional as F
        audio, rate = sf.read(str(path), dtype='float32', always_2d=True)
        waveform = torch.from_numpy(audio.mean(axis=1))
        return F.resample(waveform, rate, self.sample_rate) if rate != self.sample_rate else waveform

    def score(self, request):
        import torch
        import torch.nn.functional as F
        reference, generated = self._load(request.reference), self._load(request.generated)
        length = min(reference.numel(), generated.numel())
        reference, generated = reference[:length].unsqueeze(0), generated[:length].unsqueeze(0)
        if length < self.sample_rate:
            # Same short-clip handling as the protection fidelity metrics.
            reference = F.pad(reference, (0, self.sample_rate - length))
            generated = F.pad(generated, (0, self.sample_rate - length))
        with torch.no_grad():
            return {'stoi': float(-self.model(generated, reference).item())}

    def close(self):
        self.model = None
