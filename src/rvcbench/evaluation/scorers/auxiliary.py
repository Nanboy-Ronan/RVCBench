"""Individually scheduled wrappers for the existing auxiliary metric implementations."""
from pathlib import Path
import importlib
import sys
from rvcbench.evaluation.assets import asset_dir
from rvcbench.benchmark.artifacts import file_hash


class AuxiliaryScorer:
    version = 'auxiliary_scoped_assets_v2'
    dependencies = ('torch', 'torchaudio', 'speechbrain', 'onnxruntime', 'numpy', 'librosa')

    def __init__(self, name, device, logger):
        self.name, self.device, self.logger, self.model = name, device, logger, None
        self.dependencies = {
            'speechmos': ('torch', 'torchaudio', 'numpy', 'soundfile'),
            'dnsmos': ('onnxruntime', 'numpy', 'librosa', 'soundfile', 'scipy', 'soxr', 'numba'),
        }.get(name, self.dependencies)

    def prepare(self):
        if self.name == 'dnsmos':
            from rvcbench.evaluation.fidelity import _DNSMOSPredictor
            root = asset_dir('dnsmos')
            files = [root / 'sig_bak_ovr.onnx', root / 'model_v8.onnx']
            for path in files:
                if not path.is_file():
                    raise FileNotFoundError(f'DNSMOS checkpoint missing: {path}')
            self.model = _DNSMOSPredictor(*files, personalized=False)
        elif self.name == 'speechmos':
            import torch
            hub = Path(torch.hub.get_dir())
            root = hub / 'tarepan_SpeechMOS_main'
            weights = hub / 'checkpoints/utmos22_strong_step7459_v1.pt'
            if not root.is_dir() or not weights.is_file():
                raise FileNotFoundError('SpeechMOS requires cached tarepan/SpeechMOS source and utmos22_strong_step7459_v1.pt; '
                                        'run `rvcbench setup-scorers`')
            original_path = list(sys.path)
            try:
                sys.path.insert(0, str(root))
                module = importlib.import_module('speechmos.utmos22.strong.model')
                if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
                    raise ImportError('Imported SpeechMOS source differs from its configured cache')
                self.model = module.UTMOS22Strong()
                self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
                self.model.to(self.device).eval()
            finally:
                sys.path[:] = original_path
        else:
            raise ValueError(f'Unknown metric {self.name}')
        if self.model is None:
            raise RuntimeError(f'Unable to initialize {self.name}')
        self.model_provenance = {'metric': self.name,
            'cached_files': {str(p.relative_to(root)): file_hash(p) for p in (files if self.name == 'dnsmos' else sorted(root.rglob('*')))
                if p.is_file() and p.suffix in ('.onnx', '.ckpt', '.pt', '.pth', '.yaml', '.py', '.txt')},
            'upstream_revision_pinned': False}
        if self.name == 'speechmos':
            self.model_provenance['weights'] = {weights.name: file_hash(weights)}

    def score(self, request):
        if self.name == 'dnsmos':
            values = self.model(request.generated)
            return {f'dnsmos_{k}': v for k, v in (values or {}).items()}
        import torchaudio
        import torch
        from ..audio_io import load_audio
        audio, rate = load_audio(request.generated)
        audio = torchaudio.functional.resample(audio, rate, 24000).mean(dim=0)
        with torch.inference_mode():
            score = self.model(audio.unsqueeze(0).to(self.device), 24000)
        if not isinstance(score, torch.Tensor) or score.numel() != 1:
            raise ValueError('SpeechMOS returned a malformed score')
        return {'speechmos_mos': float(score.detach().cpu().item())}

    def close(self):
        self.model = None
