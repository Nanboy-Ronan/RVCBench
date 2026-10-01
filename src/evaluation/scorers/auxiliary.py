"""Individually scheduled wrappers for the existing auxiliary metric implementations."""
from pathlib import Path
from src.benchmark.artifacts import file_hash


class AuxiliaryScorer:
    version = 'historical_auxiliary_v1'
    dependencies = ('torch', 'torchaudio', 'speechbrain', 'onnxruntime', 'numpy', 'librosa')

    def __init__(self, name, device, logger):
        self.name, self.device, self.logger, self.model = name, device, logger, None

    def prepare(self):
        if self.name == 'emotion':
            from src.evaluation.generation import _load_emotion_recognizer
            self.model = _load_emotion_recognizer(self.device, self.logger)
            root = Path('checkpoints/emotion-recognition-wav2vec2-IEMOCAP')
        elif self.name == 'dnsmos':
            from src.evaluation.fidelity import _load_dnsmos_predictor
            self.model = _load_dnsmos_predictor(self.logger)
            root = Path('checkpoints/dnsmos')
        elif self.name == 'speechmos':
            import torch
            from src.evaluation.fidelity import _load_speechmos_predictor
            self.model = _load_speechmos_predictor(self.logger, torch.device(self.device))
            root = Path(torch.hub.get_dir())
        else:
            raise ValueError(f'Unknown metric {self.name}')
        if self.model is None:
            raise RuntimeError(f'Unable to initialize {self.name}')
        self.model_provenance = {'metric': self.name,
            'cached_files': {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*'))
                if p.is_file() and p.suffix in ('.onnx', '.ckpt', '.pt', '.pth', '.yaml', '.py', '.txt')},
            'upstream_revision_pinned': False}

    def score(self, request):
        if self.name == 'emotion':
            from src.evaluation.generation import _predict_emotion_label
            ref = _predict_emotion_label(self.model, request.reference, self.logger)
            gen = _predict_emotion_label(self.model, request.generated, self.logger)
            return {'reference_emotion': ref, 'generated_emotion': gen,
                    'emotion_match': ref == gen if ref is not None and gen is not None else None}
        if self.name == 'dnsmos':
            from src.evaluation.fidelity import _predict_dnsmos_scores
            values = _predict_dnsmos_scores(self.model, request.generated, self.logger)
            return {f'dnsmos_{k}': v for k, v in (values or {}).items()}
        import torchaudio
        from src.evaluation.fidelity import _predict_speechmos_mos
        audio, rate = torchaudio.load(str(request.generated))
        audio = torchaudio.functional.resample(audio, rate, 24000).mean(dim=0)
        return {'speechmos_mos': _predict_speechmos_mos(self.model, request.generated, audio, 24000, self.logger)}

    def close(self):
        self.model = None
