from pathlib import Path
from hydra.utils import to_absolute_path
from rvcbench.benchmark.artifacts import file_hash


class SpeakerScorer:
    version = 'ecapa_verify_files_v1'
    dependencies = ('speechbrain', 'torch', 'torchaudio', 'numpy', 'huggingface-hub')

    def __init__(self, device, logger):
        self.device, self.logger, self.model = device, logger, None

    def prepare(self):
        from speechbrain.inference.speaker import SpeakerRecognition
        root = Path(to_absolute_path('checkpoints/spkrec-ecapa-voxceleb'))
        root.mkdir(parents=True, exist_ok=True)
        # Reuse complete released assets locally; otherwise let SpeechBrain fetch them.
        source = str(root) if all((root / f).is_file() for f in
            ('hyperparams.yaml', 'embedding_model.ckpt', 'mean_var_norm_emb.ckpt', 'classifier.ckpt')) else 'speechbrain/spkrec-ecapa-voxceleb'
        self.model = SpeakerRecognition.from_hparams(source=source, savedir=str(root),
                                                     run_opts={'device': str(self.device)})
        self.model_provenance = {'model': 'speechbrain/spkrec-ecapa-voxceleb',
            'files': {p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()}}

    def score(self, request):
        score, decision = self.model.verify_files(str(request.reference), str(request.generated))
        return {'sim': float(score), 'sva': bool(decision)}

    def close(self):
        self.model = None
