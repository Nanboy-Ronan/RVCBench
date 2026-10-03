from rvcbench.benchmark.artifacts import file_hash
from rvcbench.evaluation.assets import asset_dir


class SpeakerScorer:
    version = 'ecapa_verify_files_v1'
    dependencies = ('speechbrain', 'torch', 'torchaudio', 'numpy', 'huggingface-hub')

    def __init__(self, device, logger):
        self.device, self.logger, self.model = device, logger, None

    def prepare(self):
        from speechbrain.inference.speaker import SpeakerRecognition
        root = asset_dir('spkrec-ecapa-voxceleb')
        root.mkdir(parents=True, exist_ok=True)
        options = {}
        if all((root / f).is_file() for f in
               ('hyperparams.yaml', 'embedding_model.ckpt', 'mean_var_norm_emb.ckpt', 'classifier.ckpt')):
            source = str(root)  # complete released assets
        else:
            from speechbrain.utils.fetching import LocalStrategy
            source = 'speechbrain/spkrec-ecapa-voxceleb'
            # Copy instead of linking into the Hugging Face cache, which may move or be cleared.
            options['local_strategy'] = LocalStrategy.COPY
        self.model = SpeakerRecognition.from_hparams(source=source, savedir=str(root),
                                                     run_opts={'device': str(self.device)}, **options)
        self.model_provenance = {'model': 'speechbrain/spkrec-ecapa-voxceleb',
            'files': {p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()}}

    def score(self, request):
        score, decision = self.model.verify_files(str(request.reference), str(request.generated))
        return {'sim': float(score), 'sva': bool(decision)}

    def close(self):
        self.model = None
