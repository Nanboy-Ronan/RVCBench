from rvcbench.evaluation.assets import asset_dir


class SpeakerScorer:
    version = 'ecapa_verify_files_v1'
    dependencies = ('speechbrain', 'torch', 'torchaudio', 'numpy', 'huggingface-hub')

    def __init__(self, device, logger):
        self.device, self.logger, self.model = device, logger, None

    def prepare(self):
        from speechbrain.inference.speaker import SpeakerRecognition
        from ..setup import setup_speaker
        from ..locked_assets import scorer_lock, verify_files
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy
        root = asset_dir('spkrec-ecapa-voxceleb')
        setup_speaker(check_only=True)
        spec = scorer_lock('sim')
        if (root / 'label_encoder.ckpt').exists():
            verify_files(root, {'label_encoder.ckpt': spec['files']['label_encoder.txt']})
        self.model = SpeakerRecognition.from_hparams(source=str(root), savedir=str(root),
            overrides={'pretrained_path': str(root)}, run_opts={'device': str(self.device)},
            local_strategy=LocalStrategy.COPY, fetch_config=FetchConfig(allow_network=False))
        self.model_provenance = {'model': 'speechbrain/spkrec-ecapa-voxceleb',
            'revision': spec['revision'], 'files': dict(spec['files'])}

    def score(self, request):
        score, decision = self.model.verify_files(str(request.reference), str(request.generated))
        return {'sim': float(score), 'sva': bool(decision)}

    def close(self):
        self.model = None
