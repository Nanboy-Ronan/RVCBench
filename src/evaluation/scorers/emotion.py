"""Pinned SpeechBrain emotion model with its native custom inference interface."""
import inspect
from pathlib import Path

from hydra.utils import to_absolute_path

from src.benchmark.artifacts import file_hash

REVISION = '117a9c3dff08be81a3628eecf6a66b547ec1659b'
BASE_REVISION = '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8'
ASSETS = {
    'custom_interface.py': 'b3b2060058f220e40c582f91da38cee416b3ce80a365f7dee46063a04ed180d7',
    'hyperparams.yaml': '4519de8fee8fef69ae5aeb8d2d307fa45a5d1795f0597b06e42aebbd96597237',
    'label_encoder.txt': '30afecda60619b67b50375bac880df4edc3862c14fb724c86701200b13e080ca',
    'wav2vec2.ckpt': 'b6db170a487b2912cfe10359fad40dbb7563120656f26f47c521b1c7e578bdd0',
    'model.ckpt': 'e351bafc8c0a2699f174e2f43f13e732836731621a8e595b53ddaf5334abcefb',
}


class EmotionScorer:
    version = 'speechbrain_native_iemocap_117a9c3_v2'
    dependencies = ('torch', 'torchaudio', 'speechbrain', 'transformers',
                    'hyperpyyaml', 'huggingface-hub', 'numpy', 'soundfile')

    def __init__(self, device, logger):
        self.device, self.logger, self.model = device, logger, None

    def prepare(self):
        root = Path(to_absolute_path('checkpoints/emotion-native'))
        for name, expected in ASSETS.items():
            if not (root / name).is_file():
                raise FileNotFoundError(f'Pinned emotion asset missing: {root / name}')
            if file_hash(root / name) != expected:
                raise ValueError(f'Pinned emotion asset hash differs: {name}')
        base = Path(to_absolute_path('wav2vec2_checkpoints')) / 'models--facebook--wav2vec2-base/snapshots' / BASE_REVISION
        for name in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin'):
            if not (base / name).is_file():
                raise FileNotFoundError(f'Emotion base initializer asset missing: {base / name}')
        from speechbrain.inference.interfaces import foreign_class
        from speechbrain.utils.fetching import FetchConfig
        self.model = foreign_class(source=str(root), pymodule_file='custom_interface.py',
            classname='CustomEncoderWav2vec2Classifier', savedir=str(root),
            overrides={'pretrained_path': str(root), 'wav2vec2_hub': str(base)},
            run_opts={'device': str(self.device)}, fetch_config=FetchConfig(allow_network=False))
        if Path(inspect.getfile(self.model.classify_batch)).resolve() != (root / 'custom_interface.py').resolve():
            self.close()
            raise ImportError('Emotion custom interface imported outside selected source directory')
        self.model.hparams.label_encoder.expect_len(4)
        self.model_provenance = {'repo_id': 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
            'revision': REVISION, 'verified_assets': dict(ASSETS),
            'base_initializer_cached_revision': BASE_REVISION,
            'base_initializer_assets': {p.name: file_hash(p) for p in sorted(base.iterdir()) if p.is_file()},
            'interface': 'CustomEncoderWav2vec2Classifier', 'sample_rate': 16000,
            'labels': ['neu', 'ang', 'hap', 'sad'],
            'note': 'Base cache content hashed; cache directory revision is not independent upstream hash verification.'}

    def _label(self, path):
        import torch
        import torchaudio
        audio, rate = torchaudio.load(str(path))
        if not audio.numel() or not torch.isfinite(audio).all():
            raise ValueError('Emotion input is empty or nonfinite')
        audio = torchaudio.functional.resample(audio, rate, 16000).mean(dim=0, keepdim=True)
        with torch.inference_mode():
            _, _, _, labels = self.model.classify_batch(audio.to(self.device))
        if len(labels) != 1 or labels[0] not in self.model_provenance['labels']:
            raise ValueError('Emotion classifier returned an unknown or malformed label')
        return labels[0]

    def score(self, request):
        ref, gen = self._label(request.reference), self._label(request.generated)
        return {'reference_emotion': ref, 'generated_emotion': gen, 'emotion_match': ref == gen}

    def close(self):
        self.model = None
