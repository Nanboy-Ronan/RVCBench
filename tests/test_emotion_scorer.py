import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from rvcbench.benchmark.artifacts import file_hash
from rvcbench.evaluation.scorers import ScoreInput, create_scorer
from rvcbench.evaluation.scorers.emotion import EmotionScorer


def test_emotion_rejects_changed_pinned_assets_before_native_loading(tmp_path, monkeypatch):
    root = tmp_path / 'emotion-native'
    root.mkdir()
    asset = root / 'custom_interface.py'
    asset.write_text('original interface')
    monkeypatch.setattr('rvcbench.evaluation.scorers.emotion.ASSETS', {asset.name: file_hash(asset)})
    monkeypatch.setenv('RVCBENCH_ASSET_DIR', str(tmp_path))
    asset.write_text('changed interface')
    with pytest.raises(ValueError, match='asset hash differs'):
        create_scorer('emotion', 'cpu', logging.getLogger()).prepare()


def test_emotion_returns_reference_and_generated_labels_and_rejects_unknown_label(tmp_path):
    pytest.importorskip('torchaudio', reason='native emotion scoring requires optional evaluation dependencies')
    wav = tmp_path / 'audio.wav'
    sf.write(wav, np.ones(1600) * .01, 16000)
    labels = iter(['neu', 'ang', 'unrecognized'])
    scorer = EmotionScorer('cpu', logging.getLogger())
    scorer.model_provenance = {'labels': ['neu', 'ang', 'hap', 'sad']}
    scorer.model = SimpleNamespace(classify_batch=lambda _: (None, None, None, [next(labels)]))
    result = scorer.score(ScoreInput(wav, wav, 'Hello.', 'EN'))
    assert result == {'reference_emotion': 'neu', 'generated_emotion': 'ang', 'emotion_match': False}
    with pytest.raises(ValueError, match='unknown or malformed'):
        scorer.score(ScoreInput(wav, wav, 'Hello.', 'EN'))
    scorer.close()
    assert scorer.model is None
