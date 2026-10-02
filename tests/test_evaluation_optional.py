"""Evaluator aggregation checks; run with .[eval,test], no checkpoint downloads."""
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

generation = pytest.importorskip('rvcbench.evaluation.generation', reason='optional evaluation dependencies unavailable')


def test_nonfinite_metric_is_missing_and_csv_keeps_identity(tmp_path, monkeypatch):
    audio = tmp_path / 'audio'
    audio.mkdir()
    wav = audio / 'sample.wav'
    sf.write(wav, np.zeros(1600), 16000)
    models = (
        SimpleNamespace(transcribe=lambda *a, **k: {'text': 'hello', 'language': 'en'}),
        SimpleNamespace(verify_files=lambda *a: (float('nan'), True)),
        SimpleNamespace(calculate_mcd=lambda *a: .25), None, None, None)
    monkeypatch.setattr(generation, '_load_evaluation_components', lambda *a: models)
    result = generation.evaluate_pairs([(wav, wav, {'sample_id': 'pair-1', 'text': 'hello', 'speaker_id': 's'})],
                                       audio, 'cpu', logging.getLogger('test-eval'))
    assert result['sim_pairs'] == 0 and result['avg_sim'] is None
    assert result['mcd_pairs'] == 1 and result['avg_mcd'] == .25
    assert result['wer_pairs'] == 1 and result['avg_wer'] == 0
    import csv
    with open(result['sample_metrics_csv']) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['sample_id'] == 'pair-1' and rows[0]['sim'] == ''
