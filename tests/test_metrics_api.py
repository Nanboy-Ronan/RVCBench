import numpy as np
import pytest
import soundfile as sf

from rvcbench import metrics
from rvcbench.evaluation import pipeline


class FakeScorer:
    created = []
    closed = 0
    VALUES = {'sim': {'sim': 0.61, 'sva': True}, 'wer': {'wer': 0.05, 'predicted_text': 'hello'},
              'speechmos': {'speechmos_mos': 4.1}, 'mcd': {'mcd': 6.2}, 'stoi': {'stoi': 0.93},
              'emotion': {'emotion_match': False, 'reference_emotion': 'neu', 'generated_emotion': 'hap'}}

    def __init__(self, name, device, logger):
        self.name = name
        FakeScorer.created.append(name)
        self.requests = []

    def prepare(self):
        pass

    def score(self, request):
        self.requests.append(request)
        return dict(self.VALUES[self.name])

    def close(self):
        FakeScorer.closed += 1


@pytest.fixture
def fake(monkeypatch, tmp_path):
    FakeScorer.created, FakeScorer.closed = [], 0
    monkeypatch.setattr(pipeline, 'create_scorer', FakeScorer)
    for name in ('gen.wav', 'ref.wav'):
        sf.write(tmp_path / name, np.zeros(1600), 16000)
    return tmp_path


def test_evaluator_loads_each_model_once_and_returns_metric_names(fake):
    with metrics.Evaluator(['sim', 'sva', 'wer', 'speechmos', 'emotion'], device='cpu') as evaluator:
        for _ in range(3):
            scores = evaluator.score(fake / 'gen.wav', reference=fake / 'ref.wav', text='hello', language='en')
    assert scores == {'sim': 0.61, 'sva': True, 'wer': 0.05, 'speechmos': 4.1, 'emotion': False}
    assert sorted(FakeScorer.created) == ['emotion', 'sim', 'speechmos', 'wer']  # sim and sva share one model
    assert FakeScorer.closed == 4


def test_missing_inputs_and_unknown_metrics_are_explained(fake):
    with pytest.raises(ValueError, match='Unknown metrics'):
        metrics.Evaluator(['sim', 'loudness'])
    with metrics.Evaluator(['sim', 'wer']) as evaluator:
        with pytest.raises(ValueError, match='reference, text is required'):
            evaluator.score(fake / 'gen.wav')
        with pytest.raises(FileNotFoundError):
            evaluator.score(fake / 'missing.wav', reference=fake / 'ref.wav', text='hello')
    with metrics.Evaluator(['speechmos']) as evaluator:
        assert evaluator.score(fake / 'gen.wav') == {'speechmos': 4.1}  # needs neither reference nor text


def test_one_line_functions(fake):
    assert metrics.speaker_similarity(fake / 'gen.wav', fake / 'ref.wav') == 0.61
    assert metrics.word_error_rate(fake / 'gen.wav', 'hello', language='en') == 0.05
    assert metrics.mos(fake / 'gen.wav') == 4.1
    assert metrics.mel_cepstral_distortion(fake / 'gen.wav', fake / 'ref.wav') == 6.2
    assert metrics.stoi(fake / 'gen.wav', fake / 'ref.wav') == 0.93
    assert FakeScorer.closed == 5


def test_every_metric_has_a_description_and_a_scored_column():
    from rvcbench.benchmark.artifacts import METRIC_COLUMNS
    assert metrics.available() == list(metrics.METRICS)
    for name, (description, needs) in metrics.METRICS.items():
        assert name in METRIC_COLUMNS and description.endswith('.') and needs in (None, 'reference', 'text')


@pytest.mark.parametrize('failure', [False, True])
def test_scoring_preserves_callers_rng_and_backend_flags(fake, monkeypatch, failure):
    import os
    import random
    import torch
    monkeypatch.setenv('PYTHONHASHSEED', '913')
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', False)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', True)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    random.seed(123)
    np.random.seed(321)
    torch.manual_seed(456)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    if failure:
        monkeypatch.setattr(FakeScorer, 'score', lambda *args: (_ for _ in ()).throw(RuntimeError('scorer failed')))
    with metrics.Evaluator('sim') as evaluator:
        if failure:
            with pytest.raises(RuntimeError, match='scorer failed'):
                evaluator.score(fake / 'gen.wav', reference=fake / 'ref.wav')
        else:
            evaluator.score(fake / 'gen.wav', reference=fake / 'ref.wav')
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and np.array_equal(after[1], before[1][1]) and after[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert os.environ['PYTHONHASHSEED'] == '913'
    assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32


def test_cpu_evaluation_does_not_initialize_or_reseed_unrelated_gpus(fake, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda *a, **k: pytest.fail('CPU scoring touched CUDA RNG'))
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a, **k: pytest.fail('CPU scoring seeded all GPUs'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: pytest.fail('CPU scoring cleared CUDA cache'))
    with metrics.Evaluator('sim', device='cpu') as evaluator:
        evaluator.score(fake / 'gen.wav', reference=fake / 'ref.wav')


def test_same_text_target_and_voice_reference_are_separate(fake):
    sf.write(fake / 'target.wav', np.ones(1600) * .01, 16000)
    with metrics.Evaluator('all') as evaluator:
        scores = evaluator.score(fake / 'gen.wav', reference=fake / 'ref.wav', target=fake / 'target.wav', text='hello')
        assert set(scores) == set(metrics.available())
        assert evaluator._pool.get('mcd').requests[0].reference == fake / 'target.wav'
        assert evaluator._pool.get('sim').requests[0].reference == fake / 'ref.wav'


def test_nonfinite_metric_is_rejected(fake, monkeypatch):
    monkeypatch.setattr(FakeScorer, 'score', lambda *args: {'sim': float('nan')})
    with metrics.Evaluator('sim') as evaluator, pytest.raises(ValueError, match='invalid score'):
        evaluator.score(fake / 'gen.wav', reference=fake / 'ref.wav')
