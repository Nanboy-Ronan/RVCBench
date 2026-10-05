"""Speech metrics with the definitions RVCBench uses, for your own evaluation code.

One file at a time::

    from rvcbench import metrics

    metrics.speaker_similarity("generated.wav", "reference.wav")
    metrics.word_error_rate("generated.wav", "The text it should say.", language="en")

Many files, with each metric model loaded once::

    with metrics.Evaluator(["sim", "wer", "speechmos"], device="cuda") as evaluator:
        scores = evaluator.score("generated.wav", reference="reference.wav",
                                 text="The text it should say.", language="en")
        # {'sim': 0.61, 'wer': 0.05, 'speechmos': 4.12}

The metric models are the ones ``rvcbench score`` uses; fetch them once with
``rvcbench setup-scorers``. Scores agree with ``rvcbench score`` for the same files, except
that a suite seeds Whisper per sample, which matters only when its decoding falls back to sampling.
"""
from __future__ import annotations

import logging
from pathlib import Path

#: Metric name -> (what it measures, which input it compares the generated audio with).
METRICS = {
    'sim': ('Speaker similarity: cosine similarity of ECAPA-TDNN speaker embeddings (SpeechBrain, '
            'VoxCeleb). Higher is better.', 'reference'),
    'sva': ('Speaker verification: True when the same ECAPA-TDNN model accepts both recordings as one '
            'speaker.', 'reference'),
    'wer': ('Word error rate of the Whisper medium transcript against the text, after lower-casing and '
            'removing punctuation; Chinese is segmented into words with jieba. Lower is better.', 'text'),
    'speechmos': ('Predicted naturalness (UTMOS22 strong, via SpeechMOS), 1 to 5. Higher is better.', None),
    'mcd': ('Mel-cepstral distortion (pymcd, DTW-aligned) against a recording of the same text. Lower is '
            'better.', 'reference'),
    'stoi': ('Short-time objective intelligibility against a recording of the same text. Higher is better.',
             'reference'),
    'emotion': ('True when the emotion recognized in the generated audio matches the reference '
                '(wav2vec2 on IEMOCAP, SpeechBrain).', 'reference'),
}
_SHARED_MODEL = {'sva': 'sim'}  # sim and sva come from one speaker model


def available():
    """Names of the metrics this module computes."""
    return list(METRICS)


class Evaluator:
    """Score generated speech with a fixed set of metrics, loading each metric model once.

    ``device`` is a torch device such as ``"cpu"`` or ``"cuda"``. Use it as a context manager,
    or call ``close()`` to release the models.
    """

    def __init__(self, metrics=('sim', 'wer', 'speechmos'), *, device='cpu', seed=42, logger=None):
        from rvcbench.evaluation.pipeline import ScorerPool
        if isinstance(metrics, str):
            metrics = available() if metrics == 'all' else [metrics]
        unknown = [m for m in metrics if m not in METRICS]
        if unknown or not metrics:
            raise ValueError(f'Unknown metrics {unknown}; available: {", ".join(METRICS)}')
        self.metrics = list(dict.fromkeys(metrics))
        self.seed = seed
        self._pool = ScorerPool(device, logger or logging.getLogger('rvcbench.metrics'), seed=seed)

    def score(self, generated, *, reference=None, target=None, text=None, language=None):
        """Return ``{metric: value}`` for one generated file.

        ``reference`` is a recording of the target speaker. ``target`` is a recording of the same
        text for MCD/STOI (defaults to ``reference`` for compatibility). ``text`` is what the generated
        audio should say; ``language`` is a code such as
        ``"en"`` or ``"zh"`` that Whisper uses as a hint.
        """
        from rvcbench.benchmark.artifacts import METRIC_COLUMNS, metric_value_valid
        from rvcbench.evaluation.scorers import ScoreInput
        from rvcbench.utils.seeding import isolated_seed
        needs = {('target' if m in ('mcd', 'stoi') and target is not None else METRICS[m][1]) for m in self.metrics}
        missing = sorted(needs - {None} - ({'reference'} if reference is not None else set())
                         - ({'target'} if target is not None else set()) - ({'text'} if text else set()))
        if missing:
            raise ValueError(f'{", ".join(missing)} is required for {", ".join(self.metrics)}')
        generated = Path(generated)
        if not generated.is_file():
            raise FileNotFoundError(generated)
        scores = {}
        for group in dict.fromkeys(_SHARED_MODEL.get(m, m) for m in self.metrics):
            compared = target if group in ('mcd', 'stoi') and target is not None else reference
            if compared is not None and not Path(compared).is_file():
                raise FileNotFoundError(compared)
            request = ScoreInput(Path(compared) if compared is not None else None, generated, text or '', language)
            with isolated_seed(self.seed, device=self._pool.device):
                values = self._pool.get(group).score(request)
            for metric in self.metrics:
                if _SHARED_MODEL.get(metric, metric) == group:
                    value = values[METRIC_COLUMNS[metric]]
                    if not metric_value_valid(metric, value):
                        raise ValueError(f'{metric} returned an invalid score: {value!r}')
                    scores[metric] = value if isinstance(value, bool) else float(value)
        return scores

    def close(self):
        self._pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _one(metric, generated, *, device='cpu', **inputs):
    with Evaluator([metric], device=device) as evaluator:
        return evaluator.score(generated, **inputs)[metric]


def speaker_similarity(generated, reference, *, device='cpu'):
    """ECAPA-TDNN cosine similarity between a generated file and a recording of the target speaker."""
    return _one('sim', generated, reference=reference, device=device)


def word_error_rate(generated, text, *, language=None, device='cpu'):
    """Word error rate of the Whisper medium transcript of ``generated`` against ``text``."""
    return _one('wer', generated, text=text, language=language, device=device)


def mos(generated, *, device='cpu'):
    """Predicted naturalness (UTMOS22 strong via SpeechMOS) of a generated file."""
    return _one('speechmos', generated, device=device)


def mel_cepstral_distortion(generated, reference):
    """Mel-cepstral distortion between a generated file and a recording of the same text."""
    return _one('mcd', generated, reference=reference)


def stoi(generated, reference):
    """STOI between a generated file and a recording of the same text."""
    return _one('stoi', generated, reference=reference)
