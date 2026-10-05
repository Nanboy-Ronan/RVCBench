"""Exercise durable suite scoring with the real cache and a small deterministic scorer."""
import json
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from rvcbench.benchmark import submission
from test_submission import suite, export, echo  # noqa: F401


class Scorer:
    version = 'resume-fixture-v1'
    dependencies = ()
    model_provenance = {'weights': 'fixed'}
    calls = []
    interrupt = False

    def prepare(self):
        pass

    def score(self, request):
        if self.interrupt and len(self.calls) == 1:
            raise KeyboardInterrupt('interrupted')
        self.calls.append(request.generated.name)
        return {'sim': .5}

    def close(self):
        pass


@pytest.fixture
def scoring(suite, monkeypatch):
    entries, prompts = export(suite)
    generated = suite[2] / 'generated'
    echo(entries, prompts, generated)
    Scorer.calls, Scorer.interrupt = [], False
    monkeypatch.setattr('rvcbench.evaluation.pipeline.create_scorer', lambda *a: Scorer())
    def run(**kwargs):
        return submission.score_submission(suite[0], generated, suite[2] / 'scored', model='test',
                                           data_root=suite[1], **kwargs)
    return run, entries, generated


def test_interrupted_scoring_resumes_only_missing_scores(suite, scoring):
    run, entries, generated = scoring
    Scorer.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        run()
    record = json.loads((suite[2] / 'scored/submission.json').read_text())
    assert record['status'] == 'failed' and len(Scorer.calls) == 1
    Scorer.interrupt = False
    result = run(resume=True)
    assert result['status'] == 'complete' and len(Scorer.calls) == len(entries)
    original = result['tasks']['toy']['means']
    result = run(resume=True)
    assert len(Scorer.calls) == len(entries) and result['tasks']['toy']['means'] == original
    # Changed generated content must invalidate only that sample's cached score.
    sf.write(generated / entries[0]['output_file'], np.ones(1600) * .0123, 16000)
    run(resume=True)
    assert len(Scorer.calls) == len(entries) + 1


def test_resume_retries_missing_audio_and_rejects_changed_suite(suite, scoring):
    run, entries, generated = scoring
    path = generated / entries[0]['output_file']
    original = path.read_bytes()
    path.unlink()
    assert run()['status'] == 'partial'
    path.write_bytes(original)
    assert run(resume=True)['status'] == 'complete'
    assert len(Scorer.calls) == len(entries)
    suite[0].write_text(suite[0].read_text() + '\n')
    before = (suite[2] / 'scored/submission.json').read_bytes()
    with pytest.raises(ValueError, match='cannot resume'):
        run(resume=True)
    assert (suite[2] / 'scored/submission.json').read_bytes() == before


def test_resume_refuses_another_writer(suite, scoring):
    import fcntl
    run, _, _ = scoring
    run()
    with (suite[2] / 'scored/.score.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='another scoring process'):
            run(resume=True)


def test_multimodel_resume_finishes_remaining_models(suite, scoring):
    _, entries, generated = scoring
    out = suite[2] / 'batch'
    Scorer.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        submission.score_submissions(suite[0], [generated, generated], out, models=['a', 'b'], data_root=suite[1])
    Scorer.interrupt = False
    result = submission.score_submissions(suite[0], [generated, generated], out, models=['a', 'b'],
                                          data_root=suite[1], resume=True)
    assert all(m['status'] == 'complete' for m in result['models'])
    assert len(Scorer.calls) == 2 * len(entries)
    assert result['protocol_compatible']
