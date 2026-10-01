import copy
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from test_benchmark import setup_run
from src.evaluation.pipeline import evaluate_run
from src.benchmark.artifacts import input_fingerprint, input_records, sample_id
from src.datasets.manifest_utils import select_manifest_variant


def test_mcd_cache_fingerprint_tracks_native_feature_and_alignment_libraries():
    from src.evaluation.pipeline import _scorer_provenance
    from src.evaluation.scorers.mcd import MCDScorer
    from src.benchmark.artifacts import digest
    scorer = MCDScorer('cpu', logging.getLogger())
    scorer.model_provenance = {'implementation': 'pymcd', 'MCD_mode': 'dtw'}
    versions = {name: 'fixture-1' for name in scorer.dependencies}
    with patch('importlib.metadata.version', side_effect=versions.__getitem__):
        baseline = digest(_scorer_provenance(scorer, 42, None))
        for name in ('pyworld', 'pysptk', 'fastdtw', 'soundfile', 'soxr'):
            versions[name] = 'fixture-2'
            assert digest(_scorer_provenance(scorer, 42, None)) != baseline
            versions[name] = 'fixture-1'


class FakeScorer:
    version = 'fixture-v1'
    dependencies = ()

    def __init__(self, name, calls, failing=None):
        self.name, self.calls, self.failing = name, calls, failing
        self.model_provenance = {'fixture': name}

    def prepare(self):
        self.calls.append(('prepare', self.name))

    def score(self, request):
        self.calls.append(('score', self.name, request.generated.name))
        if self.failing:
            raise RuntimeError('metric-specific failure')
        return {self.name: .5}

    def close(self):
        self.calls.append(('close', self.name))


def test_metric_resume_after_interruption_and_independent_failures(setup_run, tmp_path):
    _, _, run, _ = setup_run
    _, manifest, _ = run('source')
    rows = manifest['samples']
    calls = []
    def interrupt(row):
        raise KeyboardInterrupt()
    output = tmp_path / 'scores'
    with patch('src.evaluation.pipeline.create_scorer', side_effect=lambda n, *_: FakeScorer(n, calls)):
        with pytest.raises(KeyboardInterrupt):
            evaluate_run(copy.deepcopy(rows), ['sim', 'wer'], output, 'cpu', logging.getLogger(), on_sample=interrupt)
        assert calls[-1] == ('close', 'sim')
        assert len(list((output / 'metric_cache').glob('*.json'))) == 1
        evaluate_run(copy.deepcopy(rows), ['sim', 'wer'], output, 'cpu', logging.getLogger())
    assert sum(c[:2] == ('score', 'sim') for c in calls) == 2  # first successful score was reused
    assert calls.index(('close', 'sim')) < calls.index(('prepare', 'wer'))
    calls.clear()
    with patch('src.evaluation.pipeline.create_scorer', side_effect=lambda n, *_: FakeScorer(n, calls, n == 'wer')):
        fresh = copy.deepcopy(rows)
        result = evaluate_run(fresh, ['sim', 'wer'], tmp_path / 'failed-wer', 'cpu', logging.getLogger())
    assert result['sim_pairs'] == 2 and result['wer_pairs'] == 0
    assert all(r['status'] == 'metric_failed' and 'wer' in r['metric_errors'] for r in fresh)


def test_metric_cache_rejects_modified_source(setup_run, tmp_path):
    _, _, run, _ = setup_run
    _, manifest, _ = run('source')
    rows = manifest['samples']
    Path(rows[0]['generated_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed before evaluation'):
        evaluate_run(rows, ['sim'], tmp_path / 'scores', 'cpu', logging.getLogger())


def test_variants_preserve_both_transcripts_and_disambiguate_identity(setup_run):
    import pandas as pd
    from dataclasses import replace
    _, dataset, _, _ = setup_run
    sample = dataset.get_zero_shot_samples()[0]
    a = dict(sample.extra, source_manifest='one.json')
    b = dict(sample.extra, source_manifest='one_text.json', prompt_text='--hello')
    frame = pd.DataFrame([a, b]).drop(columns=['manifest_variant'])
    both = select_manifest_variant(frame)
    assert len(both) == 2 and frame.prompt_text.tolist() == ['hello', '--hello']
    assert len(select_manifest_variant(frame, 'speaker')) == 1
    assert select_manifest_variant(frame, 'speaker_text').iloc[0].prompt_text == '--hello'
    variants = [replace(sample, extra=r) for r in both.to_dict(orient='records')]
    assert sample_id(variants[0]) != sample_id(variants[1])
    with pytest.raises(ValueError, match='unavailable'):
        select_manifest_variant(frame, 'unknown')


def test_frozen_subset_roundtrip_keeps_identity_and_input_fingerprint(setup_run, tmp_path):
    from src.benchmark.subsets import freeze
    from src.datasets.zero_shot import ZeroShotDataset
    conf, dataset, _, _ = setup_run
    out = freeze(dataset, tmp_path / 'subset', speakers=1, pairs_per_speaker=2)
    conf.dataset.manifest_filename = str(out / 'metadata.json')
    reloaded = ZeroShotDataset(conf, conf.dataset, logging.getLogger())
    assert input_fingerprint(input_records(dataset.get_zero_shot_samples())) == input_fingerprint(input_records(reloaded.get_zero_shot_samples()))
