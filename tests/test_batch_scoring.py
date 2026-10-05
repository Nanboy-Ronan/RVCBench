import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from rvcbench.benchmark import comparison, submission
from rvcbench.benchmark.artifacts import file_hash
from rvcbench.evaluation import pipeline
from test_submission import PAIRS, export, fake_evaluate, suite  # noqa: F401  (fixture)


def flat_echo(entries, prompts, directory):
    """What a batch script writes: every output as <id>.wav in one directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        shutil.copyfile(prompts / entry['reference_audio'], directory / f"{entry['id']}.wav")


def read_list(path, separator):
    return [line.split(separator) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('name, separator', [('prompts.tsv', '\t'), ('prompts.lst', '|')])
def test_batch_lists_name_each_output_and_use_absolute_reference_paths(suite, name, separator):
    entries, prompts = export(suite)
    lines = read_list(prompts / name, separator)
    assert len(lines) == len(entries) == len(PAIRS)
    for (utt, reference_text, reference, text), entry in zip(lines, entries):
        assert utt == entry['id'] == f"toy__{entry['pair_id']}"
        assert (reference_text, text) == (entry['reference_text'], entry['text'])
        assert Path(reference).is_absolute() and file_hash(reference) == entry['reference_sha256']
    assert name in (prompts / 'README.md').read_text()


def test_flat_outputs_are_scored_like_nested_ones(suite):
    entries, prompts = export(suite)
    flat_echo(entries, prompts, suite[2] / 'flat')
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=fake_evaluate):
        result = submission.score_submission(suite[0], suite[2] / 'flat', suite[2] / 'scored', model='echo',
                                             data_root=suite[1])
    task = result['tasks']['toy']
    assert result['status'] == 'complete' and task['coverage']['generated'] == len(PAIRS)
    stored = sorted(p.name for p in (suite[2] / 'scored' / 'toy' / 'generated_audio').iterdir())
    assert stored == sorted(f"{entry['pair_id']}.wav" for entry in entries)


def test_a_directory_without_any_expected_file_fails_before_writing_results(suite):
    export(suite)
    (suite[2] / 'wrong').mkdir()
    sf.write(suite[2] / 'wrong' / 'utt1.wav', np.zeros(1600), 16000)
    with pytest.raises(ValueError, match='No audio for suite toy-v1.*toy__Toy-s1-000000.wav'):
        submission.score_submission(suite[0], suite[2] / 'wrong', suite[2] / 'scored', model='echo', data_root=suite[1])
    assert not (suite[2] / 'scored').exists()


def test_a_text_with_a_list_separator_is_refused(tmp_path):
    entry = {'id': 'toy__a', 'reference_text': 'one|two', 'reference_audio': 'references/toy/a.wav', 'text': 'x'}
    with pytest.raises(ValueError, match="toy__a.*prompts.lst"):
        submission.write_batch_lists([entry], tmp_path)


def test_line_breaks_in_texts_become_spaces_in_the_lists(tmp_path):
    entry = {'id': 'toy__a', 'reference_text': 'ref', 'reference_audio': 'references/toy/a.wav',
             'text': 'Elements:\n        *   one\r\ntwo'}
    submission.write_batch_lists([entry], tmp_path)
    for name, separator in submission.BATCH_LISTS.items():
        lines = (tmp_path / name).read_text().splitlines()
        assert len(lines) == 1 and lines[0].split(separator)[3] == 'Elements: *   one two'


def test_packaged_suites_can_be_written_as_batch_lists():
    for name in submission.available_suites():
        spec = submission.load_suite(name)
        for task in spec['tasks']:
            if task.get('derived_from'):
                continue
            for row in submission.read_json(spec['_directory'] / task['manifest']):
                for text in (row['prompt_text'], row['target_text']):
                    assert not any(c in text for c in '\t|'), (name, task['task'], row['pair_id'])


def two_models(suite):
    entries, prompts = export(suite)
    flat_echo(entries, prompts, suite[2] / 'outputs' / 'model-a')
    flat_echo(entries[1:], prompts, suite[2] / 'outputs' / 'model-b')
    return [suite[2] / 'outputs' / 'model-a', suite[2] / 'outputs' / 'model-b']


def test_several_models_are_scored_into_one_comparison(suite):
    directories = two_models(suite)
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=fake_evaluate):
        result = submission.score_submissions(suite[0], directories, suite[2] / 'results', data_root=suite[1])
    assert [(m['model'], m['status']) for m in result['models']] == [('model-a', 'complete'), ('model-b', 'partial')]
    results = suite[2] / 'results'
    assert json.loads((results / 'model-a' / 'submission.json').read_text())['model'] == 'model-a'
    assert {p.name for p in results.glob('comparison.*')} == {'comparison.md', 'comparison.csv', 'comparison.json'}
    table = (results / 'comparison.md').read_text()
    assert '| Task | Metric | model-a | model-b |' in table
    assert '| toy | SIM ↑ | 0.500 | — |' in table  # the incomplete task has no mean, so nothing is bold
    with pytest.raises(ValueError, match='already holds results'):
        submission.score_submissions(suite[0], directories, results, data_root=suite[1])


def test_model_names_must_match_the_directories(suite):
    directories = two_models(suite)
    with pytest.raises(ValueError, match='2 output directories but 1 model names'):
        submission.score_submissions(suite[0], directories, suite[2] / 'r', models=['only-one'], data_root=suite[1])
    with pytest.raises(ValueError, match='unique'):
        submission.score_submissions(suite[0], directories, suite[2] / 'r', models=['same', 'same'], data_root=suite[1])


def fake_submission(path, model, means, *, suite_sha='abc', changes=None):
    tasks = {'clean': {'evaluation': 'Clean', 'required_metrics': ['sim', 'wer'], 'status': 'complete',
                       'means': means['clean'], 'evaluation_fingerprint': 'fixed-test-scorer'},
             'noisy': {'evaluation': 'Noisy', 'anchor': 'clean', 'required_metrics': ['sim', 'wer'],
                       'status': 'complete', 'means': means['noisy'], 'evaluation_fingerprint': 'fixed-test-scorer',
                       'relative_change_percent': changes or {}}}
    path.mkdir(parents=True)
    (path / 'submission.json').write_text(json.dumps({
        'schema_version': 1, 'protocol': 'rvcbench-submission-v1', 'suite': 'toy-v1', 'version': 1,
        'suite_sha256': suite_sha, 'leaderboard': False, 'label': 'Toy suite.', 'model': model,
        'status': 'complete', 'tasks': tasks}))
    return path


def test_comparison_marks_the_best_value_by_metric_direction(tmp_path):
    a = fake_submission(tmp_path / 'a', 'a', {'clean': {'sim': 0.6, 'wer': 0.10}, 'noisy': {'sim': 0.3, 'wer': 0.2}},
                        changes={'sim': -50.0, 'wer': 100.0})
    b = fake_submission(tmp_path / 'b', 'b', {'clean': {'sim': 0.5, 'wer': 0.05}, 'noisy': {'sim': 0.4, 'wer': 0.3}})
    table = comparison.compare_submissions([a, b / 'submission.json'], tmp_path / 'out')['markdown']
    assert '| clean | SIM ↑ | **0.600** | 0.500 |' in table
    assert '| clean | WER ↓ | 0.100 | **0.050** |' in table
    assert '| noisy | SIM ↑ | 0.300 (-50.0%) | **0.400** |' in table
    rows = (tmp_path / 'out' / 'comparison.csv').read_text().splitlines()
    assert rows[0] == ','.join(comparison.CSV_FIELDS) and len(rows) == 1 + 2 * 2 * 2


def test_submissions_of_different_suite_versions_are_not_compared(tmp_path):
    means = {'clean': {'sim': 0.6, 'wer': 0.1}, 'noisy': {'sim': 0.3, 'wer': 0.2}}
    a = fake_submission(tmp_path / 'a', 'a', means)
    b = fake_submission(tmp_path / 'b', 'b', means, suite_sha='other')
    with pytest.raises(ValueError, match='only submissions of the same suite'):
        comparison.compare_submissions([a, b])


def test_compare_command_prints_the_table(tmp_path, monkeypatch, capsys):
    from rvcbench.benchmark import cli
    means = {'clean': {'sim': 0.6, 'wer': 0.1}, 'noisy': {'sim': 0.3, 'wer': 0.2}}
    paths = [fake_submission(tmp_path / name, name, means) for name in ('a', 'b')]
    monkeypatch.setattr(sys, 'argv', ['rvcbench', 'compare', *map(str, paths), '--output', str(tmp_path / 'out')])
    cli.main()
    assert '| Task | Metric | a | b |' in capsys.readouterr().out
    assert (tmp_path / 'out' / 'comparison.md').is_file()


def test_score_command_scores_several_directories(suite, monkeypatch, capsys):
    from rvcbench.benchmark import cli
    directories = two_models(suite)
    monkeypatch.setattr(sys, 'argv', ['rvcbench', 'score', '--suite', str(suite[0]), '--generated', *map(str, directories),
                                      '--output', str(suite[2] / 'cli'), '--data-root', str(suite[1])])
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=fake_evaluate), pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code in (True, 1)  # model-b is partial
    assert '| Task | Metric | model-a | model-b |' in capsys.readouterr().out


class CountingScorer:
    created = 0
    closed = 0
    version = 'test'
    dependencies = ()
    model_provenance = None

    def __init__(self, name, device, logger):
        CountingScorer.created += 1

    def prepare(self):
        pass

    def score(self, request):
        return {'sim': 0.5, 'sva': 1}

    def close(self):
        CountingScorer.closed += 1


def rows_for(tmp_path, name):
    target, generated = tmp_path / f'{name}-t.wav', tmp_path / f'{name}-g.wav'
    for path, scale in ((target, 0.1), (generated, 0.2)):
        sf.write(path, scale * np.sin(np.arange(1600) * 0.05), 16000)
    return [{'sample_id': 'ab' * 32, 'speaker_id': 's1', 'target_path': str(target), 'generated_path': str(generated),
             'target_sha256': file_hash(target), 'generated_sha256': file_hash(generated),
             'target_text': 'hello', 'target_language': 'EN'}]


def test_a_scorer_pool_loads_each_metric_model_once_across_runs(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    CountingScorer.created = CountingScorer.closed = 0
    monkeypatch.setattr(pipeline, 'create_scorer', CountingScorer)
    logger = SimpleNamespace(info=lambda *a: None, warning=lambda *a: None, error=lambda *a: None)
    with pipeline.ScorerPool('cpu', logger) as pool:
        for name in ('first', 'second'):
            rows = rows_for(tmp_path, name)
            pipeline.evaluate_run(rows, ['sim'], tmp_path / name, 'cpu', logger,
                                  bootstrap_config={'enabled': False}, scorers=pool)
            assert rows[0]['status'] == 'complete'
        assert (CountingScorer.created, CountingScorer.closed) == (1, 0)
    assert CountingScorer.closed == 1
    pipeline.evaluate_run(rows_for(tmp_path, 'third'), ['sim'], tmp_path / 'third', 'cpu', logger,
                          bootstrap_config={'enabled': False})
    assert (CountingScorer.created, CountingScorer.closed) == (2, 2)  # without a pool: load and release per run


def test_a_gzipped_suite_exports_and_scores_like_a_plain_one(suite):
    import gzip
    spec_path, data_root, tmp = suite
    spec = json.loads(spec_path.read_text())
    for key in ('manifest', 'selection'):
        plain = spec_path.parent / spec['tasks'][0][key]
        with gzip.open(plain.with_name(plain.name + '.gz'), 'wt', encoding='utf-8') as handle:
            handle.write(plain.read_text())
        plain.unlink()
        spec['tasks'][0][key] += '.gz'
    selection = spec_path.parent / spec['tasks'][0]['selection']
    payload = submission.read_json(selection)
    payload['metadata_sha256'] = file_hash(spec_path.parent / spec['tasks'][0]['manifest'])
    with gzip.open(selection, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle)
    spec_path.write_text(json.dumps(spec))
    entries, prompts = export(suite)
    assert len(entries) == len(PAIRS)
    flat_echo(entries, prompts, tmp / 'flat')
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=fake_evaluate):
        result = submission.score_submission(spec_path, tmp / 'flat', tmp / 'scored', model='echo', data_root=data_root)
    assert result['status'] == 'complete'


@pytest.mark.parametrize('fingerprint', ['different-scorer', None])
def test_different_or_missing_scorers_require_explicit_unranked_comparison(tmp_path, fingerprint):
    means = {'clean': {'sim': 0.6, 'wer': 0.1}, 'noisy': {'sim': 0.3, 'wer': 0.2}}
    a = fake_submission(tmp_path / 'a', 'a', means)
    b = fake_submission(tmp_path / 'b', 'b', means)
    path = b / 'submission.json'
    record = json.loads(path.read_text())
    record['tasks']['clean']['evaluation_fingerprint'] = fingerprint
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Incompatible scoring protocols'):
        comparison.compare_submissions([a, b], tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()
    result = comparison.compare_submissions([a, b], allow_incompatible=True)
    assert not result['protocol_compatible'] and not result['leaderboard']
    assert 'unranked inspection' in result['label'] and '**0.600**' not in result['markdown']
