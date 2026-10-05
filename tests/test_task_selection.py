"""Task selection preserves dependencies, frozen inputs and comparison boundaries."""
import json
import sys
from unittest.mock import patch

import pytest

from rvcbench.benchmark import cli, submission
from rvcbench.benchmark.comparison import compare_submissions
from test_submission import suite, echo, fake_evaluate  # noqa: F401


def names(spec):
    return [task['task'] for task in spec['tasks']]


def test_selection_closes_dependencies_and_has_canonical_identity():
    original = submission.load_suite('core-v1')
    selected = submission.load_suite('core-v1', tasks=['background', 'compression-mp3-64k'])
    assert names(selected) == ['audioshift', 'background-clean', 'background', 'compression-mp3-64k']
    assert selected['parent_suite_sha256'] == original['_sha256']
    assert selected['_sha256'] != original['_sha256']
    explicit = submission.load_suite('core-v1', tasks=list(reversed(names(selected))) + ['background'])
    assert submission.suite_record(selected) == submission.suite_record(explicit)
    assert submission.suite_record(original) == submission.suite_record(
        submission.load_suite('core-v1', tasks=names(original)))
    assert names(submission.load_suite('core-v1', tasks='chinese')) == ['chinese']


@pytest.mark.parametrize('tasks', [[], ['typo'], ['chinese', 'typo']])
def test_invalid_selection_is_rejected(tasks):
    with pytest.raises(ValueError, match='Available:'):
        submission.load_suite('core-v1', tasks=tasks)


def test_unavailable_full_suite_task_is_rejected():
    with pytest.raises(ValueError, match='adv-spec'):
        submission.load_suite('full-v1', tasks=['adv-spec'])


def test_tasks_cli_lists_paper_mapping_and_dependencies(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['rvcbench', 'tasks', '--suite', 'core-v1', '--json'])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    background = next(task for task in result['tasks'] if task['task'] == 'background')
    assert background['anchor'] == 'background-clean'
    assert background['evaluation'] == 'RVC-PassiveNoise/Background'
    assert background['required_metrics'] == ['sim', 'speechmos', 'wer', 'mcd']


def test_selected_cli_export_score_resume_and_comparison(suite, monkeypatch, capsys):
    path, data_root, tmp = suite
    spec = json.loads(path.read_text())
    # Deliberately broken unselected task: it must never be read or downloaded.
    spec['tasks'].append({**spec['tasks'][0], 'task': 'unselected', 'manifest': 'absent.json'})
    path.write_text(json.dumps(spec))
    prompts = tmp / 'prompts'
    monkeypatch.setattr(sys, 'argv', ['rvcbench', 'prompts', '--suite', str(path), '--tasks', 'toy',
                                     '--output', str(prompts), '--data-root', str(data_root)])
    cli.main()
    entries = [json.loads(line) for line in (prompts / 'prompts.jsonl').read_text().splitlines()]
    assert {entry['task'] for entry in entries} == {'toy'}
    assert '--tasks toy --generated' in (prompts / 'README.md').read_text()
    generated = tmp / 'generated'
    echo(entries, prompts, generated)
    args = ['rvcbench', 'score', '--suite', str(path), '--tasks', 'toy', '--generated', str(generated),
            '--output', str(tmp / 'scored'), '--data-root', str(data_root)]
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=fake_evaluate):
        monkeypatch.setattr(sys, 'argv', args)
        with pytest.raises(SystemExit) as outcome:
            cli.main()
        assert outcome.value.code == 0
        monkeypatch.setattr(sys, 'argv', args + ['--resume'])
        with pytest.raises(SystemExit) as outcome:
            cli.main()
        assert outcome.value.code == 0
        batch = submission.score_submissions(path, [generated, generated], tmp / 'batch', models=['a', 'b'],
                                             data_root=data_root, tasks=['toy'])
        assert all(model['status'] == 'complete' for model in batch['models'])
        submission.score_submissions(path, [generated, generated], tmp / 'batch', models=['a', 'b'],
                                     data_root=data_root, tasks=['toy'], resume=True)
        (path.parent / 'absent.json').write_bytes((path.parent / 'toy.metadata.json').read_bytes())
        with pytest.raises(ValueError, match='cannot resume'):
            submission.score_submission(path, generated, tmp / 'scored', model='generated',
                                         data_root=data_root, resume=True)
    result = json.loads((tmp / 'scored/submission.json').read_text())
    exported = json.loads((prompts / 'suite.json').read_text())
    assert result['status'] == 'complete' and set(result['tasks']) == {'toy'}
    assert result['selected_tasks'] == ['toy']
    assert result['suite_sha256'] == exported['suite_sha256']
    # Existing comparison checks must reject this selection against a full-suite result.
    result['suite_sha256'] = submission.load_suite(path)['_sha256']
    other = tmp / 'other.json'
    other.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='suite_sha256 differs'):
        compare_submissions([tmp / 'scored', other])


def test_dependency_selection_scores_anchor_and_compression(suite):
    from test_submission_core import rewrite, fake_evaluate as evaluate
    path, data_root, tmp = suite
    rewrite(suite, [
        {'task': 'toy', 'group_by': 'speaker_id'},
        {'task': 'toy-noisy', 'anchor': 'toy'},
        {'task': 'compressed', 'derived_from': 'toy-noisy', 'transform': {'kind': 'narrowband'}},
        {'task': 'unused', 'manifest': 'absent.json'},
    ])
    prompts = tmp / 'prompts'
    submission.export_prompts(path, prompts, tasks=['compressed'], data_root=data_root)
    entries = [json.loads(line) for line in (prompts / 'prompts.jsonl').read_text().splitlines()]
    assert {entry['task'] for entry in entries} == {'toy', 'toy-noisy'}
    echo(entries, prompts, tmp / 'generated')
    # Exercise the dependency flow; codec implementations have separate real-ffmpeg tests.
    import shutil
    with patch('rvcbench.evaluation.pipeline.evaluate_run', side_effect=evaluate), \
         patch.object(submission, '_ffmpeg'), \
         patch.object(submission, 'transform_audio', side_effect=lambda src, dst, *a: shutil.copyfile(src, dst)):
        result = submission.score_submission(path, tmp / 'generated', tmp / 'scored', model='echo',
                                             tasks=['compressed'], data_root=data_root)
    assert result['status'] == 'complete'
    assert list(result['tasks']) == ['toy', 'toy-noisy', 'compressed']
    assert result['tasks']['toy-noisy']['relative_change_percent']['sim'] == pytest.approx(-20)
    assert set(result['tasks']['toy']['group_means']['speaker_id']) == {'s1', 's2'}
