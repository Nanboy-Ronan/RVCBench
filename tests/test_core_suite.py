import json

import pytest

from rvcbench.benchmark import submission
from rvcbench.benchmark.artifacts import file_hash, input_fingerprint


@pytest.fixture(scope='module')
def core():
    return submission.load_suite('core-v1')


def generated(spec):
    return [task for task in spec['tasks'] if not task.get('derived_from')]


def test_core_is_pinned_and_not_yet_a_leaderboard_suite(core):
    assert core['suite'] == 'core-v1' and core['leaderboard'] is False
    assert len(core['hf_revision']) == 40 and core['paper'].endswith('2602.00443')
    assert set(core['not_included']) == {'RVC-Detectability/GroundTruth, Deepfake', 'RVC-Expression/Persuasion EmTXT'}


def test_core_covers_the_papers_evaluations(core):
    labels = ' '.join(task['evaluation'] for task in core['tasks'])
    for evaluation in ('RVC-AudioShift/Demography', 'RVC-TextShift/Hallucination', 'RVC-TextShift/Scam',
                       'RVC-Expression/Persuasion', 'RVC-Multilingual/English-VC', 'RVC-Multilingual/Chinese-VC',
                       'RVC-Multilingual/CrossLingual', 'RVC-LongContext/LongText', 'RVC-LongContext/LongAudio',
                       'RVC-Compression/CodecCompression', 'RVC-Compression/NarrowBand', 'RVC-PassiveNoise/Background',
                       'RVC-PassiveNoise/MultiSpeaker', 'RVC-AdvNoise/Adversary', 'RVC-AdvNoise/Gaussian',
                       'RVC-AntiProtect/AntiProtection'):
        assert evaluation in labels, evaluation
    assert {task['dimension'] for task in core['tasks']} == {'input', 'generation', 'output', 'perturbation'}
    assert len(generated(core)) == 22 and len(core['tasks']) == 29


def test_core_task_files_are_internally_consistent(core):
    total = 0
    for task in generated(core):
        manifest = core['_directory'] / task['manifest']
        selection = json.loads((core['_directory'] / task['selection']).read_text())
        assert selection['metadata_sha256'] == file_hash(manifest), task['task']
        assert input_fingerprint(selection['samples']) == selection['input_fingerprint'], task['task']
        assert len(json.loads(manifest.read_text())) == len(selection['samples']) == selection['requested']
        total += selection['requested']
    assert total == 480


def test_perturbation_tasks_share_pairs_with_their_clean_anchor(core):
    def rows(name):
        task = next(t for t in core['tasks'] if t['task'] == name)
        return [(r['pair_id'], r['target_file_name']) for r in json.loads((core['_directory'] / task['manifest']).read_text())]
    for task in core['tasks']:
        if task['task'].startswith(('adv-', 'antiprotect', 'background', 'multispeaker')) and task.get('anchor'):
            assert [target for _, target in rows(task['task'])] == [target for _, target in rows(task['anchor'])], task['task']


def test_core_metrics_follow_the_paper(core):
    by_name = {task['task']: task for task in core['tasks']}
    assert core['evaluation']['required_metrics'] == ['sim', 'speechmos', 'wer', 'mcd']
    for name in ('textshift-scam', 'textshift-scam-standard'):
        assert 'mcd' not in by_name[name]['required_metrics'] and 'emotion' in by_name[name]['required_metrics']
    for task in core['tasks']:
        if task.get('derived_from'):
            assert task['derived_from'] == 'audioshift' and task['required_metrics'] == ['stoi', 'mcd', 'sim', 'wer']
