import json
import pytest

from test_benchmark import setup_run, mocked_evaluator, fake_evaluate
from src.benchmark.comparability import check_comparability
from src.benchmark.artifacts import digest


def test_complete_runs_require_matched_scorer_protocol(setup_run):
    conf, _, run, _ = setup_run
    conf.vc.generate_only = False
    directories = []
    with mocked_evaluator(fake_evaluate):
        for name in ('left', 'right'):
            directory, manifest, _ = run(name)
            manifest['config']['vc']['model'] = name
            (directory / 'run_manifest.json').write_text(json.dumps(manifest))
            spec = {'version': 'fixture', 'seed': 42}
            (directory / 'scoring_manifest.json').write_text(json.dumps({'sim': {'fingerprint': digest(spec), **spec}}))
            directories.append(directory)
    assert check_comparability(*directories, ['sim'])['status'] == 'comparable'
    (directories[1] / 'scoring_manifest.json').write_text(json.dumps({'sim': {'fingerprint': 'new-weights'}}))
    result = check_comparability(*directories, ['sim'])
    assert result['status'] == 'incompatible' and any('sim scorer' in r for r in result['reasons'])
    (directories[1] / 'scoring_manifest.json').unlink()
    assert check_comparability(*directories, ['sim'])['status'] == 'incompatible'
    path = directories[1] / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['samples'][0]['target_text'] = 'changed transcript'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Input fingerprint'):
        check_comparability(*directories, ['sim'])
