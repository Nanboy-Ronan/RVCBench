import json
import sys

import pytest

from rvcbench.benchmark.artifacts import digest, file_hash
from rvcbench.benchmark.source_audit import audit_recorded_sources


@pytest.fixture
def recorded(tmp_path):
    root, run = tmp_path / 'checkout', tmp_path / 'run'
    (root / 'src').mkdir(parents=True)
    run.mkdir()
    (root / 'src' / 'adapter.py').write_text('ADAPTER = 1\n')
    (root / 'src' / 'generator.py').write_text('GENERATOR = 1\n')

    def write(files=None, **provenance):
        if files is None:
            files = {name: file_hash(root / name) for name in ('src/adapter.py', 'src/generator.py')}
        provenance = {'source_files': files, 'source_sha256': digest(files), **provenance}
        (run / 'run_manifest.json').write_text(json.dumps({'generation_provenance': provenance}))
        return files

    return root, run, write


def statuses(result):
    return {record['recorded_path']: record['status'] for record in result['files']}


def test_unchanged_recorded_sources_match(recorded):
    root, run, write = recorded
    write()
    before = file_hash(run / 'run_manifest.json')
    result = audit_recorded_sources(run, root)
    assert result['status'] == 'recorded_sources_match'
    assert result['counts'] == {'match': 2}
    assert result['source_manifest_sha256'] == before == file_hash(run / 'run_manifest.json')
    assert sorted(path.name for path in run.iterdir()) == ['run_manifest.json']


def test_changed_and_missing_sources_differ(recorded):
    root, run, write = recorded
    write()
    (root / 'src' / 'adapter.py').write_text('ADAPTER = 2\n')
    (root / 'src' / 'generator.py').unlink()
    result = audit_recorded_sources(run, root)
    assert result['status'] == 'recorded_sources_differ'
    assert statuses(result) == {'src/adapter.py': 'changed', 'src/generator.py': 'missing'}
    assert result['counts'] == {'changed': 1, 'missing': 1}


@pytest.mark.parametrize('manifest', [{}, {'generation_provenance': None},
                                      {'generation_provenance': {'source_files': {}}},
                                      {'generation_provenance': {'source_files': ['src/adapter.py']}}])
def test_runs_without_recorded_sources_are_not_reported_as_matching(recorded, manifest):
    root, run, _ = recorded
    (run / 'run_manifest.json').write_text(json.dumps(manifest))
    result = audit_recorded_sources(run, root)
    assert result['status'] == 'source_provenance_unavailable'
    assert result['files'] == []


@pytest.mark.parametrize('sha', ['A' * 64, 'a' * 63, '', None])
def test_malformed_recorded_hash_is_invalid(recorded, sha):
    root, run, write = recorded
    write({'src/adapter.py': sha})
    assert audit_recorded_sources(run, root)['status'] == 'invalid_source_provenance'


def test_edited_file_list_is_invalid(recorded):
    root, run, write = recorded
    files = write()
    write(files, source_sha256=digest({**files, 'src/removed.py': 'a' * 64}))
    result = audit_recorded_sources(run, root)
    assert result['status'] == 'invalid_source_provenance'
    assert result['files'] == []


def test_relative_path_cannot_leave_the_checkout(recorded, tmp_path):
    root, run, write = recorded
    outside = tmp_path / 'outside.py'
    outside.write_text('OUTSIDE = 1\n')
    write({'../outside.py': file_hash(outside)})
    result = audit_recorded_sources(run, root)
    assert result['status'] == 'recorded_sources_differ'
    assert statuses(result) == {'../outside.py': 'invalid_relative_path'}
    assert 'checked_path' not in result['files'][0]


def test_absolute_upstream_path_is_checked_at_its_recorded_location(recorded, tmp_path):
    root, run, write = recorded
    upstream = tmp_path / 'upstream' / 'model.py'
    upstream.parent.mkdir()
    upstream.write_text('MODEL = 1\n')
    write({str(upstream): file_hash(upstream)})
    assert audit_recorded_sources(run, root)['status'] == 'recorded_sources_match'
    upstream.write_text('MODEL = 2\n')
    assert statuses(audit_recorded_sources(run, root)) == {str(upstream): 'changed'}


@pytest.mark.parametrize('change,expected', [(False, 0), (True, 1)])
def test_cli_exit_status_and_report(recorded, tmp_path, monkeypatch, capsys, change, expected):
    from rvcbench.benchmark import cli
    root, run, write = recorded
    write()
    if change:
        (root / 'src' / 'adapter.py').write_text('ADAPTER = 2\n')
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['rvcbench', 'audit-source', str(run), '--root', str(root), '--output', str(report)])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert int(stopped.value.code) == expected
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(report.read_text())
    assert printed['status'] == ('recorded_sources_differ' if change else 'recorded_sources_match')
