"""Compare recorded generation sources without loading models or altering runs."""
import hashlib
import json
import re
from pathlib import Path

from .artifacts import digest, file_hash


def audit_recorded_sources(run_dir, root):
    manifest_path = Path(run_dir) / 'run_manifest.json'
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    provenance = manifest.get('generation_provenance') or {}
    files = provenance.get('source_files') if isinstance(provenance, dict) else None
    result = {
        'source_manifest': str(manifest_path.resolve()),
        'source_manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
        'root': str(Path(root).resolve()),
        'status': 'source_provenance_unavailable',
        'files': [],
        'limitations': [
            'Checks only files recorded by this run; does not establish completeness of current import coverage.',
            'Does not verify packages, weights, audio, metrics or native generation equivalence.',
            'Source changes do not invalidate historical results produced by the recorded source version.',
        ],
    }
    if not isinstance(files, dict) or not files:
        return result
    if any(not isinstance(name, str) or not name or not isinstance(sha, str)
           or not re.fullmatch(r'[0-9a-f]{64}', sha) for name, sha in files.items()):
        result['status'] = 'invalid_source_provenance'
        return result
    if digest(files) != provenance.get('source_sha256'):
        result['status'] = 'invalid_source_provenance'
        return result
    root = Path(root).resolve()
    for name, expected in sorted(files.items()):
        path = Path(name)
        record = {'recorded_path': name, 'expected_sha256': expected}
        if not path.is_absolute():
            path = (root / path).resolve()
            if not path.is_relative_to(root):
                record['status'] = 'invalid_relative_path'
                result['files'].append(record)
                continue
        record['checked_path'] = str(path)
        try:
            if not path.is_file():
                record['status'] = 'missing'
            else:
                actual = file_hash(path)
                record.update(actual_sha256=actual,
                              status='match' if actual == expected else 'changed')
        except OSError as exc:
            record.update(status='unreadable', error=f'{type(exc).__name__}: {exc}')
        result['files'].append(record)
    statuses = [record['status'] for record in result['files']]
    result['counts'] = {status: statuses.count(status) for status in sorted(set(statuses))}
    result['status'] = ('recorded_sources_match' if all(status == 'match' for status in statuses)
                        else 'recorded_sources_differ')
    return result
