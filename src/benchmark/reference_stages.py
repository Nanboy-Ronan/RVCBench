"""Explicit reference-stage binding without clean-audio fallback."""
from dataclasses import replace
import json
from pathlib import Path

from .artifacts import digest, file_hash, sample_id


def _unique_file(candidates, description):
    matches = {p.resolve() for p in candidates if p.is_file()}
    if not matches:
        raise FileNotFoundError(f'Missing {description}')
    if len(matches) != 1:
        raise ValueError(f'Ambiguous {description}: {sorted(map(str, matches))}')
    return matches.pop()


def bind_reference_stage(samples, dataset_root, reference_root, kind='external_reference'):
    """Keep targets/text/identities intact while binding all selected references.

    Legacy directories have content provenance only. Producer provenance is
    recorded when an adjacent stage_manifest.json exists, without inferring
    the producer from a directory name.
    """
    root = Path(reference_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Reference stage directory does not exist: {root}')
    clean_root = Path(dataset_root).resolve()
    bound, bindings, used = [], [], {}
    for sample in samples:
        declared = sample.extra.get('prompt_file_name')
        if declared:
            relative = Path(str(declared))
            candidates = ([relative] if relative.is_absolute() else
                          [clean_root / relative, clean_root.parent / relative])
        else:
            relative = None
            candidates = [Path(sample.prompt_path)] if sample.prompt_path else []
        clean = _unique_file(candidates, f'clean reference for sample {sample_id(sample)}')
        stage_candidates = [root / str(sample.speaker_id) / clean.name, root / clean.name]
        if relative and not relative.is_absolute() and '..' not in relative.parts:
            stage_candidates.append(root / relative)
        selected = _unique_file(stage_candidates, f'{kind} reference for sample {sample_id(sample)}')
        identity = (str(sample.speaker_id), str(clean))
        if selected in used and used[selected] != identity:
            raise ValueError(f'Reference stage file maps to multiple clean identities: {selected}')
        used[selected] = identity
        row = {'sample_id': sample_id(sample), 'speaker_id': str(sample.speaker_id),
               'clean_prompt_path': str(clean), 'clean_prompt_sha256': file_hash(clean),
               'reference_path': str(selected), 'reference_sha256': file_hash(selected)}
        bindings.append(row)
        bound.append(replace(sample, prompt_path=selected))
    parents = {p.resolve() for p in [root / 'stage_manifest.json', root.parent / 'stage_manifest.json']
               if p.is_file()}
    if len(parents) > 1:
        raise ValueError('Ambiguous reference stage producer manifests')
    parent = next(iter(parents), None)
    producer_status = 'legacy_directory_content_only'
    if parent:
        producer = json.loads(parent.read_text())
        if not isinstance(producer, dict):
            raise ValueError('Invalid reference stage producer manifest')
        if 'status' in producer and producer['status'] != 'complete':
            raise ValueError('Reference stage producer is not complete')
        producer_status = 'manifest_recorded'
        if producer.get('variant') in ('gr_archived_noise_replay_v1', 'dns64_dataset_rate_v1'):
            records = producer.get('rows', [])
            if (producer.get('schema_version') != 1 or producer.get('status') != 'complete' or
                    not records or producer.get('verified') != len(records) or
                    producer.get('requested') != len(records)):
                raise ValueError('Incomplete archived-noise producer verification')
            by_id = {r['sample_id']: r for r in records}
            if len(by_id) != len(records):
                raise ValueError('Duplicate producer sample identities')
            for binding in bindings:
                record = by_id.get(binding['sample_id'])
                if not record or any(record.get(k) != binding[k] for k in
                        ('speaker_id', 'clean_prompt_sha256', 'reference_sha256')):
                    raise ValueError('Reference stage output differs from its producer manifest')
                if (producer['variant'] == 'gr_archived_noise_replay_v1' and
                        record.get('historical_sha256') != binding['reference_sha256']):
                    raise ValueError('Reference stage output is not historically verified')
            producer_status = 'verified_selected_output_hashes'
    lineage = {'kind': str(kind), 'directory': str(root), 'bindings': bindings,
               'producer_manifest': str(parent) if parent else None,
               'producer_manifest_sha256': file_hash(parent) if parent else None,
               'producer_status': producer_status}
    lineage['fingerprint'] = digest({'kind': lineage['kind'],
        'producer_manifest_sha256': lineage['producer_manifest_sha256'],
        'bindings': [{k: r[k] for k in ('sample_id', 'speaker_id', 'clean_prompt_sha256', 'reference_sha256')}
                     for r in bindings]})
    return bound, lineage
