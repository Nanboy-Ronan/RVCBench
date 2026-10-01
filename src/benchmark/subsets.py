"""Deterministic, outcome-independent subset selection."""
from pathlib import Path
from .artifacts import atomic_json, digest, file_hash, input_records, input_fingerprint, sample_id


def freeze(dataset, output, speakers=8, pairs_per_speaker=2, seed=20260930):
    samples = dataset.get_zero_shot_samples()
    groups = {}
    for sample in samples:
        groups.setdefault(sample.speaker_id, []).append(sample)
    if speakers <= 0 or pairs_per_speaker <= 0 or len(groups) < speakers:
        raise ValueError('Invalid subset size or insufficient speakers')
    selected_speakers = sorted(groups, key=lambda sid: digest([seed, sid]))[:speakers]
    selected = []
    for sid in selected_speakers:
        if len(groups[sid]) < pairs_per_speaker:
            raise ValueError(f'Insufficient pairs for speaker {sid}; do not silently substitute')
        selected += sorted(groups[sid], key=lambda s: digest([seed, sample_id(s)]))[:pairs_per_speaker]
    selected.sort(key=lambda s: s.index)
    records = input_records(selected)
    if any(not r['prompt_sha256'] or not r['target_sha256'] for r in records):
        raise ValueError('All selected audio must be present before freezing')
    metadata = []
    fields = ('dataset_name', 'split', 'pair_id', 'speaker_id', 'manifest_variant', 'source_manifest', 'source_row',
              'prompt_file_name', 'target_file_name', 'prompt_text', 'target_text', 'prompt_language', 'target_language')
    for sample in selected:
        metadata.append({**{k: sample.extra.get(k) for k in fields}, 'source_index': sample.index})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'metadata.json', metadata)
    atomic_json(output / 'selection.json', {
        'schema_version': 1, 'selection': 'sha256_rank_speakers_then_pairs_v1', 'selection_seed': seed,
        'speakers': speakers, 'pairs_per_speaker': pairs_per_speaker, 'requested': len(selected),
        'population': len(samples), 'variant_selection': dataset.variant_selection,
        'input_fingerprint': input_fingerprint(records), 'metadata_sha256': file_hash(output / 'metadata.json'),
        'samples': [{k: v for k, v in r.items() if k not in ('prompt_path', 'target_path', 'status', 'error')}
                    for r in records],
        'interpretation': 'Matched-subset regression evidence. Does not establish full-table reproduction or model ranking.',
    })
    return output
