#!/usr/bin/env python3
"""Read-only audit of LibriTTS audio counts, transcript variants and result provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    path = root / 'metadata.parquet'
    frame = pd.read_parquet(path)
    speaker_rows = frame[frame.source_manifest.eq(frame.speaker_id.astype(str) + '.json')]
    text_rows = frame[frame.source_manifest.eq(frame.speaker_id.astype(str) + '_text.json')]
    keys = ['speaker_id', 'pair_id']
    for label, selected in [('speaker', speaker_rows), ('text', text_rows)]:
        if selected.duplicated(keys).any():
            raise ValueError(f'Duplicate identities within {label} variant')
    pairs = speaker_rows.merge(text_rows, on=keys, suffixes=('_speaker', '_text'), validate='one_to_one')
    differences = {}
    for name in ('prompt', 'target'):
        left, right = pairs[name + '_text_speaker'], pairs[name + '_text_text']
        normalize = lambda s: re.sub(r'[^\w]', '', str(s))
        differences[name] = {
            'text_changed': int((left != right).sum()),
            'changed_beyond_punctuation_and_whitespace': int((left.map(normalize) != right.map(normalize)).sum()),
            'same_audio_path': int((pairs[name + '_file_name_speaker'] == pairs[name + '_file_name_text']).sum()),
        }
    prompts, targets = set(speaker_rows.prompt_file_name), set(speaker_rows.target_file_name)
    return {'manifest_path': str(path.resolve()), 'manifest_sha256': sha256(path),
        'exported_rows': len(frame), 'speakers': int(speaker_rows.speaker_id.nunique()),
        'speaker_manifest_pairs': len(speaker_rows), 'text_manifest_pairs': len(text_rows),
        'matched_variant_pairs': len(pairs),
        'prompt_waveforms': len(prompts), 'target_waveforms': len(targets),
        'unique_waveforms': len(prompts | targets), 'prompt_target_overlap': len(prompts & targets),
        'physical_wav_files': sum(1 for _ in (root / 'audios').rglob('*.wav')),
        'variant_differences': differences}, speaker_rows, text_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('data/Libritts'))
    parser.add_argument('--historical-results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, speaker_rows, text_rows = audit(args.dataset_root)
    result['paper_reference'] = 'https://arxiv.org/html/2602.00443v2#A2.SS1'
    result['interpretation'] = {
        'paper_utterance_count': 4000,
        'historical_evaluation_population': '2000 reference-target pairs from speaker.json',
        'paper_prose_discrepancy': 'B.1 says 100 paired entries per speaker; local manifests have 50 pairs using 100 distinct waveforms.',
        'variant_origin': 'Creation script not located; punctuation and phonetic differences are observed, original intent remains unverified.',
        'preservation': 'Audit reads existing datasets and results without modifying them.',
    }
    if args.historical_results:
        lookup = {Path(r.target_file_name).name: r.target_text for r in speaker_rows.itertuples()}
        alternate = {Path(r.target_file_name).name: r.target_text for r in text_rows.itertuples()}
        results = []
        for csv in sorted(args.historical_results.glob('*_ots_on_libr*/*/generation_sample_metrics.csv')):
            if csv.parent.parent.name.split('_ots_on_')[-1] not in ('libratts', 'libritts'):
                continue
            scores = pd.read_csv(csv)
            matches = {'distinguishable_targets': 0, 'speaker_transcript': 0, 'text_transcript': 0, 'neither': 0}
            for row in scores.itertuples():
                name = Path(row.ground_truth_path).name
                if name not in lookup or name not in alternate or lookup[name] == alternate[name]:
                    continue
                matches['distinguishable_targets'] += 1
                matches['speaker_transcript'] += row.ground_truth_text == lookup[name]
                matches['text_transcript'] += row.ground_truth_text == alternate[name]
                matches['neither'] += row.ground_truth_text not in (lookup[name], alternate[name])
            results.append({'csv': str(csv.resolve()), 'sha256': sha256(csv), 'rows': len(scores), **matches})
        result['historical_results'] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
