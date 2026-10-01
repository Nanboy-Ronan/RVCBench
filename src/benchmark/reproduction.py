"""Explicit matched-pair checks against audited speaker-manifest result artifacts."""
import csv
import json
import re
from pathlib import Path

from .artifacts import atomic_json, file_hash, finite, load_run, validate_report


def historical_prompt_ledger(historical_csv, rows):
    """Recover target-only legacy names only with a complete, aligned request log."""
    pattern = re.compile(r'\[[^\]]+\] \[(\d+)/(\d+)\] speaker=(\S+) prompt=(\S+) .*?text="(.*?)"(?: transcript=|$)')
    ledgers = []
    for path in sorted(Path(historical_csv).parent.glob('*.log')):
        requests = [match for line in path.read_text().splitlines() if (match := pattern.search(line))]
        if len(requests) != len(rows):
            continue
        log_sha256 = file_hash(path)
        ledger = {}
        for index, (request, row) in enumerate(zip(requests, rows), 1):
            text = ' '.join(row['ground_truth_text'].strip().split())
            preview = text if len(text) <= 120 else text[:119] + '…'
            if (int(request[1]) != index or int(request[2]) != len(rows)
                    or request[3] != row['speaker_id'] or request[5] != preview):
                break
            ledger[row['generated_path']] = {'prompt_name': request[4],
                'source_log': str(path.resolve()), 'source_log_sha256': log_sha256}
        else:
            if len(ledger) == len(rows):
                ledgers.append(ledger)
    if len(ledgers) != 1:
        raise ValueError('Target-only legacy names require one complete, CSV-aligned generation log')
    return ledgers[0]


def infer_english_wer_protocol(historical_csv):
    """Verify the recorded formula against every saved English transcript."""
    import jiwer
    from src.evaluation.scorers.wer import normalize_english
    with Path(historical_csv).open() as handle:
        rows = list(csv.DictReader(handle))
    protocols = ('lowercase_v1', 'ascii_punctuation_removed_v2')
    counts = {p: sum(abs(jiwer.wer(normalize_english(r['ground_truth_text'], p),
                                  normalize_english(r['predicted_text'], p)) - float(r['wer'])) < 1e-8
                     for r in rows) for p in protocols}
    matched = [p for p in protocols if rows and counts[p] == len(rows)]
    if len(matched) != 1:
        raise ValueError(f'Historical WER protocol is unsupported or ambiguous: {counts}')
    return {'normalization': matched[0], 'rows': len(rows), 'formula_matches': counts}


def historical_manifest_prompt(root, row, old):
    """Resolve a legacy cohort whose manifest points outside its audio directory.

    A filename alone cannot establish the reference: require one matching source
    record and hash its target against the historical CSV before using its prompt.
    """
    path = root / 'filelists' / f'{row["speaker_id"]}.json'
    records = json.loads(path.read_text())
    matches = []
    for index, record in enumerate(records):
        prompt = Path(str(record.get('ori_pth') or record.get('prompt_file_name') or ''))
        target = Path(str(record.get('gt_pth') or record.get('target_file_name') or ''))
        text = record.get('gt_text', record.get('target_text'))
        if (prompt.name == Path(row['prompt_path']).name
                and target.name == Path(row['target_path']).name and text == old['ground_truth_text']):
            def resolve(value):
                if value.is_absolute():
                    return value
                # Same ordered search as the historical zero-shot loader.
                choices = [root.parent / value, root / value]
                return next((p.resolve() for p in choices if p.is_file()), choices[0].resolve())
            matches.append((index, resolve(prompt), resolve(target)))
    if len(matches) != 1:
        raise ValueError('Historical external-audio cohort requires one exact source manifest pair')
    index, prompt, target = matches[0]
    if file_hash(target) != file_hash(old['ground_truth_path']):
        raise ValueError('Historical source manifest target differs from recorded CSV audio')
    return prompt, {'source': 'speaker_manifest_external_audio', 'source_manifest': str(path.resolve()),
                    'source_manifest_sha256': file_hash(path), 'source_row': index}


def match_historical(run, historical_csv):
    historical_csv = Path(historical_csv).resolve()
    historical = json.loads((historical_csv.parent / 'metrics.json').read_text())
    dataset_name = str(historical['config']['dataset']['name']).lower()
    if dataset_name not in ('libritts', 'vctk'):
        raise ValueError('This matcher requires the audited LibriTTS or VCTK pair naming protocol')
    if str(run['config']['dataset']['name']).lower() != dataset_name:
        raise ValueError('Historical dataset identity differs from the current run')
    aliases = {'qwentts': 'qwen3_tts', 'xtts_v2': 'xtts', 'glmtts': 'glm_tts', 'cozyvoice': 'cosyvoice'}
    left, right = str(run['config']['vc']['model']), str(historical['config']['vc']['model'])
    if aliases.get(left, left) != aliases.get(right, right):
        raise ValueError('Historical model identity differs from the current run')
    if historical.get('vc', {}).get('protected_audio_path'):
        raise ValueError('A protected run cannot be used as a clean baseline')
    root = Path(historical['config']['dataset']['root_path'])
    if not root.is_absolute():
        root = historical_csv.parents[3] / root
    with historical_csv.open() as f:
        old_rows = list(csv.DictReader(f))
    by_name = {}
    for row in old_rows:
        by_name.setdefault(Path(row['generated_path']).name, []).append(row)
    matches = []
    ledger = None
    for row in run['samples']:
        expected = Path(row['prompt_path']).stem + '_to_' + Path(row['target_path']).stem + '_cloned.wav'
        candidates = by_name.get(expected, [])
        if not candidates:
            legacy_name = Path(row['target_path']).stem + '_cloned.wav'
            candidates = by_name.get(legacy_name, [])
            if len(candidates) == 1:
                if ledger is None:
                    ledger = historical_prompt_ledger(historical_csv, old_rows)
                evidence = ledger[candidates[0]['generated_path']]
                if evidence['prompt_name'] != Path(row['prompt_path']).name:
                    raise ValueError(f'Historical reference audio differs: {row["sample_id"]}')
                candidates[0] = dict(candidates[0], prompt_evidence=evidence)
        if len(candidates) != 1:
            raise ValueError(f'Expected one exact historical pair for {row["sample_id"]}; found {len(candidates)}')
        old = candidates[0]
        prompt = root / 'audios' / row['speaker_id'] / Path(row['prompt_path']).name
        if not prompt.is_file():
            prompt, evidence = historical_manifest_prompt(root, row, old)
            old = dict(old, prompt_evidence=evidence)
        if (old['speaker_id'] != row['speaker_id'] or old['ground_truth_text'] != row['target_text']
                or file_hash(old['ground_truth_path']) != row['target_sha256']
                or file_hash(prompt) != row['prompt_sha256']):
            raise ValueError(f'Historical input/content mismatch: {row["sample_id"]}')
        matches.append((row, old))
    return historical, matches


def compare(run_dir, historical_csv, output):
    import numpy as np
    run = load_run(run_dir)
    validate_report(run)
    historical, pairs = match_historical(run, historical_csv)
    historical_wer_protocol = infer_english_wer_protocol(historical_csv)
    scoring = json.loads((Path(run_dir) / 'scoring_manifest.json').read_text())
    wer_version = scoring['wer']['version']
    current_normalization = (wer_version.removeprefix('whisper_medium_')
        if wer_version != 'whisper_medium_historical_normalization_v1' else 'ascii_punctuation_removed_v2')
    if current_normalization != historical_wer_protocol['normalization']:
        raise ValueError('WER normalization differs from historical results; rescore with evaluation.wer_normalization=' +
                         historical_wer_protocol['normalization'])
    speakers = sorted({r['speaker_id'] for r, _ in pairs})
    metrics, samples = {}, []
    for metric in ('mcd', 'wer', 'sim'):
        if any(not finite(new.get('metrics', {}).get(metric)) or not finite(old.get(metric)) for new, old in pairs):
            raise ValueError(f'Incomplete matched metric: {metric}')
        new_values = np.array([float(new['metrics'][metric]) for new, old in pairs])
        old_values = np.array([float(old[metric]) for new, old in pairs])
        delta = new_values - old_values
        groups = [np.array([i for i, (r, _) in enumerate(pairs) if r['speaker_id'] == speaker]) for speaker in speakers]
        rng = np.random.default_rng(20260930)
        draws = [float(delta[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])].mean()) for _ in range(2000)]
        metrics[metric] = {'current_mean': float(new_values.mean()), 'historical_mean': float(old_values.mean()),
            'paired_mean_delta': float(delta.mean()), 'mean_absolute_delta': float(np.abs(delta).mean()),
            'speaker_bootstrap_95_ci_delta': np.quantile(draws, [.025, .975]).tolist()}
    for new, old in pairs:
        samples.append({'sample_id': new['sample_id'], 'speaker_id': new['speaker_id'],
                       'historical_generated_path': old['generated_path'],
                       'historical_generated_sha256': file_hash(old['generated_path']),
                       'historical_prompt_evidence': old.get('prompt_evidence', {'source': 'pair_encoded_filename'}),
                       'current': {m: float(new['metrics'][m]) for m in metrics},
                       'historical': {m: float(old[m]) for m in metrics}})
    report = {'status': 'matched_subset_comparison', 'model': run['config']['vc']['model'],
        'matched_pairs': len(pairs), 'speakers': len(speakers),
        'run_manifest': str(Path(run_dir).resolve() / 'run_manifest.json'),
        'run_manifest_sha256': file_hash(Path(run_dir) / 'run_manifest.json'),
        'historical_csv': str(Path(historical_csv).resolve()), 'historical_csv_sha256': file_hash(historical_csv),
        'historical_config': historical['config'], 'metrics': metrics, 'samples': samples,
        'historical_wer_protocol': historical_wer_protocol,
        'interpretation': 'Descriptive paired regression check, not an equivalence test or full-table reproduction claim.',
        'historical_provenance_limit': 'Legacy result does not provide modern per-sample generation, weight and scorer fingerprints.'}
    atomic_json(output, report)
    return report
