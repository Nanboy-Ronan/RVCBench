"""Legacy reference identity must be proved, including target-only file names."""
from pathlib import Path

import pytest

from src.benchmark.reproduction import historical_prompt_ledger


def test_target_only_ledger_requires_complete_ordered_log(tmp_path):
    rows = [dict(speaker_id='one', ground_truth_text='First sentence.', generated_path='first.wav'),
            dict(speaker_id='two', ground_truth_text='Second sentence.', generated_path='second.wav')]
    log = tmp_path / 'generation.log'
    log.write_text('[FishSpeech] [1/2] speaker=one prompt=ref1.wav (sr=24000) text="First sentence."\n'
                   '[FishSpeech] [2/2] speaker=two prompt=ref2.wav (sr=24000) text="Second sentence."\n')
    ledger = historical_prompt_ledger(tmp_path / 'scores.csv', rows)
    assert ledger['first.wav']['prompt_name'] == 'ref1.wav'
    assert ledger['first.wav']['source_log_sha256']
    with pytest.raises(ValueError, match='complete, CSV-aligned'):
        historical_prompt_ledger(tmp_path / 'scores.csv', list(reversed(rows)))
    log.write_text(log.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='complete, CSV-aligned'):
        historical_prompt_ledger(tmp_path / 'scores.csv', rows)


def test_wer_protocol_inferred_from_all_transcripts(tmp_path):
    import csv
    from src.benchmark.reproduction import infer_english_wer_protocol
    path = tmp_path / 'scores.csv'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ground_truth_text', 'predicted_text', 'wer'])
        writer.writeheader()
        writer.writerows([dict(ground_truth_text='Hello, world.', predicted_text='Hello world', wer=1),
                          dict(ground_truth_text='Same.', predicted_text='Same.', wer=0)])
    assert infer_english_wer_protocol(path)['normalization'] == 'lowercase_v1'
    with path.open('a') as handle:
        handle.write('different,words,0\n')
    with pytest.raises(ValueError, match='unsupported or ambiguous'):
        infer_english_wer_protocol(path)
