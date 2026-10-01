from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import soundfile as sf

from test_benchmark import setup_run
from src.benchmark.artifacts import file_hash
from src.benchmark.reference_stages import bind_reference_stage


def stage_directory(dataset, tmp_path):
    root = tmp_path / 'stage-audio'
    original = dataset.get_zero_shot_samples()[0].prompt_path
    audio, rate = sf.read(original)
    path = root / 'one' / original.name
    path.parent.mkdir(parents=True)
    sf.write(path, audio * .5, rate)
    return root, original, path


def test_runner_binds_stage_and_preserves_clean_target_identity_and_dataset(setup_run):
    conf, dataset, run, tmp_path = setup_run
    root, clean, staged = stage_directory(dataset, tmp_path)
    _, baseline, _ = run('clean')
    conf.vc.reference_audio_dir = str(root)
    conf.vc.reference_stage = 'protected'
    _, manifest, _ = run('protected')
    assert manifest['reference_stage']['producer_status'] == 'legacy_directory_content_only'
    assert manifest['input_fingerprint'] != baseline['input_fingerprint']
    for row, old in zip(manifest['samples'], baseline['samples']):
        assert row['sample_id'] == old['sample_id']
        assert row['target_sha256'] == old['target_sha256']
        assert row['clean_prompt_sha256'] == file_hash(clean)
        assert row['prompt_sha256'] == row['generated_sha256'] == file_hash(staged)
    assert dataset.get_zero_shot_samples()[0].prompt_path == clean


@pytest.mark.parametrize('ambiguous', [False, True])
def test_stage_missing_or_ambiguous_reference_prevents_model_loading(setup_run, ambiguous):
    conf, dataset, run, tmp_path = setup_run
    root, clean, staged = stage_directory(dataset, tmp_path)
    conf.vc.reference_audio_dir = str(root)
    if ambiguous:
        (root / clean.name).write_bytes(staged.read_bytes())
    else:
        staged.unlink()
    with patch('src.benchmark.backends.select_adversary', side_effect=AssertionError('must not load')):
        with pytest.raises((FileNotFoundError, ValueError), match='reference'):
            run('bad-stage')


def test_stage_rejects_flat_file_shared_between_speakers(setup_run):
    _, dataset, _, tmp_path = setup_run
    root, clean, staged = stage_directory(dataset, tmp_path)
    staged.unlink()
    (root / clean.name).write_bytes(clean.read_bytes())
    sample = dataset.get_zero_shot_samples()[0]
    other = replace(sample, speaker_id='other', extra={**sample.extra, 'pair_id': 'other'})
    with pytest.raises(ValueError, match='multiple clean identities'):
        bind_reference_stage([sample, other], dataset._dataset_root, root)


def test_resume_rejects_clean_lineage_change_even_with_identical_stage_and_target(setup_run):
    conf, dataset, run, tmp_path = setup_run
    root, clean, _ = stage_directory(dataset, tmp_path)
    target = clean.with_name('target.wav')
    target.write_bytes(clean.read_bytes())
    dataset._zero_shot_samples = [replace(s, target_path=target) for s in dataset.get_zero_shot_samples()]
    conf.vc.reference_audio_dir = str(root)
    directory, _, _ = run('protected')
    audio, rate = sf.read(clean)
    sf.write(clean, audio * .8, rate)
    conf.vc.resume_from = str(directory)
    with pytest.raises(ValueError, match='reference stage lineage differs'):
        run('changed-clean-source')


def test_producer_manifest_changes_lineage_without_changing_audio(setup_run):
    _, dataset, _, tmp_path = setup_run
    root, _, _ = stage_directory(dataset, tmp_path)
    producer = root.parent / 'stage_manifest.json'
    producer.write_text(json.dumps({'producer': 'first'}))
    samples = dataset.get_zero_shot_samples()
    _, before = bind_reference_stage(samples, dataset._dataset_root, root)
    producer.write_text(json.dumps({'producer': 'second'}))
    _, after = bind_reference_stage(samples, dataset._dataset_root, root)
    assert before['producer_status'] == 'manifest_recorded'
    assert before['bindings'] == after['bindings']
    assert before['fingerprint'] != after['fingerprint']
    (root / 'stage_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='Ambiguous reference stage producer'):
        bind_reference_stage(samples, dataset._dataset_root, root)


@pytest.mark.parametrize('failure', ['failed', 'stale_output', 'wrong_count', None])
@pytest.mark.parametrize('variant', ['gr_archived_noise_replay_v1', 'gr_seeded_batch_rng_v1', 'dns64_dataset_rate_v1'])
def test_archived_producer_must_be_complete_and_match_selected_outputs(setup_run, failure, variant):
    conf, dataset, run, tmp_path = setup_run
    root, _, _ = stage_directory(dataset, tmp_path)
    _, baseline = bind_reference_stage(dataset.get_zero_shot_samples(), dataset._dataset_root, root)
    rows = [{**b, 'historical_sha256': b['reference_sha256']} for b in baseline['bindings']]
    producer = {'schema_version': 1, 'variant': variant,
                'status': 'complete', 'requested': len(rows), 'verified': len(rows), 'rows': rows}
    if variant == 'gr_seeded_batch_rng_v1':
        producer['rng_verification'] = {'requested_batches': 1, 'verified_batches': 1}
    if failure == 'failed':
        producer['status'] = 'failed'
    elif failure == 'stale_output':
        rows[0]['reference_sha256'] = 'stale'
    elif failure == 'wrong_count':
        producer['verified'] = 0
    (root / 'stage_manifest.json').write_text(json.dumps(producer))
    conf.vc.reference_audio_dir = str(root)
    if failure:
        with patch('src.benchmark.backends.select_adversary', side_effect=AssertionError('must not load')):
            with pytest.raises(ValueError):
                run('bad-producer')
    else:
        _, manifest, _ = run('verified-producer')
        assert manifest['reference_stage']['producer_status'] == 'verified_selected_output_hashes'


def test_seeded_producer_requires_full_rng_verification(setup_run):
    _, dataset, _, tmp_path = setup_run
    root, _, _ = stage_directory(dataset, tmp_path)
    samples = dataset.get_zero_shot_samples()
    _, baseline = bind_reference_stage(samples, dataset._dataset_root, root)
    rows = [{**b, 'historical_sha256': b['reference_sha256']} for b in baseline['bindings']]
    producer = {'schema_version': 1, 'variant': 'gr_seeded_batch_rng_v1',
                'status': 'complete', 'requested': len(rows), 'verified': len(rows), 'rows': rows,
                'rng_verification': {'requested_batches': 2, 'verified_batches': 1}}
    (root / 'stage_manifest.json').write_text(json.dumps(producer))
    with pytest.raises(ValueError, match='regenerated RNG'):
        bind_reference_stage(samples, dataset._dataset_root, root)
