import csv
from contextlib import contextmanager
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from src.benchmark.artifacts import (coverage, input_records, output_path, sample_id,
                                     validate_report)
from src.benchmark.runner import run_zero_shot
from src.datasets.zero_shot import ZeroShotDataset
from src.adversary.smoke import SmokeAdversary


@contextmanager
def mocked_evaluator(function):
    def bridge(rows, required, directory, device, logger, **kwargs):
        pairs = [(Path(r['target_path']), Path(r['generated_path']),
                  {'sample_id': r['sample_id'], 'speaker_id': r['speaker_id'], 'text': r['target_text']})
                 for r in rows if r.get('generated_sha256')]
        result = function(pairs, directory / 'generated_audio', device, logger)
        with Path(result['sample_metrics_csv']).open() as f:
            scores = {r['sample_id']: r for r in csv.DictReader(f)}
        for row in rows:
            if row.get('generated_sha256'):
                row['metrics'] = scores.get(row['sample_id'], {})
                row['status'] = 'complete'
        result['evaluation_fingerprint'] = 'test-only-fingerprint'
        return result
    with patch('src.evaluation.pipeline.evaluate_run', side_effect=bridge):
        yield


@pytest.fixture
def setup_run(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    sf.write(root / 'a.wav', np.sin(np.arange(1600) * .1) * .1, 16000)
    rows = [dict(pair_id=f'p{i}', speaker_id=speaker, prompt_file_name='a.wav',
                 target_file_name='a.wav', prompt_text='hello', target_text='hello')
            for i, speaker in enumerate(['one', 'one', 'two'])]
    (root / 'metadata.json').write_text(json.dumps(rows))
    conf = OmegaConf.create({'base_dir': '.', 'seed': 42, 'vc': {'mode': 'ots', 'model': 'smoke', 'generate_only': True},
        'adversary': {}, 'dataset': {'root_path': str(root), 'use_hf_dataset': False,
        'manifest_filename': 'metadata.json', 'speaker_id': 'one'}, 'evaluation': {'required_metrics': ['sim']}})
    logger = logging.getLogger('test-run')
    dataset = ZeroShotDataset(conf, conf.dataset, logger)
    def run(name, config=None, data=None):
        out = tmp_path / name
        out.mkdir()
        result = run_zero_shot(config or conf, Path.cwd(), 'cpu', data or dataset, out, logger)
        return out, json.loads((out / 'run_manifest.json').read_text()), result
    return conf, dataset, run, tmp_path


def test_speaker_and_limits(setup_run):
    _, dataset, _, _ = setup_run
    assert len(dataset.get_zero_shot_samples()) == 2
    assert {s.speaker_id for s in dataset.get_zero_shot_samples()} == {'one'}
    with pytest.raises(ValueError):
        dataset.get_zero_shot_samples(max_samples=0)


def test_identity_and_duplicate_detection(setup_run):
    _, dataset, _, _ = setup_run
    a, b = dataset.get_zero_shot_samples()
    assert sample_id(a) != sample_id(b)  # same filenames, distinct pairs
    assert output_path('/a', a).name == output_path('/b', a).name
    with pytest.raises(ValueError, match='Duplicate'):
        input_records([a, a])


def test_generate_and_resume(setup_run):
    conf, _, run, _ = setup_run
    first, old, _ = run('first')
    assert old['status'] == 'generated'
    assert old['coverage']['generated'] == 2
    assert not old['coverage']['eligible_for_comparison']
    conf.vc.resume_from = str(first)
    with patch('src.benchmark.backends.select_adversary', side_effect=AssertionError('must not regenerate')):
        _, new, _ = run('resumed')
    assert all(r['reused'] for r in new['samples'])
    assert (first / 'run_manifest.json').read_text() == json.dumps(old, indent=2, ensure_ascii=False) + '\n'


def test_resume_rejects_changed_config_and_audio(setup_run):
    conf, _, run, root = setup_run
    first, old, _ = run('first')
    conf.vc.resume_from = str(first)
    conf.seed = 999
    with pytest.raises(ValueError, match='settings'):
        run('bad-settings')
    conf.seed = 42
    Path(old['samples'][0]['generated_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        run('bad-audio')
    assert json.loads((root / 'bad-audio/run_manifest.json').read_text())['status'] == 'failed'


def test_native_seed_provenance_survives_resume_and_evaluation(setup_run):
    conf, _, run, _ = setup_run
    class NativeSeedAdapter(SmokeAdversary):
        def attack(self, **kwargs):
            self._generator = SimpleNamespace(last_native_seed=42,
                last_native_requested_seed=41, config=SimpleNamespace(native_seed_policy='legacy_fixed'))
            return super().attack(**kwargs)
    adapter = NativeSeedAdapter(conf, conf.dataset, 'cpu', logging.getLogger())
    with patch('src.benchmark.backends.select_adversary', return_value=adapter):
        first, old, _ = run('native-seed')
    assert [r['seed'] for r in old['samples']] == [42, 43]
    assert all(r['native_seed'] == 42 and r['native_seed_policy'] == 'legacy_fixed' for r in old['samples'])
    conf.vc.resume_from = str(first)
    with patch('src.benchmark.backends.select_adversary', side_effect=AssertionError('must not generate')):
        _, resumed, _ = run('native-resumed')
        conf.vc.generate_only = False
        with mocked_evaluator(fake_evaluate):
            _, scored, _ = run('native-scored')
        del conf.vc.resume_from
        conf.vc.evaluate_only = True
        conf.vc.evaluation = {'generated_audio_dir': str(first / 'generated_audio')}
        with mocked_evaluator(fake_evaluate):
            _, evaluated, _ = run('native-evaluated')
    for manifest in (resumed, scored, evaluated):
        assert all(r['native_seed'] == 42 and r['native_seed_policy'] == 'legacy_fixed' for r in manifest['samples'])
        assert all(r['native_requested_seed'] == 41 for r in manifest['samples'])


def test_retry_and_missing_samples(setup_run):
    conf, dataset, run, _ = setup_run
    class Flaky(SmokeAdversary):
        calls = 0
        def attack(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('temporary failure')
            return super().attack(**kwargs)
    conf.vc.retries = 1
    with patch('src.benchmark.backends.select_adversary', return_value=Flaky(conf, conf.dataset, 'cpu', logging.getLogger())):
        _, manifest, _ = run('retry')
    assert manifest['samples'][0]['attempts'] == 2
    assert manifest['coverage']['generated'] == 2
    class Missing(SmokeAdversary):
        def attack(self, **kwargs):
            pass
    with patch('src.benchmark.backends.select_adversary', return_value=Missing(conf, conf.dataset, 'cpu', logging.getLogger())):
        _, manifest, _ = run('missing')
    assert manifest['status'] == 'partial'
    assert manifest['coverage']['requested'] == 2
    assert manifest['coverage']['generation_failed'] == 2
    assert all(r['error'] for r in manifest['samples'])


def fake_evaluate(pairs, directory, *args, **kwargs):
    path = Path(directory).parent / 'generation_sample_metrics.csv'
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'sim'])
        writer.writeheader()
        writer.writerows({'sample_id': meta['sample_id'], 'sim': .5} for _, _, meta in pairs)
    return {'sample_metrics_csv': str(path), 'avg_sim': .5}


@pytest.mark.parametrize('message', ['CUDA error: device-side assert triggered',
                                   'CUDA error: an illegal memory access was encountered'])
def test_fatal_cuda_error_stops_retries_and_preserves_pending_samples(setup_run, message):
    conf, _, run, root = setup_run
    conf.vc.retries = 2
    conf.vc.generate_only = False
    class Fatal(SmokeAdversary):
        calls = 0
        def attack(self, **kwargs):
            self.calls += 1
            raise RuntimeError(message)
    adapter = Fatal(conf, conf.dataset, 'cpu', logging.getLogger())
    with patch('src.benchmark.backends.select_adversary', return_value=adapter), \
            patch('src.evaluation.pipeline.evaluate_run') as scorer:
        with pytest.raises(RuntimeError, match='CUDA error'):
            run('fatal')
    manifest = json.loads((root / 'fatal/run_manifest.json').read_text())
    assert adapter.calls == 1
    scorer.assert_not_called()
    assert manifest['status'] == 'failed'
    assert manifest['samples'][0]['fatal_runtime_error']
    assert manifest['samples'][0]['attempts'] == 1
    assert manifest['samples'][1]['status'] == 'pending'
    assert manifest['coverage']['generation_failed'] == 1
    assert manifest['coverage']['pending'] == 1


def test_missing_dataset_has_actionable_error(tmp_path):
    conf = OmegaConf.create({'dataset': {'root_path': str(tmp_path), 'use_hf_dataset': False}})
    with pytest.raises(ValueError, match='manifest_filename'):
        ZeroShotDataset(conf, conf.dataset, logging.getLogger())


def test_eval_separate_and_report_gate(setup_run):
    conf, _, run, _ = setup_run
    first, _, _ = run('first')
    conf.vc.generate_only = False
    conf.vc.evaluate_only = True
    conf.vc.evaluation = {'generated_audio_dir': str(first / 'generated_audio')}
    with mocked_evaluator(fake_evaluate):
        out, manifest, _ = run('evaluated')
    assert manifest['status'] == 'complete'
    assert validate_report(manifest)['requested'] == 2
    manifest['samples'][0]['metrics']['sim'] = 'nan'
    with pytest.raises(ValueError, match='Coverage'):
        validate_report(manifest)


def test_metric_coverage_is_per_requested_population():
    rows = [{'generated_sha256': 'x', 'metrics': {'sim': .5}},
            {'generated_sha256': 'y', 'metrics': {'sim': None}}, {'metrics': {}}]
    result = coverage(rows, ['sim'], evaluated=True)
    assert result['requested'] == 3 and result['generated'] == 2
    assert result['metric_valid']['sim'] == 1
    assert not result['eligible_for_comparison']


def test_no_training_or_eval_imports():
    subprocess.run([sys.executable, '-c', '''
import sys
from src.datasets.zero_shot import ZeroShotDataset
from src.adversary.qwen3_tts_ots import Qwen3TTSZeroShotAdversary
assert 'src.models.text' not in sys.modules
assert 'src.datasets.data_utils' not in sys.modules
assert 'src.evaluation.generation' not in sys.modules
assert 'whisper' not in sys.modules
'''], check=True)


def test_public_quickstart_help():
    for name in ('run_qwen3tts_quickstart', 'run_fishspeech_quickstart', 'run_fishspeech_s2_quickstart', 'run_protect_qwen3tts_quickstart'):
        subprocess.run([sys.executable, f'scripts/{name}.py', '--help'], check=True, capture_output=True)


def test_hydra_entrypoint_writes_status(setup_run):
    conf, _, _, root = setup_run
    proc = subprocess.run([sys.executable, 'run_vc.py', '--config-name', 'ots_vc/clean/libritts/qwen3_tts_ots',
        'vc.model=smoke', '+vc.generate_only=true', 'device=cpu',
        f'base_dir={root}', f'dataset.root_path={conf.dataset.root_path}',
        'dataset.use_hf_dataset=false', 'dataset.speaker_id=one', 'dataset.manifest_variant=null',
        '+dataset.manifest_filename=metadata.json', 'adversary.max_samples=1',
        f'hydra.run.dir={root}/hydra'], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    manifests = list(root.glob('results/*/*/run_manifest.json'))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest['coverage']['requested'] == 1
    assert manifest['status'] == 'generated'
    assert (manifests[0].parent / 'metrics.json').is_file()


def test_interruption_retains_progress(setup_run):
    conf, _, run, root = setup_run
    class Interrupted(SmokeAdversary):
        def attack(self, **kwargs):
            raise KeyboardInterrupt()
    with patch('src.benchmark.backends.select_adversary', return_value=Interrupted(conf, conf.dataset, 'cpu', logging.getLogger())):
        with pytest.raises(KeyboardInterrupt):
            run('interrupted')
    manifest = json.loads((root / 'interrupted/run_manifest.json').read_text())
    assert manifest['status'] == 'interrupted'
    assert not manifest['coverage']['eligible_for_comparison']


def test_eval_failure_persists_manifest(setup_run):
    conf, _, run, root = setup_run
    conf.vc.generate_only = False
    def fail(*args, **kwargs):
        raise RuntimeError('evaluator unavailable')
    with mocked_evaluator(fail):
        with pytest.raises(RuntimeError, match='evaluator unavailable'):
            run('eval-failed')
    manifest = json.loads((root / 'eval-failed/run_manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['coverage']['generated'] == 2


def test_model_catalog_and_public_links():
    import re
    catalog = json.loads(Path('src/benchmark/model_catalog.json').read_text())
    assert len({r['key'] for r in catalog}) == len(catalog)
    assert sum(r['status'] == 'historical_results' for r in catalog) == 18
    for name in set(re.findall(r'scripts/[\w.-]+\.py', Path('README.md').read_text())):
        assert Path(name).is_file(), name


def test_report_export_and_site_gate(setup_run, monkeypatch):
    conf, _, run, root = setup_run
    conf.vc.generate_only = False
    with mocked_evaluator(fake_evaluate):
        out, manifest, _ = run('scored-fixture')
    # A simulated real adapter label tests the publication path without model downloads.
    manifest['config']['vc']['model'] = 'test-adapter'
    (out / 'run_manifest.json').write_text(json.dumps(manifest))
    reports = root / 'reports'
    result = reports / 'fixture.json'
    from src.benchmark.cli import main
    monkeypatch.setattr(sys, 'argv', ['rvcbench', 'report', str(out), '--output', str(result)])
    main()
    payload = json.loads(result.read_text())
    assert payload['means'] == {'sim': .5}
    monkeypatch.syspath_prepend(str(Path('docs/site-src').resolve()))
    import render
    assert 'test-adapter' in render.render_validated_runs(reports)
    payload['means']['sim'] = .9
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='disagree'):
        render.render_validated_runs(reports)


def test_journal_recovers_progress_and_ignores_torn_tail(setup_run):
    from src.benchmark.artifacts import load_run
    conf, _, run, _ = setup_run
    out, manifest, _ = run('journal')
    expected = manifest['samples']
    manifest['status'] = 'running'
    for row in manifest['samples']:
        row.pop('generated_sha256', None)
        row['status'] = 'pending'
    (out / 'run_manifest.json').write_text(json.dumps(manifest))
    with (out / 'sample_events.jsonl').open('a') as f:
        f.write('{"sample_id":')
    recovered = load_run(out)
    assert recovered['coverage']['generated'] == 2
    assert all(r['status'] == 'generated' for r in recovered['samples'])
    assert recovered['status'] == 'running'  # progress is not process liveness/completion
    conf.vc.resume_from = str(out)
    with patch('src.benchmark.backends.select_adversary', side_effect=AssertionError('must replay journal')):
        _, resumed, _ = run('journal-resumed')
    assert resumed['coverage']['generated'] == 2
