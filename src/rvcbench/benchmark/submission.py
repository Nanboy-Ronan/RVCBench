"""Score audio generated outside RVCBench against a fixed, versioned suite.

``export_prompts`` writes the reference audio, transcripts and expected output file
names of a suite. Anyone can then generate speech with their own code, and
``score_submission`` scores the files with the suite's fixed evaluation protocol.
Target recordings are never exported; they are only read while scoring.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from .artifacts import (SCHEMA_VERSION, atomic_json, append_sample_event, coverage, digest, file_hash,
                        input_fingerprint, input_records, metric_means, provenance, runtime_root)

PROTOCOL = 'rvcbench-submission-v1'
SUITES_DIR = Path(__file__).resolve().parents[1] / 'suites'
CONFIGS_DIR = Path(__file__).resolve().parents[1] / 'configs'
_SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')


def available_suites():
    return sorted(path.parent.name.replace('_', '-') for path in SUITES_DIR.glob('*/suite.json'))


def load_suite(name_or_path):
    """Load a packaged suite by name (``onboarding-v1``) or a ``suite.json`` path."""
    path = Path(str(name_or_path))
    if path.suffix != '.json':
        path = SUITES_DIR / str(name_or_path).replace('-', '_') / 'suite.json'
        if not path.is_file():
            raise ValueError(f"Unknown suite '{name_or_path}'. Available: {', '.join(available_suites())}")
    raw = path.read_bytes()
    spec = json.loads(raw)
    for key in ('suite', 'version', 'leaderboard', 'label', 'evaluation', 'tasks'):
        if key not in spec:
            raise ValueError(f'Suite {path} is missing {key!r}')
    tasks = [task['task'] for task in spec['tasks']]
    if not tasks or len(set(tasks)) != len(tasks) or not all(_SAFE_NAME.fullmatch(t) for t in tasks):
        raise ValueError(f'Suite {path} needs unique, file-name-safe task names')
    spec['_directory'] = path.parent
    spec['_sha256'] = file_hash(path)
    return spec


def suite_record(spec):
    return {'suite': spec['suite'], 'version': spec['version'], 'leaderboard': bool(spec['leaderboard']),
            'label': spec['label'], 'suite_sha256': spec['_sha256']}


def _frozen_selection(spec, task):
    selection = json.loads((spec['_directory'] / task['selection']).read_text())
    return selection, {sample['sample_id']: sample for sample in selection['samples']}


def _fetch_from_hub(spec, task, manifest_rows):
    from huggingface_hub import snapshot_download
    config = task['hf_config_name']
    names = sorted({row[key] for row in manifest_rows for key in ('prompt_file_name', 'target_file_name')})
    local = snapshot_download(repo_id=spec['hf_dataset_id'], repo_type='dataset', revision=spec['hf_revision'],
                              allow_patterns=[f'{config}/{name}' for name in names])
    return Path(local) / config


def _task_samples(spec, task, data_root, logger):
    """Load one task's frozen samples and verify every input against the frozen hashes."""
    from rvcbench.datasets.zero_shot import ZeroShotDataset
    manifest = (spec['_directory'] / task['manifest']).resolve()
    rows = json.loads(manifest.read_text())
    pair_ids = [str(row.get('pair_id') or '') for row in rows]
    if len(set(pair_ids)) != len(pair_ids) or not all(_SAFE_NAME.fullmatch(p) for p in pair_ids):
        raise ValueError(f"Task {task['task']}: pair_id values must be unique and file-name safe")
    root = (Path(data_root) / task['hf_config_name']) if data_root else _fetch_from_hub(spec, task, rows)
    dataset_config = OmegaConf.load(CONFIGS_DIR / 'dataset' / f"{task['dataset_config']}.yaml")
    dataset_config.root_path = str(root)
    dataset_config.use_hf_dataset = False
    dataset_config.manifest_filename = str(manifest)
    dataset_config.speaker_id = None
    dataset = ZeroShotDataset(OmegaConf.create({}), dataset_config, logger)
    samples = dataset.get_zero_shot_samples()
    records = input_records(samples)
    selection, frozen = _frozen_selection(spec, task)
    mismatched = [r['pair_id'] for r in records
                  if r['sample_id'] not in frozen
                  or r['prompt_sha256'] != frozen[r['sample_id']]['prompt_sha256']
                  or r['target_sha256'] != frozen[r['sample_id']]['target_sha256']]
    if mismatched or input_fingerprint(records) != selection['input_fingerprint']:
        raise ValueError(f"Task {task['task']}: inputs under {root} differ from the frozen suite "
                         f"(mismatched pairs: {', '.join(mismatched) or 'transcripts or metadata'})")
    return dataset, samples, records


def export_prompts(suite, output, *, data_root=None, logger=None):
    """Write reference audio, transcripts and expected output names; never target audio."""
    logger = logger or logging.getLogger('rvcbench.prompts')
    spec = load_suite(suite)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; choose a new directory')
    entries = []
    for task in spec['tasks']:
        _, samples, records = _task_samples(spec, task, data_root, logger)
        for sample, record in zip(samples, records):
            # Hub files are symlinks into a content-addressed cache; take the suffix from the manifest.
            suffix = Path(str(sample.extra.get('prompt_file_name') or '')).suffix or '.wav'
            reference = Path('references') / task['task'] / f"{record['pair_id']}{suffix}"
            (output / reference).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sample.prompt_path, output / reference)
            entries.append({'task': task['task'], 'pair_id': record['pair_id'], 'speaker_id': record['speaker_id'],
                            'reference_audio': reference.as_posix(), 'reference_sha256': record['prompt_sha256'],
                            'reference_text': record['prompt_text'], 'text': record['target_text'],
                            'language': record['target_language'],
                            'output_file': f"{task['task']}/{record['pair_id']}.wav"})
    with (output / 'prompts.jsonl').open('w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    atomic_json(output / 'suite.json', {**suite_record(spec), 'tasks': [t['task'] for t in spec['tasks']],
                                        'prompts': len(entries)})
    (output / 'README.md').write_text(_PROMPTS_README.format(
        suite=spec['suite'], label=spec['label'], count=len(entries)), encoding='utf-8')
    return {'suite': spec['suite'], 'prompts': len(entries), 'output': str(output.resolve())}


_PROMPTS_README = """# {suite} prompts

{label}

`prompts.jsonl` lists {count} utterances. For each line:

1. Synthesize `text` in the voice of `reference_audio` (its transcript is `reference_text`).
2. Save the result as a mono WAV file at `<your output directory>/<output_file>`.

Then score the directory:

    rvcbench score --suite {suite} --generated <your output directory> --model <model name> --output <results directory>

A missing or invalid file counts as a failed sample. Do not use the target recordings of
the dataset; a file identical to a target recording is rejected.
"""


def _score_task(spec, task, generated, output, label, device, data_root, logger):
    evaluation = spec['evaluation']
    dataset, samples, rows = _task_samples(spec, task, data_root, logger)
    task_dir = output / task['task']
    audio_dir = task_dir / 'generated_audio'
    audio_dir.mkdir(parents=True)
    events = task_dir / 'sample_events.jsonl'
    for sample, row in zip(samples, rows):
        submitted = Path(generated) / task['task'] / f"{row['pair_id']}.wav"
        row.update(metrics={}, attempts=0, submitted_path=str(submitted))
        if not submitted.is_file():
            row.update(status='generation_failed', error=f"Missing submitted audio: {task['task']}/{row['pair_id']}.wav")
        else:
            from .runner import check_audio
            try:
                check_audio(submitted)
            except Exception as exc:
                row.update(status='generation_failed', error=f'Invalid submitted audio: {type(exc).__name__}: {exc}')
            else:
                sha = file_hash(submitted)
                if sha == row['target_sha256']:
                    row.update(status='generation_failed', error='Submitted audio is the target recording')
                else:
                    stored = audio_dir / submitted.name
                    shutil.copyfile(submitted, stored)
                    row.update(status='generated', error=None, generated_path=str(stored.resolve()),
                               generated_sha256=sha)
        append_sample_event(events, row)
    required = list(evaluation['required_metrics'])
    generation_config = {'model': label, 'adversary': {}, 'sample_seed_policy': 'external_submission_v1'}
    manifest = {'schema_version': SCHEMA_VERSION, 'protocol': PROTOCOL, 'status': 'running', 'evaluated': False,
                'suite': {**suite_record(spec), 'task': task['task']},
                'config': {'vc': {'mode': 'ots', 'model': label}, 'dataset': OmegaConf.to_container(dataset.dataset_config),
                           'evaluation': evaluation},
                'provenance': provenance(runtime_root()), 'generation_config': generation_config,
                'generation_provenance': None, 'generation_fingerprint': digest(generation_config),
                'input_fingerprint': input_fingerprint(rows), 'samples': rows,
                'variant_selection': dataset.variant_selection, 'reference_stage': None,
                'comparison_status': 'requires_pairwise_protocol_check',
                'coverage': coverage(rows, required, evaluated=False)}
    manifest_path = task_dir / 'run_manifest.json'
    atomic_json(manifest_path, manifest)
    metrics = {}
    try:
        if any(row.get('generated_sha256') for row in rows):
            from rvcbench.evaluation.pipeline import evaluate_run
            manifest['evaluation_device'] = str(device)
            metrics = evaluate_run(rows, required, task_dir, device, logger, seed=int(evaluation.get('seed', 42)),
                                   cap=evaluation.get('generated_audio_max_seconds'),
                                   bootstrap_config=evaluation.get('bootstrap'),
                                   wer_normalization=evaluation.get('wer_normalization', 'ascii_punctuation_removed_v2'),
                                   on_sample=lambda row: append_sample_event(events, row))
            manifest['evaluated'] = True
            manifest['evaluation_fingerprint'] = metrics['evaluation_fingerprint']
        manifest['coverage'] = coverage(rows, required, evaluated=manifest['evaluated'])
        manifest['status'] = manifest['coverage']['status']
        manifest['metrics'] = metrics
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        atomic_json(manifest_path, manifest)
    result = {'status': manifest['status'], 'coverage': manifest['coverage'],
              'run_manifest': f"{task['task']}/run_manifest.json", 'run_manifest_sha256': file_hash(manifest_path),
              'evaluation_fingerprint': manifest.get('evaluation_fingerprint'),
              'failures': {row['pair_id']: row['error'] for row in rows if row['status'] == 'generation_failed'},
              'generated_sha256': {row['pair_id']: row.get('generated_sha256') for row in rows}}
    if manifest['status'] == 'complete':
        result['means'] = metric_means(rows, required)
    return result


def score_submission(suite, generated, output, *, model, device='cpu', data_root=None, logger=None):
    """Score externally generated audio; writes one run per task and ``submission.json``."""
    import rvcbench
    logger = logger or logging.getLogger('rvcbench.score')
    spec = load_suite(suite)
    if not str(model).strip():
        raise ValueError('A model name is required')
    if not Path(generated).is_dir():
        raise ValueError(f'Generated audio directory not found: {generated}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = {task['task']: _score_task(spec, task, Path(generated), output, str(model), device, data_root, logger)
             for task in spec['tasks']}
    complete = all(task['status'] == 'complete' for task in tasks.values())
    submission = {'schema_version': 1, 'protocol': PROTOCOL, **suite_record(spec), 'model': str(model),
                  'rvcbench_version': rvcbench.__version__, 'status': 'complete' if complete else 'partial',
                  'tasks': tasks}
    atomic_json(output / 'submission.json', submission)
    return submission
