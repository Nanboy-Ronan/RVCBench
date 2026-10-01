"""Sample-level zero-shot execution with immutable resume sources and coverage gates."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

from omegaconf import OmegaConf
from .artifacts import (SCHEMA_VERSION, METRIC_COLUMNS, atomic_json, append_sample_event, load_run, coverage, digest,
                        file_hash, input_fingerprint, input_records, output_path, provenance)


class SampleView:
    def __init__(self, dataset, sample):
        self.dataset, self.sample = dataset, sample

    def get_zero_shot_samples(self, **kwargs):
        return [self.sample]

    def iter_zero_shot_samples(self, **kwargs):
        return iter([self.sample])

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class SampleErrors(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def check_audio(path):
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(str(path))
    if rate <= 0 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('Empty or non-finite generated audio')


def run_zero_shot(conf, base_dir, device, dataset, exp_dir, logger, protected_audio_dir=None):
    from src.utils.seeding import configure_seeds
    from src.workflows.vc import _select_adversary

    options = conf.vc
    evaluation = options.get('evaluation') or {}
    evaluate_only = bool(options.get('evaluate_only', False))
    generate_only = bool(options.get('generate_only', False))
    if evaluate_only and generate_only:
        raise ValueError('evaluate_only and generate_only are mutually exclusive')
    required = list(OmegaConf.select(conf, 'evaluation.required_metrics', default=['mcd', 'wer', 'sim']))
    if not required or set(required) - METRIC_COLUMNS.keys():
        raise ValueError(f'required_metrics must be a nonempty subset of {list(METRIC_COLUMNS)}')
    limit = evaluation.get('max_samples', conf.adversary.get('max_samples'))
    samples = dataset.get_zero_shot_samples(max_samples=limit)
    if not samples:
        raise ValueError('Empty evaluation selection')
    rows = input_records(samples)
    audio_dir = Path(exp_dir) / 'generated_audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    effective_config = OmegaConf.to_container(conf, resolve=True)
    seed = conf.get('seed')
    if seed is None:
        seed = conf.adversary.get('seed')
    seed = int(seed if seed is not None else 42)
    generation_config = {'model': options.get('model'), 'adversary': effective_config['adversary'],
                         'seed': seed, 'sample_seed_policy': 'seed_plus_original_index_v1'}
    manifest = {'schema_version': SCHEMA_VERSION, 'protocol': 'rvcbench-zero-shot-v1',
                'status': 'running', 'evaluated': False, 'config': effective_config,
                'provenance': provenance(Path(__file__).resolve().parents[2]), 'generation_config': generation_config,
                'generation_fingerprint': digest(generation_config),
                'input_fingerprint': input_fingerprint(rows), 'samples': rows,
                'coverage': coverage(rows, required, evaluated=False)}
    path = Path(exp_dir) / 'run_manifest.json'
    events = Path(exp_dir) / 'sample_events.jsonl'
    if path.exists() or events.exists():
        raise ValueError('Run directory already contains artifacts; use a new directory or resume_from')
    atomic_json(path, manifest)
    metrics = {}
    handler = SampleErrors()
    logger.addHandler(handler)
    started = time.perf_counter()
    try:
        model_ref = conf.adversary.get('checkpoint_path')
        resolved_conf = OmegaConf.create(effective_config)
        if not evaluate_only and options.get('model') in ('qwen3_tts', 'qwentts') and model_ref and not Path(str(model_ref)).exists():
            from huggingface_hub import HfApi
            revision = HfApi().model_info(str(model_ref), revision=conf.adversary.get('revision')).sha
            OmegaConf.update(resolved_conf, 'adversary.revision', revision, force_add=True)
            generation_config['resolved_model_revision'] = revision
            manifest['model_reference'] = {'repo_id': str(model_ref), 'revision': revision}
        elif model_ref and Path(str(model_ref)).is_dir():
            weights = {str(p.relative_to(model_ref)): file_hash(p) for p in sorted(Path(str(model_ref)).rglob('*'))
                       if p.is_file() and p.suffix in ('.safetensors', '.bin', '.pt', '.pth', '.json')}
            generation_config['checkpoint_sha256'] = digest(weights)
            manifest['model_reference'] = {'path': str(model_ref), 'files': weights}
        else:
            manifest['model_reference'] = {'configured_adversary': effective_config['adversary'],
                                          'note': 'Adapter-specific model assets; no automatic remote revision resolution.'}
        if options.get('model') in ('qwen3_tts', 'qwentts') and resolved_conf.adversary.get('seed') is None:
            OmegaConf.update(resolved_conf, 'adversary.seed', seed, force_add=True)
        manifest['generation_fingerprint'] = digest(generation_config)
        resume = options.get('resume_from')
        source_dir = evaluation.get('generated_audio_dir') if evaluate_only else None
        if evaluate_only and not source_dir:
            raise ValueError('evaluate_only requires vc.evaluation.generated_audio_dir')
        if resume and evaluate_only:
            raise ValueError('Use resume_from or evaluate_only, not both')
        source = Path(str(resume)).resolve() if resume else (Path(str(source_dir)).resolve().parent if source_dir else None)
        previous = {}
        if source:
            old = load_run(source)
            if old.get('schema_version') != SCHEMA_VERSION or old.get('input_fingerprint') != manifest['input_fingerprint']:
                raise ValueError('Resume/evaluation input manifest differs from the source run')
            if evaluate_only and old['config']['vc']['model'] != options.get('model'):
                raise ValueError('Evaluation model label differs from source generation run')
            if not evaluate_only and old.get('generation_fingerprint') != manifest['generation_fingerprint']:
                raise ValueError('Resume generation settings differ from the source run')
            if not evaluate_only and old['provenance'].get('source_sha256') != manifest['provenance']['source_sha256']:
                raise ValueError('Resume runtime source changed; start a new run')
            if not evaluate_only and old['provenance'].get('packages') != manifest['provenance']['packages']:
                raise ValueError('Resume Python environment changed; start a new run')
            previous = {r['sample_id']: r for r in old['samples']}
            manifest['source_run'] = str(source)
            if evaluate_only:
                manifest['generation_fingerprint'] = old['generation_fingerprint']
                manifest['source_generation_provenance'] = old.get('source_generation_provenance', old['provenance'])
                manifest['generation_config'] = old.get('generation_config')
                manifest['model_reference'] = old.get('model_reference')
        adversary = None
        retries = int(options.get('retries', 0))
        if retries < 0:
            raise ValueError('vc.retries must be >= 0')
        atomic_json(path, manifest)
        for i, (sample, row) in enumerate(zip(samples, rows)):
            dest = output_path(audio_dir, sample)
            row['generated_path'] = str(dest.resolve())
            row['attempts'] = 0
            row['metrics'] = {}
            prior = previous.get(row['sample_id'], {})
            if prior.get('generated_sha256'):
                src = Path(prior['generated_path'])
                if not src.is_file() or file_hash(src) != prior['generated_sha256']:
                    raise ValueError(f'Resume artifact changed or missing: {src}')
                check_audio(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                row.update(status='generated', generated_sha256=prior['generated_sha256'],
                           synthesis_time_sec=prior.get('synthesis_time_sec'), reused=True)
            elif evaluate_only:
                row.update(status='generation_failed', error=prior.get('error') or 'Missing generated artifact in source run')
            elif not row['prompt_sha256'] or not row['target_sha256'] or not str(row['target_text']).strip():
                row.update(status='input_failed', error='Missing prompt audio, target audio, or target transcript')
            else:
                if adversary is None:
                    adversary = _select_adversary(resolved_conf, conf.dataset, device, logger)
                for attempt in range(retries + 1):
                    row['attempts'] += 1
                    handler.messages.clear()
                    row.update(status='generating', error=None)
                    append_sample_event(events, row)
                    configure_seeds(seed + sample.index, logger=None)
                    tick = time.perf_counter()
                    try:
                        adversary.attack(output_path=str(audio_dir), dataset=SampleView(dataset, sample),
                                         protected_audio_path=str(protected_audio_dir) if protected_audio_dir else None)
                        if not dest.is_file():
                            raise RuntimeError('Adapter produced no audio for this sample')
                        check_audio(dest)
                        elapsed = time.perf_counter() - tick
                        # Prefer the adapter's synthesis-only timing when available.
                        for timing in getattr(adversary, '_synthesis_timing_records', []):
                            if Path(timing['generated_path']).resolve() == dest.resolve():
                                elapsed = timing['synthesis_time_sec']
                        row.update(status='generated', generated_sha256=file_hash(dest), synthesis_time_sec=elapsed)
                        if handler.messages:
                            row['warnings'] = list(handler.messages)
                        break
                    except Exception as exc:
                        dest.unlink(missing_ok=True)
                        row.update(status='generation_failed', error=f'{type(exc).__name__}: {exc}',
                                   warnings=list(handler.messages))
                        logger.warning('Sample %s attempt %d failed: %s', row['sample_id'], attempt + 1, exc)
            manifest['coverage'] = coverage(rows, required, evaluated=False)
            append_sample_event(events, row)
            logger.info('[%d/%d] %s; generated=%d failed=%d', i + 1, len(rows), row['status'],
                        manifest['coverage']['generated'], sum(r['status'].endswith('failed') for r in rows))
        with (audio_dir / 'synthesis_timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['generated_path', 'synthesis_time_sec'])
            writer.writeheader()
            writer.writerows({k: r.get(k) for k in writer.fieldnames} for r in rows if r.get('generated_sha256'))
        pairs = [(sample.target_path, Path(row['generated_path']),
                  {'sample_id': row['sample_id'], 'speaker_id': sample.speaker_id,
                   'text': sample.target_text, 'language': sample.target_language or sample.prompt_language})
                 for sample, row in zip(samples, rows) if row.get('generated_sha256')]
        if not generate_only and pairs:
            configure_seeds(seed, logger=None)
            # Release generation weights before loading the evaluation stack.
            del adversary
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from src.evaluation import generation
            metrics = generation.evaluate_pairs(pairs, str(audio_dir), device, logger,
                synthesis_time_sec=sum(r.get('synthesis_time_sec') or 0 for r in rows),
                bootstrap_config=OmegaConf.select(conf, 'evaluation.bootstrap'),
                max_generated_audio_seconds=evaluation.get('generated_audio_max_seconds',
                    OmegaConf.select(conf, 'evaluation.generated_audio_max_seconds')))
            csv_path = metrics.get('sample_metrics_csv')
            if csv_path:
                with Path(csv_path).open() as f:
                    scores = {r['sample_id']: r for r in csv.DictReader(f)}
                for row in rows:
                    if row.get('generated_sha256'):
                        row['metrics'] = scores.get(row['sample_id'], {})
                        from .artifacts import finite
                        missing = [m for m in required if not finite(row['metrics'].get(METRIC_COLUMNS[m]))]
                        row.update(status='metric_failed' if missing else 'complete',
                                   error='Missing/invalid metrics: ' + ', '.join(missing) if missing else None)
            manifest['evaluated'] = True
        manifest['evaluation_fingerprint'] = digest({
            'source': manifest['provenance']['source_sha256'],
            'config': effective_config.get('evaluation', {}),
            'vc_evaluation': {k: v for k, v in (effective_config['vc'].get('evaluation') or {}).items() if k != 'generated_audio_dir'},
            'seed': seed, 'packages': manifest['provenance']['packages']})
        manifest['coverage'] = coverage(rows, required, evaluated=manifest['evaluated'])
        manifest['status'] = manifest['coverage']['status']
        metrics['coverage'] = manifest['coverage']
        metrics['run_manifest'] = str(path)
        manifest['metrics'] = metrics
        logger.info('Run status=%s coverage=%s', manifest['status'], manifest['coverage'])
        return metrics, protected_audio_dir, audio_dir
    except BaseException as exc:
        manifest['status'] = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        logger.removeHandler(handler)
        manifest['elapsed_sec'] = time.perf_counter() - started
        manifest['coverage'] = coverage(rows, required, evaluated=manifest['evaluated'])
        atomic_json(path, manifest)
