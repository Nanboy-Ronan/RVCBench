"""Sample-level zero-shot execution with immutable resume sources and coverage gates."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path
from numbers import Integral

from omegaconf import OmegaConf
from rvcbench.utils.runtime_errors import invalid_cuda_context
from .artifacts import (SCHEMA_VERSION, METRIC_COLUMNS, atomic_json, append_sample_event, load_run, coverage, digest,
                        file_hash, input_fingerprint, input_records, output_path, provenance)


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
    from rvcbench.utils.seeding import configure_seeds
    from .backends import create_backend, GenerationRequest

    seed = conf.get('seed')
    if seed is None:
        seed = conf.adversary.get('seed')
    seed = 42 if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError('Run seed must be an integer; booleans, floats and strings are not accepted')
    seed = int(seed)
    if not -(2**63) <= seed < 2**64:
        raise ValueError('Run seed is outside the supported Torch range')
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
    reference_stage = None
    if (options.get('reference_audio_dir') and protected_audio_dir and
            Path(str(options.reference_audio_dir)).expanduser().resolve() !=
            Path(protected_audio_dir).expanduser().resolve()):
        raise ValueError('Conflicting reference_audio_dir and protected_audio_dir')
    reference_root = options.get('reference_audio_dir') or protected_audio_dir
    if reference_root:
        from .reference_stages import bind_reference_stage
        samples, reference_stage = bind_reference_stage(samples, dataset._dataset_root,
            reference_root, options.get('reference_stage', 'external_reference'))
    rows = input_records(samples)
    if reference_stage:
        for row, binding in zip(rows, reference_stage['bindings']):
            row.update({k: binding[k] for k in ('clean_prompt_path', 'clean_prompt_sha256')})
    audio_dir = Path(exp_dir) / 'generated_audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    effective_config = OmegaConf.to_container(conf, resolve=True)
    generation_config = {'model': options.get('model'), 'adversary': effective_config['adversary'],
                         'seed': seed, 'sample_seed_policy': 'seed_plus_source_index_v2', 'device': str(device)}
    if reference_stage:
        generation_config['reference_stage_fingerprint'] = reference_stage['fingerprint']
    runtime_provenance = provenance(Path(__file__).resolve().parents[3])
    from .fingerprints import generation_runtime
    generation_provenance = (None if evaluate_only else
        generation_runtime(Path(__file__).resolve().parents[3], conf, runtime_provenance['packages']))
    manifest = {'schema_version': SCHEMA_VERSION, 'protocol': 'rvcbench-zero-shot-v2',
                'status': 'running', 'evaluated': False, 'config': effective_config,
                'provenance': runtime_provenance, 'generation_config': generation_config,
                'generation_provenance': generation_provenance,
                'generation_fingerprint': digest(generation_config),
                'input_fingerprint': input_fingerprint(rows), 'samples': rows,
                'variant_selection': getattr(dataset, 'variant_selection', None),
                'reference_stage': reference_stage,
                'comparison_status': 'requires_pairwise_protocol_check',
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
    backend = None
    request_attempts = 0
    manifest['request_timing_profile'] = None
    try:
        resolved_conf = OmegaConf.create(effective_config)
        if not evaluate_only:
            from .model_assets import resolve_model_assets
            resolved_conf, reference, asset_fingerprint = resolve_model_assets(conf, logger=logger)
            manifest['model_reference'] = reference
            generation_config['model_assets_fingerprint'] = asset_fingerprint
        # One effective run seed; adapters using internal RNGs must agree with it.
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
            if ((old.get('reference_stage') or {}).get('fingerprint') !=
                    (reference_stage or {}).get('fingerprint')):
                raise ValueError('Resume/evaluation reference stage lineage differs from the source run')
            if evaluate_only and old['config']['vc']['model'] != options.get('model'):
                raise ValueError('Evaluation model label differs from source generation run')
            if not evaluate_only and old.get('generation_fingerprint') != manifest['generation_fingerprint']:
                raise ValueError('Resume generation settings differ from the source run')
            if not evaluate_only and old.get('generation_provenance', old['provenance']).get('source_sha256') != generation_provenance['source_sha256']:
                raise ValueError('Resume runtime source changed; start a new run')
            if not evaluate_only and old.get('generation_provenance', old['provenance']).get('packages') != generation_provenance['packages']:
                raise ValueError('Resume Python environment changed; start a new run')
            if not evaluate_only and old.get('generation_provenance', {}).get('worker_environment') != generation_provenance.get('worker_environment'):
                raise ValueError('Resume worker Python environment changed; start a new run')
            previous = {r['sample_id']: r for r in old['samples']}
            manifest['source_run'] = str(source)
            if evaluate_only:
                manifest['generation_fingerprint'] = old['generation_fingerprint']
                manifest['source_generation_provenance'] = old.get('source_generation_provenance', old.get('generation_provenance', old['provenance']))
                manifest['generation_provenance'] = old.get('generation_provenance')
                manifest['generation_config'] = old.get('generation_config')
                manifest['model_reference'] = old.get('model_reference')
        generated_count = failed_count = 0
        retries = int(options.get('retries', 0))
        if retries < 0:
            raise ValueError('vc.retries must be >= 0')
        atomic_json(path, manifest)
        for i, (sample, row) in enumerate(zip(samples, rows)):
            dest = output_path(audio_dir, sample)
            row['generated_path'] = str(dest.resolve())
            row['attempts'] = 0
            row['metrics'] = {}
            row['seed'] = seed + sample.index
            prior = previous.get(row['sample_id'], {})
            if prior.get('generated_sha256'):
                src = Path(prior['generated_path'])
                if not src.is_file() or file_hash(src) != prior['generated_sha256']:
                    raise ValueError(f'Resume artifact changed or missing: {src}')
                check_audio(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                row.update(status='generated', generated_sha256=prior['generated_sha256'],
                           synthesis_time_sec=prior.get('synthesis_time_sec'),
                           timing_scope=prior.get('timing_scope'),
                           adapter_call_time_sec=prior.get('adapter_call_time_sec'), reused=True)
                for key in ('native_seed', 'native_seed_policy', 'native_requested_seed', 'conditioning_variant', 'native_initialization',
                            'request_wall_time_sec', 'request_timing_phase',
                            'request_timing_profile_fingerprint', 'request_timing_profile_after_fingerprint'):
                    if key in prior:
                        row[key] = prior[key]
            elif evaluate_only:
                row.update(status='generation_failed', error=prior.get('error') or 'Missing generated artifact in source run')
            elif not row['prompt_sha256'] or not row['target_sha256'] or not str(row['target_text']).strip():
                row.update(status='input_failed', error='Missing prompt audio, target audio, or target transcript')
            else:
                if backend is None:
                    backend = create_backend(resolved_conf, dataset, device, logger)
                    prepare_started = time.perf_counter()
                    backend.prepare()
                    from .timing import synchronize
                    synchronize(device)
                    manifest['backend_prepare_wall_time_sec'] = time.perf_counter() - prepare_started
                for attempt in range(retries + 1):
                    row['attempts'] += 1
                    handler.messages.clear()
                    row.update(status='generating', error=None)
                    append_sample_event(events, row)
                    try:
                        from .timing import timing_profile, synchronize
                        configure_seeds(row['seed'], logger=None)
                        profile = timing_profile(device, resolved_conf)
                        if manifest['request_timing_profile'] is None:
                            manifest['request_timing_profile'] = profile
                        synchronize(device)
                        request_started = time.perf_counter()
                        phase = 'first_request_after_prepare' if request_attempts == 0 else 'subsequent_request'
                        request_attempts += 1
                        result, = backend.generate_batch([GenerationRequest(
                            sample=sample, seed=row['seed'], output_dir=audio_dir,
                            protected_audio_dir=protected_audio_dir)])
                        synchronize(device)
                        request_elapsed = time.perf_counter() - request_started
                        after_profile = timing_profile(device, resolved_conf)
                        row.update(request_wall_time_sec=request_elapsed, request_timing_phase=phase,
                                   request_timing_profile_fingerprint=profile['fingerprint'],
                                   request_timing_profile_after_fingerprint=after_profile['fingerprint'])
                        if after_profile != profile:
                            row['request_timing_changed_settings'] = after_profile
                        if result.sample_id != row['sample_id'] or result.path.resolve() != dest.resolve():
                            raise RuntimeError('Backend returned a result for the wrong request')
                        if not dest.is_file():
                            raise RuntimeError('Adapter produced no audio for this sample')
                        check_audio(dest)
                        row.update(status='generated', generated_sha256=file_hash(dest),
                                   synthesis_time_sec=result.elapsed_sec, timing_scope=result.timing_scope,
                                   adapter_call_time_sec=result.adapter_call_time_sec)
                        if result.native_initialization is not None:
                            row["native_initialization"] = result.native_initialization
                        if result.conditioning_variant is not None:
                            row["conditioning_variant"] = result.conditioning_variant
                        if result.native_seed_policy is not None:
                            row.update(native_seed=result.native_seed, native_seed_policy=result.native_seed_policy)
                            if result.native_requested_seed is not None:
                                row['native_requested_seed'] = result.native_requested_seed
                        if handler.messages:
                            row['warnings'] = list(handler.messages)
                        break
                    except Exception as exc:
                        dest.unlink(missing_ok=True)
                        row.update(status='generation_failed', error=f'{type(exc).__name__}: {exc}',
                                   warnings=list(handler.messages))
                        logger.warning('Sample %s attempt %d failed: %s', row['sample_id'], attempt + 1, exc)
                        # These CUDA errors invalidate the process's context. Retrying
                        # or evaluating subsequent samples would produce dependent
                        # failures rather than independent benchmark observations.
                        if invalid_cuda_context(exc):
                            row['fatal_runtime_error'] = True
                            append_sample_event(events, row)
                            raise
            generated_count += bool(row.get('generated_sha256'))
            failed_count += row['status'] in ('input_failed', 'generation_failed')
            append_sample_event(events, row)
            logger.info('[%d/%d] %s; generated=%d failed=%d', i + 1, len(rows), row['status'],
                        generated_count, failed_count)
        # Close workers even for generation-only runs, before loading any scorers.
        if backend is not None:
            backend.close()
            backend = None
        with (audio_dir / 'synthesis_timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['generated_path', 'synthesis_time_sec', 'timing_scope', 'adapter_call_time_sec',
                                                   'request_wall_time_sec', 'request_timing_phase'])
            writer.writeheader()
            writer.writerows({k: r.get(k) for k in writer.fieldnames} for r in rows if r.get('generated_sha256'))
        if not generate_only and any(row.get('generated_sha256') for row in rows):
            from rvcbench.evaluation.pipeline import evaluate_run
            scorer_device = (evaluation.get('device') or
                             OmegaConf.select(conf, 'evaluation.device') or str(device))
            manifest['evaluation_device'] = str(scorer_device)
            metrics = evaluate_run(rows, required, Path(exp_dir), scorer_device, logger, seed=seed,
                cap=evaluation.get('generated_audio_max_seconds',
                    OmegaConf.select(conf, 'evaluation.generated_audio_max_seconds')),
                bootstrap_config=OmegaConf.select(conf, 'evaluation.bootstrap'),
                wer_normalization=OmegaConf.select(conf, 'evaluation.wer_normalization', default='ascii_punctuation_removed_v2'),
                cache_sources=[source] if source else [],
                on_sample=lambda row: append_sample_event(events, row))
            manifest['evaluated'] = True
            manifest['evaluation_fingerprint'] = metrics['evaluation_fingerprint']
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
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                manifest['cleanup_error'] = f'{type(exc).__name__}: {exc}'
                manifest['status'] = 'failed'
                logger.error('Backend cleanup failed: %s', exc)
        logger.removeHandler(handler)
        manifest['elapsed_sec'] = time.perf_counter() - started
        manifest['coverage'] = coverage(rows, required, evaluated=manifest['evaluated'])
        atomic_json(path, manifest)
