"""Timing evidence: raw aggregation, fail-closed ranking authorization for
legacy scopes, and the uniform request wall timing protocol.

Legacy adapter scopes have no audited common measurement boundary, so scope
equality alone (especially the legacy generic label) cannot authorize speed
rankings. The request wall protocol below records its own boundary, hardware
and framework profile and is checked independently of quality scores.
"""
import math
import platform
import os
from pathlib import Path

from .artifacts import SCHEMA_VERSION, digest, file_hash, input_fingerprint, load_run


def _number(value, *, positive=False):
    try:
        value = float(value)
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except (TypeError, ValueError, OverflowError):
        return False


def timing_summary(rows):
    groups = {}
    invalid = 0
    for row in rows:
        elapsed, duration = row.get('synthesis_time_sec'), row.get('generated_duration_sec')
        if not _number(elapsed) or not _number(duration, positive=True):
            invalid += 1
            continue
        scope = row.get('timing_scope') or 'unspecified'
        group = groups.setdefault(scope, dict(synthesis_time_sec=0., generated_duration_sec=0., samples=0))
        group['synthesis_time_sec'] += float(elapsed)
        group['generated_duration_sec'] += float(duration)
        group['samples'] += 1
    for group in groups.values():
        group['rtf'] = group['synthesis_time_sec'] / group['generated_duration_sec']
    only = next(iter(groups.values())) if len(groups) == 1 else None
    raw = only['rtf'] if only and not invalid and 'unspecified' not in groups else None
    reasons = ['No audited common measurement boundary and execution protocol exists for current scopes.']
    if invalid:
        reasons.append('Missing or invalid timing/duration samples.')
    if len(groups) != 1:
        reasons.append('Missing or mixed timing scopes.')
    if 'unspecified' in groups:
        reasons.append('Timing scope is unspecified.')
    return {'rtf': raw, 'rtf_by_scope': groups, 'timing_comparability': {
        'status': 'not_comparable', 'ranking_allowed': False, 'reasons': reasons,
        'scopes': sorted(groups), 'requested_samples': len(rows), 'invalid_samples': invalid}}


def compare_timing(left_rows, right_rows):
    """Quality pairing cannot waive missing timing boundary/hardware evidence."""
    left, right = timing_summary(left_rows), timing_summary(right_rows)
    return {'status': 'not_comparable', 'ranking_allowed': False,
            'reasons': ['Timing boundaries, initialization/cache policy, synchronization and hardware must be audited and matched before comparison.'],
            'left': left, 'right': right, 'rtf_delta': None}


SCOPE = 'backend_request_wall_v1'


def synchronize(device):
    import torch
    resolved = torch.device(device)
    if resolved.type == 'cuda':
        torch.cuda.synchronize(resolved)


def timing_profile(device, conf):
    import torch
    resolved = torch.device(device)
    hardware = {'host': platform.node(), 'machine': platform.machine(),
                'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    if resolved.type == 'cuda':
        props = torch.cuda.get_device_properties(resolved)
        hardware['gpu'] = {'name': props.name, 'total_memory': props.total_memory,
                           'compute_capability': [props.major, props.minor]}
    adversary = conf.adversary
    external = any(adversary.get(k) for k in ('endpoint_url', 'server_url', 'api_url'))
    worker = 'runtime_python' in adversary or 'worker_script_path' in adversary
    spec = {'schema_version': 1, 'scope': SCOPE, 'clock': 'time.perf_counter',
        'boundary': 'before_backend_generate_batch_to_return_and_device_sync',
        'synchronization': 'requested_device_cuda' if resolved.type == 'cuda' else 'none_cpu',
        'includes': ['backend_rng_setup', 'reference_processing', 'generation', 'output_writing', 'deferred_backend_setup'],
        'excludes': ['backend_prepare', 'asset_resolution', 'output_validation', 'scoring'],
        'hardware': hardware, 'device_type': resolved.type,
        'framework': {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'cpu_threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32},
        'external_execution_declared': bool(external or worker)}
    return {'fingerprint': digest(spec), **spec}


def _valid_profile(profile):
    required = {'fingerprint', 'schema_version', 'scope', 'clock', 'boundary',
                'synchronization', 'includes', 'excludes', 'hardware', 'device_type',
                'framework', 'external_execution_declared'}
    if not isinstance(profile, dict) or set(profile) != required:
        return False
    spec = {k:v for k,v in profile.items() if k != 'fingerprint'}
    framework, hardware = profile['framework'], profile['hardware']
    return (profile['schema_version'] == 1 and profile['scope'] == SCOPE and
        profile['fingerprint'] == digest(spec) and profile['clock'] == 'time.perf_counter' and
        profile['boundary'] == 'before_backend_generate_batch_to_return_and_device_sync' and
        profile['device_type'] in ('cpu','cuda') and
        profile['synchronization'] == ('requested_device_cuda' if profile['device_type'] == 'cuda' else 'none_cpu') and
        profile['includes'] == ['backend_rng_setup', 'reference_processing', 'generation', 'output_writing', 'deferred_backend_setup'] and
        profile['excludes'] == ['backend_prepare', 'asset_resolution', 'output_validation', 'scoring'] and
        isinstance(framework,dict) and set(framework) == {'torch','cuda','cudnn','cpu_threads','interop_threads',
            'cudnn_deterministic','cudnn_benchmark','matmul_allow_tf32'} and
        isinstance(framework.get('torch'), str) and bool(framework['torch']) and all(isinstance(framework.get(k),bool) for k in
            ('cudnn_deterministic','cudnn_benchmark','matmul_allow_tf32')) and
        all(isinstance(framework.get(k),int) and not isinstance(framework[k],bool) and framework[k] > 0
            for k in ('cpu_threads','interop_threads')) and
        isinstance(hardware,dict) and bool(hardware.get('host')) and bool(hardware.get('machine')) and
        (profile['device_type'] != 'cuda' or (isinstance(hardware.get('gpu'),dict) and
            set(hardware['gpu']) == {'name','total_memory','compute_capability'} and
            isinstance(hardware['gpu']['name'], str) and bool(hardware['gpu']['name']) and
            isinstance(hardware['gpu']['total_memory'], int) and not isinstance(hardware['gpu']['total_memory'], bool) and
            hardware['gpu']['total_memory'] > 0 and
            isinstance(hardware['gpu']['compute_capability'], list) and
            len(hardware['gpu']['compute_capability']) == 2 and
            all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in hardware['gpu']['compute_capability']))) and
        isinstance(profile['external_execution_declared'],bool))


def check_timing_comparability(left_dir, right_dir):
    directories = [Path(left_dir), Path(right_dir)]
    runs = [load_run(p) for p in directories]
    reasons, profiles, summaries = [], [], []
    for index, run in enumerate(runs):
        rows = run['samples']
        if run.get('schema_version') != SCHEMA_VERSION or run.get('protocol') != 'rvcbench-zero-shot-v2':
            raise ValueError('Timing requires the current generation protocol')
        if not rows or len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Empty or duplicate timing sample identities')
        if input_fingerprint(rows) != run.get('input_fingerprint'):
            raise ValueError('Input fingerprint does not match timing sample records')
        if run.get('status') not in ('generated', 'complete', 'partial'):
            reasons.append(f'run {index}: generation is not complete')
        if run['config']['vc']['model'] == 'smoke':
            reasons.append(f'run {index}: synthetic smoke results')
        profile = run.get('request_timing_profile') or {}
        if not isinstance(profile, dict):
            profile = {}
        if not _valid_profile(profile):
            reasons.append(f'run {index}: missing or inconsistent request timing profile')
        adversary = run['config'].get('adversary', {})
        external = any(adversary.get(k) for k in ('endpoint_url', 'server_url', 'api_url'))
        worker = 'runtime_python' in adversary or 'worker_script_path' in adversary
        if profile.get('external_execution_declared') or external or worker:
            reasons.append(f'run {index}: worker/service hardware is not verified by the host timing profile')
        profiles.append(profile)
        phases = {'first_request_after_prepare': [], 'subsequent_request': []}
        for row_index, row in enumerate(rows):
            for kind in ('prompt','target','generated'):
                sha = row.get(kind + '_sha256')
                if not sha or file_hash(row.get(kind + '_path')) != sha:
                    raise ValueError(f'Timing {kind} artifact changed or missing: {row["sample_id"]}')
            duration = row.get('request_wall_time_sec')
            valid = (isinstance(duration, (int,float)) and not isinstance(duration,bool)
                     and math.isfinite(duration) and duration > 0)
            if not valid:
                reasons.append(f'run {index}: missing or invalid request time for {row["sample_id"]}')
            config_seed = (run.get('generation_config') or {}).get('seed')
            source_index = row.get('source_index')
            if (not isinstance(config_seed, int) or not isinstance(source_index, int) or
                    row.get('seed') != config_seed + source_index):
                reasons.append(f'run {index}: request seed differs from the original source index for {row["sample_id"]}')
            if row.get('conditioning_variant') == 'prompt_audio_only_after_transcript_failure':
                reasons.append(f'run {index}: transcript-conditioning retry for {row["sample_id"]}')
            if row.get('reused') or row.get('attempts') != 1:
                reasons.append(f'run {index}: reused or retried timing for {row["sample_id"]}')
            phase = row.get('request_timing_phase')
            expected_phase = 'first_request_after_prepare' if row_index == 0 else 'subsequent_request'
            if phase != expected_phase:
                reasons.append(f'run {index}: missing timing phase for {row["sample_id"]}')
            elif valid:
                phases[phase].append(duration)
            if (not profile.get('fingerprint') or
                    row.get('request_timing_profile_fingerprint') != profile.get('fingerprint') or
                    row.get('request_timing_profile_after_fingerprint') != profile.get('fingerprint')):
                reasons.append(f'run {index}: timing settings changed or unrecorded for {row["sample_id"]}')
        if len(phases['first_request_after_prepare']) != 1:
            reasons.append(f'run {index}: expected one first request after backend preparation')
        summaries.append({phase: {'count': len(values), 'total_seconds': sum(values),
            'mean_seconds': sum(values)/len(values) if values else None} for phase,values in phases.items()})
    for key in ('protocol','input_fingerprint'):
        if not runs[0].get(key) or runs[0].get(key) != runs[1].get(key):
            reasons.append(f'{key} differs or is missing')
    for key in ('seed', 'sample_seed_policy'):
        values = [(r.get('generation_config') or {}).get(key) for r in runs]
        if values[0] is None or values[0] != values[1]:
            reasons.append(f'Generation {key} differs or is missing')
    if profiles[0] != profiles[1]:
        reasons.append('request timing scope, hardware or framework settings differ')
    if directories[0].resolve() == directories[1].resolve():
        reasons.append('comparison refers to the same generation run')
    return {'status': 'comparable' if not reasons else 'incompatible', 'reasons': reasons,
        'scope': 'Recorded host request wall latency; first and subsequent requests remain separate. Does not establish isolated-device, inference-only or paper-table timing equivalence.',
        'models': [r['config']['vc']['model'] for r in runs],
        'requested_pairs': [len(r['samples']) for r in runs], 'timing': summaries,
        'sources': [{'manifest': str((p/'run_manifest.json').resolve()),
                     'sha256': file_hash(p/'run_manifest.json')} for p in directories]}
