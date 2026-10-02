"""Raw timing aggregation and fail-closed ranking authorization.

No existing backend has an audited common measurement boundary. Scope equality
alone (especially the legacy generic label) cannot authorize speed rankings.
"""
import math


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
