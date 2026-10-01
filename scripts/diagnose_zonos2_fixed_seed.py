#!/usr/bin/env python3
"""Finite fixed-native-seed control; produces a diagnostic, not a benchmark run."""
import argparse
import copy
import json
from dataclasses import fields
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
import soundfile as sf

from src.adversary.zonos2_ots import resolve_zonos_language
from src.benchmark.artifacts import atomic_json, file_hash, load_run
from src.benchmark.fingerprints import generation_runtime, worker_environment
from src.benchmark.model_assets import resolve_model_assets
from src.models.zonos2 import Zonos2Generator, Zonos2GeneratorConfig
from src.utils.seeding import configure_seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-only', action='store_true', help='Use a separate evaluation environment')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reference-score-run', type=Path)
    args = parser.parse_args()
    source = load_run(args.source_run)
    if source['config']['vc']['model'] != 'zonos2' or source['status'] not in ('generated', 'complete'):
        raise ValueError('Control requires a successfully generated ZONOS2 source run')
    if args.score_only:
        from src.evaluation.pipeline import evaluate_run
        report = json.loads((args.output / 'control_manifest.json').read_text())
        if (report['source_manifest_sha256'] != file_hash(args.source_run / 'run_manifest.json')
                or report['status'] != 'generated'):
            raise ValueError('Control source changed or generation is incomplete')
        original = {row['sample_id']: row for row in source['samples']}
        rows = report['samples']
        if len(rows) != len(original) or {r['sample_id'] for r in rows} != set(original):
            raise ValueError('Control population differs from source run')
        for row in rows:
            for key in ('source_index', 'seed', 'speaker_id', 'prompt_text', 'target_text', 'target_language'):
                if row.get(key) != original[row['sample_id']].get(key):
                    raise ValueError(f'Control input metadata differs: {key}')
            for kind in ('prompt', 'target'):
                if (row[kind + '_sha256'] != original[row['sample_id']][kind + '_sha256']
                        or file_hash(row[kind + '_path']) != row[kind + '_sha256']):
                    raise ValueError('Control inputs differ from source run')
            if file_hash(row['generated_path']) != row['generated_sha256']:
                raise ValueError('Control output changed')
        logging.basicConfig(level=logging.INFO)
        metrics = evaluate_run(rows, ['mcd', 'wer', 'sim'], args.output / 'scores', args.device,
            logging.getLogger('control-score'), seed=int(source['generation_config']['seed']),
            wer_normalization='ascii_punctuation_removed_v2')
        result = dict(kind='native_seed_control_scores', driver_sha256=file_hash(__file__),
                      control_manifest_sha256=file_hash(args.output / 'control_manifest.json'),
                      metrics=metrics, samples=rows)
        if args.reference_score_run:
            from src.benchmark.artifacts import validate_report
            baseline = load_run(args.reference_score_run)
            validate_report(baseline)
            baseline_rows = {r['sample_id']: r for r in baseline['samples']}
            if set(baseline_rows) != set(original):
                raise ValueError('Reference score population differs')
            if baseline['input_fingerprint'] != source['input_fingerprint'] or baseline['protocol'] != source['protocol']:
                raise ValueError('Reference score inputs or protocol differ')
            for row in rows:
                if any(row.get(key) != baseline_rows[row['sample_id']].get(key)
                       for key in ('seed', 'prompt_sha256', 'target_sha256', 'prompt_text', 'target_text')):
                    raise ValueError('Reference score sample metadata differs')
            from src.benchmark.artifacts import digest
            scoring_paths = (args.output / 'scores' / 'scoring_manifest.json',
                             args.reference_score_run / 'scoring_manifest.json')
            scorers = [json.loads(path.read_text()) for path in scoring_paths]
            for metric in ('mcd', 'wer', 'sim'):
                records = [scorer[metric] for scorer in scorers]
                if any(record['fingerprint'] != digest({k: v for k, v in record.items() if k != 'fingerprint'})
                       for record in records) or records[0] != records[1]:
                    raise ValueError(f'Reference {metric} scorer differs or fingerprint is inconsistent')
            result['reference_score_manifest_sha256'] = file_hash(args.reference_score_run / 'run_manifest.json')
            result['paired_mean_delta'] = {metric: sum(r['metrics'][metric] - baseline_rows[r['sample_id']]['metrics'][metric]
                for r in rows) / len(rows) for metric in ('mcd', 'wer', 'sim')}
        atomic_json(args.output / 'control_scores.json', result)
        return
    conf = OmegaConf.create(source['config'])
    environment = worker_environment(sys.executable)
    runtime = generation_runtime(Path(__file__).resolve().parents[1], conf, environment['packages'])
    if runtime != source['generation_provenance']:
        raise ValueError('Generation source/environment differs from the source run')
    _, assets, asset_fingerprint = resolve_model_assets(conf)
    if asset_fingerprint != source['generation_config']['model_assets_fingerprint']:
        raise ValueError('Generation assets differ from the source run')
    for row in source['samples']:
        for kind in ('prompt', 'target', 'generated'):
            if file_hash(row[kind + '_path']) != row[kind + '_sha256']:
                raise ValueError(f'Source {kind} audio changed: {row["sample_id"]}')
    config = Zonos2GeneratorConfig(**{f.name: conf.adversary[f.name]
        for f in fields(Zonos2GeneratorConfig) if f.name in conf.adversary})
    if config.seed is None:
        raise ValueError('Control requires an explicit native seed')
    if config.native_seed_policy != 'source_index':
        raise ValueError('Control requires the source-index native seed policy')
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('zonos2-seed-control')
    report = dict(kind='native_seed_intervention', status='preparing',
        source_manifest=str((args.source_run / 'run_manifest.json').resolve()),
        source_manifest_sha256=file_hash(args.source_run / 'run_manifest.json'),
        driver_sha256=file_hash(__file__), runtime=runtime, model_reference=assets,
        intervention='Only native sampling seed: fixed base seed instead of base plus source index.',
        timing_scope='diagnostic_no_timing_comparability_claim', samples=[])
    manifest = args.output / 'control_manifest.json'
    atomic_json(manifest, report)
    generator = Zonos2Generator(config, source['generation_config']['device'], logger)
    try:
        generator.ensure_model()
        report['status'] = 'generating'
        for original in source['samples']:
            row = copy.deepcopy(original)
            configure_seeds(row['seed'])  # Keep the source run's global RNG seed.
            # Source index is retained in the row; zero is the explicit intervention
            # argument that makes this wrapper send native seed=base_seed.
            wave, rate = generator.generate(row['target_text'], row['prompt_path'],
                row['prompt_text'], sample_index=0,
                language=resolve_zonos_language(conf.adversary.get('language'), row.get('target_language')))
            output = args.output / 'generated_audio' / row['speaker_id'] / Path(original['generated_path']).name
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, wave, rate)
            row.update(generated_path=str(output.resolve()), generated_sha256=file_hash(output),
                       native_seed=int(config.seed), native_sample_index_argument=0,
                       metrics={}, metric_errors={}, status='generated', error=None,
                       synthesis_time_sec=None, timing_scope=None)
            report['samples'].append(row)
            atomic_json(manifest, report)
            logger.info('%d/%d control samples generated', len(report['samples']), len(source['samples']))
        report['status'] = 'generated'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            generator.close()
        except BaseException as exc:
            report.update(status='failed', cleanup_error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            atomic_json(manifest, report)


if __name__ == '__main__':
    main()
