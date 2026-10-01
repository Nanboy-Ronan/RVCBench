"""Small user-facing commands for preflight, smoke checks and validated reports."""
import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='rvcbench')
    commands = parser.add_subparsers(dest='command', required=True)
    doctor = commands.add_parser('doctor', help='Check dependencies without loading models or downloading weights')
    doctor.add_argument('--model', choices=['qwen3', 'qwen3_omni'], default=None)
    doctor.add_argument('--eval', action='store_true')
    doctor.add_argument('--imports', action='store_true', help='Also import dependencies in an isolated subprocess to detect binary/version conflicts')
    status = commands.add_parser('status', help='Show run status and recovered sample coverage')
    status.add_argument('run_dir', type=Path)
    smoke = commands.add_parser('smoke', help='Run a synthetic CPU pipeline check (no model download)')
    smoke.add_argument('--output', type=Path, default=Path('results/smoke'))
    report = commands.add_parser('report', help='Validate a complete run and export a provenance-bearing JSON report')
    report.add_argument('run_dir', type=Path)
    report.add_argument('--output', type=Path, required=True)
    compare = commands.add_parser('compare-check', help='Verify whether two complete runs share a comparison protocol')
    compare.add_argument('left', type=Path)
    compare.add_argument('right', type=Path)
    compare.add_argument('--metrics', nargs='+', default=['mcd', 'wer', 'sim'])
    compare.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'compare-check':
        from .comparability import check_comparability
        from .artifacts import atomic_json
        result = check_comparability(args.left, args.right, args.metrics)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(result['status'] != 'comparable')
    if args.command == 'doctor':
        names = ['torch', 'hydra', 'pandas', 'pyarrow', 'soundfile']
        if args.model == 'qwen3':
            names += ['qwen_tts']
        elif args.model == 'qwen3_omni':
            names += ['transformers', 'qwen_omni_utils', 'accelerate']
        if args.eval:
            names += ['torchaudio', 'whisper', 'speechbrain', 'pymcd', 'jiwer', 'torch_stoi']
        missing = [n for n in names if importlib.util.find_spec(n) is None]
        import_errors = {}
        if args.imports:
            for name in names:
                if name in missing:
                    continue
                proc = subprocess.run([sys.executable, '-c', 'import ' + name], capture_output=True, text=True, timeout=120)
                if proc.returncode:
                    import_errors[name] = proc.stderr[-2000:]
            if args.model == 'qwen3_omni' and 'transformers' not in missing and 'transformers' not in import_errors:
                proc = subprocess.run([sys.executable, '-c',
                    'from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor'],
                    capture_output=True, text=True, timeout=120)
                if proc.returncode:
                    import_errors['qwen3_omni_symbols'] = proc.stderr[-2000:]
        print(json.dumps({'missing': missing, 'import_errors': import_errors, 'status': 'failed' if missing or import_errors else 'dependencies_present',
                          'note': 'Import availability only; does not validate GPU memory, weights, or inference.'}, indent=2))
        raise SystemExit(bool(missing or import_errors))
    if args.command == 'status':
        from .artifacts import load_run
        manifest = load_run(args.run_dir)
        print(json.dumps({'status': manifest['status'], 'coverage': manifest['coverage'],
                          'error': manifest.get('error')}, indent=2))
        return
    if args.command == 'report':
        from .artifacts import atomic_json, validate_report, file_hash, metric_means
        source = args.run_dir / 'run_manifest.json'
        manifest = json.loads(source.read_text())
        validate_report(manifest)
        if manifest['config']['vc']['model'] == 'smoke':
            parser.error('Synthetic smoke runs cannot be exported as benchmark results')
        atomic_json(args.output, {'schema_version': 1, 'source_manifest': str(source.resolve()),
                                 'source_sha256': file_hash(source),
                                 'means': metric_means(manifest['samples'], manifest['coverage']['required_metrics']),
                                 'run': manifest})
        print(args.output)
    else:
        import math
        import struct
        import wave
        from omegaconf import OmegaConf
        from src.datasets.zero_shot import ZeroShotDataset
        from .runner import run_zero_shot
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        wav = output / 'fixture.wav'
        with wave.open(str(wav), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(b''.join(struct.pack('<h', int(1000 * math.sin(i * .1))) for i in range(1600)))
        (output / 'metadata.json').write_text(json.dumps([{
            'pair_id': 'smoke-0', 'speaker_id': 'fixture', 'prompt_file_name': wav.name,
            'target_file_name': wav.name, 'prompt_text': 'fixture', 'target_text': 'fixture'}]))
        conf = OmegaConf.create({'base_dir': str(Path.cwd()), 'seed': 42,
            'vc': {'mode': 'ots', 'model': 'smoke', 'generate_only': True}, 'adversary': {},
            'dataset': {'root_path': str(output), 'use_hf_dataset': False, 'manifest_filename': 'metadata.json'}})
        logger = logging.getLogger('rvcbench.smoke')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        dataset = ZeroShotDataset(conf, conf.dataset, logger)
        metrics, _, _ = run_zero_shot(conf, Path.cwd(), 'cpu', dataset, output, logger)
        print(json.dumps(metrics, indent=2))
        print('Synthetic pipeline check passed; this is not a model evaluation.')
