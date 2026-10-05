"""Small user-facing commands for preflight, smoke checks and validated reports."""
import argparse
import importlib
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path


# Commands implemented as Hydra applications: everything after the command name
# (--config-name, key=value overrides, --help) is parsed by Hydra.
HYDRA_COMMANDS = {
    'run': ('rvcbench.entrypoints.vc', 'Run voice cloning and evaluation from a config'),
    'run-protected': ('rvcbench.entrypoints.vc_protect', 'Run voice cloning from protected reference audio'),
    'protect': ('rvcbench.entrypoints.protect', 'Protect source audio and measure fidelity'),
    'denoise': ('rvcbench.entrypoints.denoise', 'Denoise protected audio before cloning'),
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in HYDRA_COMMANDS:
        module = importlib.import_module(HYDRA_COMMANDS[sys.argv[1]][0])
        sys.argv = [f'rvcbench {sys.argv[1]}', *sys.argv[2:]]
        return module.main()
    parser = argparse.ArgumentParser(prog='rvcbench')
    from rvcbench import __version__
    parser.add_argument('--version', action='version', version=f'rvcbench {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, (_, description) in HYDRA_COMMANDS.items():
        commands.add_parser(name, help=description + ' (Hydra: --config-name NAME key=value ...)', add_help=False)
    doctor = commands.add_parser('doctor', help='Check dependencies without loading models or downloading weights')
    doctor.add_argument('--model', choices=['qwen3', 'qwen3_omni', 'sparktts'], default=None)
    doctor.add_argument('--eval', action='store_true')
    doctor.add_argument('--imports', action='store_true', help='Also import dependencies in an isolated subprocess to detect binary/version conflicts')
    prompts = commands.add_parser('prompts', help='Export the reference audio and texts of a benchmark suite')
    prompts.add_argument('--suite', default='onboarding-v1')
    prompts.add_argument('--output', type=Path, required=True)
    prompts.add_argument('--data-root', type=Path, help='Local dataset copy laid out like the Hub dataset (default: download)')
    score = commands.add_parser('score', help='Score audio you generated for a suite and write submission.json',
                                description='Score one model, or several models at once: with several --generated '
                                            'directories each model is written to OUTPUT/<model>/ and all of them '
                                            'are compared in OUTPUT/comparison.md.')
    score.add_argument('--suite', default='onboarding-v1')
    score.add_argument('--generated', type=Path, nargs='+', required=True,
                       help='Directory of <id>.wav (or <task>/<pair_id>.wav) files; several directories score several models')
    score.add_argument('--model', nargs='+', help='Model name per --generated directory (default: the directory name)')
    score.add_argument('--output', type=Path, required=True,
                       help='Results directory of one model; with several models, the parent of one directory per model')
    score.add_argument('--device', default='cpu')
    score.add_argument('--resume', action='store_true', help='Resume this output directory; verify inputs and reuse matching scores')
    score.add_argument('--data-root', type=Path, help='Local dataset copy laid out like the Hub dataset (default: download)')
    compare = commands.add_parser('compare', help='Compare several models scored on the same suite in one table')
    compare.add_argument('submissions', nargs='+', type=Path, help='submission.json files or results directories')
    compare.add_argument('--output', type=Path, help='Write comparison.md, .csv and .json here')
    compare.add_argument('--allow-incompatible', action='store_true', help='Inspect differing scoring protocols without ranking')
    setup = commands.add_parser('setup-scorers', help='Download and verify the model files of the scoring metrics')
    setup.add_argument('--metrics', nargs='+', default=['sim', 'speechmos', 'wer', 'mcd', 'emotion', 'stoi'])
    setup.add_argument('--check-only', action='store_true', help='Verify files that are present; download nothing')
    status = commands.add_parser('status', help='Show run status and recovered sample coverage')
    status.add_argument('run_dir', type=Path)
    audit = commands.add_parser('audit-source', help='Compare recorded generation source hashes with local files')
    audit.add_argument('run_dir', type=Path)
    audit.add_argument('--root', type=Path, default=None, help='Checkout or install root (default: this installation)')
    audit.add_argument('--output', type=Path)
    smoke = commands.add_parser('smoke', help='Run a synthetic CPU pipeline check (no model download)')
    smoke.add_argument('--output', type=Path, default=Path('results/smoke'))
    report = commands.add_parser('report', help='Validate a complete run and export a provenance-bearing JSON report')
    report.add_argument('run_dir', type=Path)
    report.add_argument('--output', type=Path, required=True)
    replay = commands.add_parser('replay-gr', help='Replay frozen GR batch noise and verify historical WAV hashes')
    for name in ('dataset-root', 'subset-manifest', 'noise-archive', 'historical-directory', 'output'):
        replay.add_argument('--' + name, type=Path, required=True)
    replay.add_argument('--batch-size', type=int, default=8)
    replay.add_argument('--sample-rate', type=int, default=24000)
    replay.add_argument('--hop-length', type=int, default=512)
    replay.add_argument('--regenerate-rng', action='store_true', help='Regenerate and verify the entire cohort noise stream')
    replay.add_argument('--seed', type=int, default=42)
    replay.add_argument('--epsilon', type=float, default=0.03137255)
    replay.add_argument('--device', default='cpu')
    denoise = commands.add_parser('denoise-dns64', help='Enhance selected references with explicit local DNS64 weights')
    for name in ('dataset-root', 'subset-manifest', 'reference-directory', 'weights', 'output'):
        denoise.add_argument('--' + name, type=Path, required=True)
    denoise.add_argument('--device', default='cpu')
    denoise.add_argument('--dry', type=float, default=0.0)
    denoise.add_argument('--dataset-rate', type=int, default=16000)
    denoise.add_argument('--runtime-python', type=Path, help='Run inference in an isolated model interpreter')
    denoise.add_argument('--timeout-seconds', type=float, default=600)
    enkidu = commands.add_parser('protect-enkidu', help='Train Enkidu on the full cohort and emit selected references')
    for name in ('dataset-root', 'subset-manifest', 'model-directory', 'output'):
        enkidu.add_argument('--' + name, type=Path, required=True)
    enkidu.add_argument('--device', default='cpu')
    enkidu.add_argument('--seed', type=int, default=42)
    enkidu.add_argument('--epochs', type=int, default=10)
    compare = commands.add_parser('compare-check', help='Verify whether two complete runs share a comparison protocol')
    compare.add_argument('left', type=Path)
    compare.add_argument('right', type=Path)
    compare.add_argument('--metrics', nargs='+', default=['mcd', 'wer', 'sim'])
    compare.add_argument('--output', type=Path)
    timing = commands.add_parser('compare-timing', help='Verify matched fresh request wall timings')
    timing.add_argument('left', type=Path)
    timing.add_argument('right', type=Path)
    timing.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'setup-scorers':
        from rvcbench.evaluation.setup import setup_scorers
        print(json.dumps(setup_scorers(args.metrics, check_only=args.check_only), indent=2))
        return
    if args.command == 'compare':
        from .comparison import compare_submissions
        result = compare_submissions(args.submissions, args.output, allow_incompatible=args.allow_incompatible)
        print(result['markdown'])
        if args.output:
            print(f"Wrote {args.output / 'comparison.md'}, .csv and .json")
        return
    if args.command in ('prompts', 'score'):
        from .submission import export_prompts, score_submission, score_submissions
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
        if args.command == 'prompts':
            result = export_prompts(args.suite, args.output, data_root=args.data_root)
            print(json.dumps(result, indent=2))
            print(f"Generate the {result['prompts']} utterances listed in {args.output / 'prompts.jsonl'} "
                  f"(or prompts.tsv / prompts.lst for batch scripts); see {args.output / 'README.md'}.")
            return
        if len(args.generated) > 1:
            result = score_submissions(args.suite, args.generated, args.output, models=args.model,
                                       device=args.device, data_root=args.data_root, resume=args.resume)
            print(result['markdown'])
            if not result['leaderboard']:
                print(f"Note: {result['label']}")
            print(f"Wrote {args.output / 'comparison.md'} and one directory per model under {args.output}")
            raise SystemExit(any(m['status'] != 'complete' for m in result['models']))
        if args.model and len(args.model) != 1:
            parser.error('--model takes one name per --generated directory')
        model = args.model[0] if args.model else args.generated[0].resolve().name
        result = score_submission(args.suite, args.generated[0], args.output, model=model,
                                  device=args.device, data_root=args.data_root, resume=args.resume)
        print(json.dumps({'suite': result['suite'], 'model': result['model'], 'status': result['status'],
                          'tasks': {name: {'status': task['status'], 'means': task.get('means'),
                                           'failed': len(task['failures'])}
                                    for name, task in result['tasks'].items()}}, indent=2))
        if not result['leaderboard']:
            print(f"Note: {result['label']}")
        print(f"Wrote {args.output / 'submission.json'}")
        raise SystemExit(result['status'] != 'complete')
    if args.command == 'audit-source':
        from .source_audit import audit_recorded_sources
        from .artifacts import atomic_json
        from .artifacts import runtime_root
        result = audit_recorded_sources(args.run_dir, args.root or runtime_root())
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(result['status'] != 'recorded_sources_match')
    if args.command == 'compare-timing':
        from .timing import check_timing_comparability
        from .artifacts import atomic_json
        result = check_timing_comparability(args.left, args.right)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(result['status'] != 'comparable')
    if args.command == 'protect-enkidu':
        from .enkidu_stage import protect_enkidu
        result = protect_enkidu(args.dataset_root, args.subset_manifest, args.model_directory,
                                args.output, args.device, args.seed, args.epochs)
        print(json.dumps({'status': result['status'], 'verified': result['verified'],
                          'manifest': str(args.output / 'stage_manifest.json')}, indent=2))
        return
    if args.command == 'denoise-dns64':
        from .denoise_stage import denoise_dns64
        result = denoise_dns64(args.dataset_root, args.subset_manifest, args.reference_directory,
            args.weights, args.output, args.device, args.dry, args.dataset_rate,
            args.runtime_python, args.timeout_seconds)
        print(json.dumps({'status': result['status'], 'verified': result['verified'],
                          'manifest': str(args.output / 'stage_manifest.json')}, indent=2))
        return
    if args.command == 'replay-gr':
        from .noise_replay import replay_gr_noise
        result = replay_gr_noise(args.dataset_root, args.subset_manifest, args.noise_archive,
            args.historical_directory, args.output, args.batch_size, args.sample_rate, args.hop_length,
            args.seed if args.regenerate_rng else None, args.epsilon, args.device)
        print(json.dumps({'status': result['status'], 'verified': result['verified'],
                          'manifest': str(args.output / 'stage_manifest.json')}, indent=2))
        return
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
        elif args.model == 'sparktts':
            names += ['numpy', 'torchaudio', 'transformers', 'safetensors', 'einops', 'einx', 'omegaconf', 'librosa', 'yaml']
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
        from rvcbench.datasets.zero_shot import ZeroShotDataset
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


if __name__ == '__main__':
    main()
