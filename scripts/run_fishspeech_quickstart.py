#!/usr/bin/env python3
"""
Quickstart script: FishSpeech zero-shot voice cloning on RVCBench.

Mirrors rvcbench_fishspeech_quickstart.ipynb as a runnable script.
All outputs are written inside the repository directory.

This script expects the Fish-Speech inference repo and the `fishaudio/s1-mini`
checkpoint to have been prepared already. It validates the public Hugging Face
dataset layout used by the notebook and can be smoke-tested with `--dry-run`.

Usage (from repo root):
    python scripts/run_fishspeech_quickstart.py
    python scripts/run_fishspeech_quickstart.py --speaker-id p226 --max-samples 5
    python scripts/run_fishspeech_quickstart.py --no-hf-download --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for parent in [candidate] + list(candidate.parents):
        if (parent / "run_vc.py").exists():
            return parent
    raise RuntimeError(
        "Cannot locate repo root (run_vc.py not found). "
        "Run this script from inside the RVCBench repository."
    )


REPO_DIR = _find_repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-id", default="p225", help="VCTK speaker ID")
    parser.add_argument("--max-samples", type=int, default=3, help="Samples to generate")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. cuda:0")
    parser.add_argument("--hf-config", default="VCTK", help="HF dataset config name")
    parser.add_argument("--fish-repo-dir", default=str(REPO_DIR / "checkpoints" / "fish_speech"),
                        help="Local Fish-Speech repo checkout")
    parser.add_argument("--fish-ckpt-dir", default=str(REPO_DIR / "checkpoints" / "fish_speech" / "openaudio-s1-mini"),
                        help="Local fishaudio/s1-mini checkpoint directory")
    parser.add_argument("--no-hf-download", action="store_true",
                        help="Skip HF download (data already present in data/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset/layout and print the run_vc.py command without launching it")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data", help="Dataset download/cache directory")
    parser.add_argument("--evaluate", action="store_true", help="Also score outputs; requires compatible .[eval] dependencies")
    return parser.parse_args()


def download_data(data_dir: Path, hf_config: str, speaker_id: str) -> Path:
    from huggingface_hub import snapshot_download

    dataset_root = data_dir / hf_config
    print(f"[data] Downloading {hf_config}/{speaker_id} from Nanboy/RVCBench …")
    snapshot_download(
        repo_id="Nanboy/RVCBench",
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=[
            f"{hf_config}/metadata.parquet",
            f"{hf_config}/audios/{speaker_id}/*.wav",
            f"{hf_config}/audios/{speaker_id}/*.bert.pt",
        ],
    )
    return dataset_root


def validate_dataset_root(dataset_root: Path, speaker_id: str) -> tuple[list[Path], list[Path]]:
    metadata_path = dataset_root / "metadata.parquet"
    speaker_dir = dataset_root / "audios" / speaker_id
    wav_files = sorted(speaker_dir.glob("*.wav"))
    bert_files = sorted(speaker_dir.glob("*.bert.pt"))

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    if not speaker_dir.exists():
        raise FileNotFoundError(f"Missing speaker directory: {speaker_dir}")
    if not wav_files:
        raise FileNotFoundError(f"No wav files found under: {speaker_dir}")

    print(f"[data] metadata      = {metadata_path}")
    print(f"[data] speaker dir   = {speaker_dir}")
    print(f"[data] wav files     = {len(wav_files)}")
    print(f"[data] bert files    = {len(bert_files)}")
    return wav_files, bert_files


def validate_fishspeech_paths(fish_repo_dir: Path, fish_ckpt_dir: Path, dry_run: bool) -> None:
    if dry_run:
        if not fish_repo_dir.exists():
            print(f"[fish] repo dir      = {fish_repo_dir} (missing; allowed in dry-run)")
        else:
            print(f"[fish] repo dir      = {fish_repo_dir}")
        if not fish_ckpt_dir.exists():
            print(f"[fish] ckpt dir      = {fish_ckpt_dir} (missing; allowed in dry-run)")
        else:
            print(f"[fish] ckpt dir      = {fish_ckpt_dir}")
        return
    if not fish_repo_dir.exists():
        raise FileNotFoundError(f"Fish-Speech repo checkout not found: {fish_repo_dir}")
    if not fish_ckpt_dir.exists():
        raise FileNotFoundError(f"FishSpeech checkpoint dir not found: {fish_ckpt_dir}")
    print(f"[fish] repo dir      = {fish_repo_dir}")
    print(f"[fish] ckpt dir      = {fish_ckpt_dir}")


def build_run_vc_cmd(
    dataset_root: Path,
    speaker_id: str,
    max_samples: int,
    device: str,
    fish_repo_dir: Path,
    fish_ckpt_dir: Path,
) -> list[str]:
    return [
        sys.executable, str(REPO_DIR / "run_vc.py"),
        "--config-name", "ots_vc/clean/vctk/fishspeech_ots",
        f"device={device}",
        f"dataset.root_path={dataset_root}",
        "dataset.use_hf_dataset=false",
        f"dataset.speaker_id={speaker_id}",
        "batch_size=1",
        f"adversary.max_samples={max_samples}",
        f"adversary.code_path={fish_repo_dir}",
        f"adversary.llama_checkpoint_path={fish_ckpt_dir}",
        f"adversary.decoder_checkpoint_path={fish_ckpt_dir / 'codec.pth'}",
    ]


def print_results(results_dir: Path) -> None:
    run_dir = max(results_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    gen = metrics.get("generation_evaluation", {})
    generated_wavs = sorted((run_dir / "generated_audio").rglob("*.wav"))

    print("\n" + "=" * 55)
    print("Generation Metrics")
    print("=" * 55)
    for key in ("avg_mcd", "avg_wer", "avg_sim", "avg_speechmos_mos",
                "avg_dnsmos_ovrl", "emotion_match_rate", "rtf"):
        if isinstance(gen.get(key), (int, float)):
            print(f"  {key:<28} {gen[key]:.4f}")

    print(f"\nGenerated {len(generated_wavs)} audio file(s) → {run_dir / 'generated_audio'}")
    print(f"Full metrics: {run_dir / 'metrics.json'}")


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")

    if args.dry_run:
        device = args.device or "cpu"
    else:
        import torch
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    os.environ.setdefault("DATA_LOADER_WORKERS", "0")
    os.environ.setdefault("AUDIOBENCH_FORCE_OFFLINE", "0")

    data_dir = args.data_dir.resolve()
    dataset_root = data_dir / args.hf_config
    fish_repo_dir = Path(args.fish_repo_dir)
    fish_ckpt_dir = Path(args.fish_ckpt_dir)

    if not args.no_hf_download:
        dataset_root = download_data(data_dir, args.hf_config, args.speaker_id)
    else:
        assert dataset_root.exists(), f"Dataset root not found: {dataset_root}"

    validate_dataset_root(dataset_root, args.speaker_id)
    validate_fishspeech_paths(fish_repo_dir, fish_ckpt_dir, args.dry_run)

    cmd = build_run_vc_cmd(
        dataset_root=dataset_root,
        speaker_id=args.speaker_id,
        max_samples=args.max_samples,
        device=device,
        fish_repo_dir=fish_repo_dir,
        fish_ckpt_dir=fish_ckpt_dir,
    )

    print(f"\n[config] REPO_DIR     = {REPO_DIR}")
    print(f"[config] DATASET_ROOT = {dataset_root}")
    print(f"[config] DEVICE       = {device}")
    print(f"[config] SPEAKER_ID   = {args.speaker_id}")
    print(f"[config] MAX_SAMPLES  = {args.max_samples}")
    print(f"[config] FISH_REPO    = {fish_repo_dir}")
    print(f"[config] FISH_CKPT    = {fish_ckpt_dir}\n")
    if not args.evaluate:
        cmd.append("+vc.generate_only=true")
    print("[vc] Command:", " ".join(cmd))

    if args.dry_run:
        print("\nDry run complete.")
        return

    subprocess.run(cmd, cwd=str(REPO_DIR), check=True)
    print_results(REPO_DIR / "results" / "fishspeech_ots_on_vctk")
    print("\nDone.")


if __name__ == "__main__":
    main()
