#!/usr/bin/env python3
"""
Quickstart script: Qwen3-TTS zero-shot voice cloning on RVCBench.

Mirrors rvcbench_qwen3tts_quickstart.ipynb as a runnable script.
All outputs are written inside the repository directory.

Usage (from repo root, qwen3 conda env):
    python scripts/run_qwen3tts_quickstart.py
    python scripts/run_qwen3tts_quickstart.py --speaker-id 1272 --max-samples 10
    python scripts/run_qwen3tts_quickstart.py --no-hf-download  # data already present
    python scripts/run_qwen3tts_quickstart.py --no-hf-download --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Repo root resolution — works whether launched from repo root or notebooks/
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-id",   default="1089",   help="LibriTTS speaker ID")
    parser.add_argument("--max-samples",  type=int, default=5, help="Samples to generate")
    parser.add_argument("--device",       default=None,     help="CUDA device, e.g. cuda:0")
    parser.add_argument("--hf-config",    default="Libritts", help="HF dataset config name")
    parser.add_argument("--qwen-checkpoint-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="HF model ID or local path for the Qwen3-TTS checkpoint")
    parser.add_argument("--no-hf-download", action="store_true",
                        help="Skip HF download (data already in data/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset/layout and print the run_vc.py command without launching it")
    parser.add_argument("--evaluate", action="store_true", help="Also run the full metric stack (install .[eval] first)")
    parser.add_argument("--resume-from", type=Path, help="Reuse verified outputs from an earlier run directory")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data", help="Dataset download/cache directory")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HF data download
# ---------------------------------------------------------------------------

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
        ],
    )
    wav_files = sorted((dataset_root / "audios" / speaker_id).glob("*.wav"))
    print(f"[data] {len(wav_files)} wav files for speaker {speaker_id} → {dataset_root}")
    return dataset_root


def validate_dataset_root(dataset_root: Path, speaker_id: str) -> list[Path]:
    metadata_path = dataset_root / "metadata.parquet"
    speaker_dir = dataset_root / "audios" / speaker_id
    wav_files = sorted(speaker_dir.glob("*.wav"))

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    if not speaker_dir.exists():
        raise FileNotFoundError(f"Missing speaker directory: {speaker_dir}")
    if not wav_files:
        raise FileNotFoundError(f"No wav files found under: {speaker_dir}")

    print(f"[data] metadata      = {metadata_path}")
    print(f"[data] speaker dir   = {speaker_dir}")
    print(f"[data] wav files     = {len(wav_files)}")
    return wav_files


# ---------------------------------------------------------------------------
# run_vc.py subprocess call
# ---------------------------------------------------------------------------

def run_vc(
    dataset_root: Path,
    speaker_id: str,
    max_samples: int,
    device: str,
    checkpoint_path: str,
) -> list[str]:
    return [
        sys.executable, str(REPO_DIR / "run_vc.py"),
        "--config-name", "ots_vc/clean/libritts/qwen3_tts_ots",
        f"device={device}",
        f"adversary.device_map={device}",
        f"dataset.root_path={dataset_root}",
        "dataset.use_hf_dataset=false",
        f"dataset.speaker_id={speaker_id}",
        f"adversary.max_samples={max_samples}",
        f"adversary.checkpoint_path={checkpoint_path}",
        "batch_size=1",
    ]


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

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
    print(f"Run manifest: {run_dir / 'run_manifest.json'}")
    if gen.get("coverage"):
        print("Run status:", gen["coverage"]["status"], "| coverage:", gen["coverage"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    data_dir     = args.data_dir.resolve()
    dataset_root = data_dir / args.hf_config

    if not args.no_hf_download:
        dataset_root = download_data(data_dir, args.hf_config, args.speaker_id)
    else:
        assert dataset_root.exists(), f"Dataset root not found: {dataset_root}"

    validate_dataset_root(dataset_root, args.speaker_id)

    print(f"\n[config] REPO_DIR     = {REPO_DIR}")
    print(f"[config] DATASET_ROOT = {dataset_root}")
    print(f"[config] DEVICE       = {device}")
    print(f"[config] SPEAKER_ID   = {args.speaker_id}")
    print(f"[config] QWEN_CKPT    = {args.qwen_checkpoint_path}")
    print(f"[config] MAX_SAMPLES  = {args.max_samples}\n")

    cmd = run_vc(
        dataset_root,
        args.speaker_id,
        args.max_samples,
        device,
        args.qwen_checkpoint_path,
    )
    if not args.evaluate:
        cmd.append("+vc.generate_only=true")
    if args.resume_from:
        cmd.append(f"+vc.resume_from={args.resume_from.resolve()}")
    run_name = "qwen3_tts_quickstart_" + uuid.uuid4().hex[:8]
    cmd.append(f"run_name={run_name}")
    print("[vc] Command:", " ".join(cmd))
    if args.dry_run:
        print("\nDry run complete.")
        return

    subprocess.run(cmd, cwd=str(REPO_DIR), check=True)

    print_results(REPO_DIR / "results" / run_name)
    print("\nDone.")


if __name__ == "__main__":
    main()
