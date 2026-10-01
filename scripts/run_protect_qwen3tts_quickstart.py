#!/usr/bin/env python3
"""
Quickstart script: protection + Qwen3-TTS voice clone attack on RVCBench.

Mirrors rvcbench_safespeech_qwen3tts_quickstart.ipynb as a runnable script.
All outputs are written inside the repository directory.

Step 1  run_protect.py  — perturb audio, measure fidelity
Step 2  run_vc_protect.py — clone from the protected prompts and measure generation quality

Default protection: Gaussian Random Noise (grnoise_on_libritts) — no checkpoints needed.
For SafeSpeech use: --protect-config safespeech_on_libritts
                    (requires BertVits2 checkpoints under checkpoints/)

Usage (from repo root, qwen3 conda env):
    python scripts/run_protect_qwen3tts_quickstart.py
    python scripts/run_protect_qwen3tts_quickstart.py --protect-config safespeech_on_libritts
    python scripts/run_protect_qwen3tts_quickstart.py --speaker-id 1272 --max-samples 10
    python scripts/run_protect_qwen3tts_quickstart.py --no-hf-download  # data already present
    python scripts/run_protect_qwen3tts_quickstart.py --no-hf-download --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for parent in [candidate] + list(candidate.parents):
        if (parent / "run_protect.py").exists():
            return parent
    raise RuntimeError(
        "Cannot locate repo root (run_protect.py not found). "
        "Run this script from inside the RVCBench repository."
    )


REPO_DIR = _find_repo_root()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-id",     default="1089",   help="LibriTTS speaker ID")
    parser.add_argument("--max-samples",    type=int, default=5, help="VC samples to generate")
    parser.add_argument("--device",         default=None,     help="CUDA device, e.g. cuda:0")
    parser.add_argument("--hf-config",      default="Libritts", help="HF dataset config name")
    parser.add_argument("--qwen-checkpoint-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="HF model ID or local path for the Qwen3-TTS checkpoint")
    parser.add_argument("--protect-config", default="grnoise_on_libritts",
                        help="Protection Hydra config name (default: grnoise_on_libritts)")
    parser.add_argument("--no-hf-download", action="store_true",
                        help="Skip HF download (data already in data/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset/layout and print the run commands without launching them")
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
# Step 1: run_protect.py
# ---------------------------------------------------------------------------

def run_protect(
    protect_config: str,
    dataset_root: Path,
    speaker_id: str,
    device: str,
) -> list[str]:
    return [
        sys.executable, str(REPO_DIR / "run_protect.py"),
        "--config-name", protect_config,
        f"device={device}",
        f"dataset.root_path={dataset_root}",
        "dataset.use_hf_dataset=false",
        f"dataset.speaker_id={speaker_id}",
        "protection.batch_size=1",
    ]


# ---------------------------------------------------------------------------
# Step 2: run_vc_protect.py
# ---------------------------------------------------------------------------

def run_vc(
    dataset_root: Path,
    protected_audio_dir: Path | str,
    speaker_id: str,
    max_samples: int,
    device: str,
    checkpoint_path: str,
) -> list[str]:
    return [
        sys.executable, str(REPO_DIR / "run_vc_protect.py"),
        "--config-name", "ots_vc/clean/libritts/qwen3_tts_ots",
        f"+protected_audio_dir={protected_audio_dir}",
        "run_name=protect_qwen3tts_attack",
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

def print_protect_results(protect_config: str) -> dict:
    results_dir = REPO_DIR / "results" / protect_config
    run_dir = max(results_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    fid = metrics.get("fidelity_evaluation", {})
    protected_wavs = sorted((run_dir / "protected_audio").rglob("*.wav"))

    print("\n" + "=" * 55)
    print("Step 1 — Fidelity (how perceptible is the protection?)")
    print("=" * 55)
    for key in ("avg_mcd", "avg_wer", "avg_speechmos_mos", "avg_dnsmos_ovrl", "avg_sim"):
        if isinstance(fid.get(key), (int, float)):
            print(f"  {key:<28} {fid[key]:.4f}")

    print(f"\n  Protected audio files : {len(protected_wavs)}")
    print(f"  Protected audio dir   : {run_dir / 'protected_audio'}")
    print(f"  Full metrics          : {run_dir / 'metrics.json'}")
    return fid


def print_vc_results() -> dict:
    results_dir = REPO_DIR / "results" / "protect_qwen3tts_attack"
    run_dir = max(results_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    gen = metrics.get("generation_evaluation", {})
    generated_wavs = sorted((run_dir / "generated_audio").rglob("*.wav"))

    print("\n" + "=" * 55)
    print("Step 2 — Generation (how well does the attacker clone?)")
    print("=" * 55)
    for key in ("avg_mcd", "avg_wer", "avg_sim", "avg_speechmos_mos",
                "avg_dnsmos_ovrl", "emotion_match_rate", "rtf"):
        if isinstance(gen.get(key), (int, float)):
            print(f"  {key:<28} {gen[key]:.4f}")

    print(f"\n  Generated {len(generated_wavs)} audio file(s) → {run_dir / 'generated_audio'}")
    print(f"  Full metrics : {run_dir / 'metrics.json'}")
    return gen


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

    print(f"\n[config] REPO_DIR       = {REPO_DIR}")
    print(f"[config] DATASET_ROOT   = {dataset_root}")
    print(f"[config] DEVICE         = {device}")
    print(f"[config] SPEAKER_ID     = {args.speaker_id}")
    print(f"[config] QWEN_CKPT      = {args.qwen_checkpoint_path}")
    print(f"[config] PROTECT_CONFIG = {args.protect_config}")
    print(f"[config] MAX_SAMPLES    = {args.max_samples}\n")

    protect_cmd = run_protect(args.protect_config, dataset_root, args.speaker_id, device)
    print("[protect] Command:", " ".join(protect_cmd))

    if args.dry_run:
        vc_cmd = run_vc(
            dataset_root,
            "<protected_audio_dir_from_step_1>",
            args.speaker_id,
            args.max_samples,
            device,
            args.qwen_checkpoint_path,
        )
        print("[vc] Command:", " ".join(vc_cmd))
        print("\nDry run complete.")
        return

    subprocess.run(protect_cmd, cwd=str(REPO_DIR), check=True)

    protection_results_root = REPO_DIR / "results" / args.protect_config
    protection_run_dir = max(
        (path for path in protection_results_root.glob("*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    protected_audio_dir = protection_run_dir / "protected_audio"
    if not protected_audio_dir.is_dir():
        raise FileNotFoundError(
            f"Protection completed but no protected audio was found at {protected_audio_dir}"
        )

    vc_cmd = run_vc(
        dataset_root,
        protected_audio_dir,
        args.speaker_id,
        args.max_samples,
        device,
        args.qwen_checkpoint_path,
    )
    print("[vc] Command:", " ".join(vc_cmd))
    subprocess.run(vc_cmd, cwd=str(REPO_DIR), check=True)

    # --- Summary ---
    fid = print_protect_results(args.protect_config)
    gen = print_vc_results()

    print("\n" + "=" * 55)
    print("Interpretation")
    print("=" * 55)
    print("Lower MCD / WER = better quality. Higher SIM = more speaker-like.")
    print("A good protection keeps step-1 MOS high while degrading step-2 SIM.")
    print("\nDone.")


if __name__ == "__main__":
    main()
