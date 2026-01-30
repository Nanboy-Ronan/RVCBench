#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a "compression" dataset using a *precomputed manifest* (top10_longest_by_speaker_manifest.csv),
and ensure consistency across all models by matching the **generated_path basename** from the manifest.

Key requirements
----------------
1) Cross-model consistency (generated audio):
   For every model under:
     <logs_root>/<xx_ots_on_vctk>/<latest_timestamp>/generated_audio/<speaker>/
   pick the generated wav that corresponds to the manifest row.

   However, different models may name files differently. We support:
   - Direct naming:                 <target>.wav
     e.g., p363_155_mic1_cloned.wav
   - StyleTTS2-like naming:         <src>_to_<target>.wav
     e.g., p363_181_mic1_to_p363_155_mic1_cloned.wav
   Matching priority is:
     (a) exact <target basename>
     (b) *_to_<target basename>
     (c) *_to_<target stem>.*
     (d) <target stem>.*

2) VCTK reference extraction (ground truth):
   The top10 set is anchored to the selected generated_path list in the manifest.
   For each manifest row, we copy its corresponding VCTK reference audio into:
     <out_root>/compression/VCTK/<speaker>/

   Priority for resolving the reference file:
     (0) manifest ground_truth_path if exists
     (1) vctk_root/<speaker>/<normalized_target_stem>.*
         - normalized_target_stem is derived from generated_path stem:
           - if it contains "_to_", keep the part after "_to_"
           - strip trailing "_cloned" if present
     (2) vctk_root/<speaker>/<gt_stem>.*
     (3) vctk_root/<speaker>/<gt_basename>

Manifest CSV requirements
-------------------------
- speaker_id
- generated_path
- ground_truth_path
Optional but supported:
- gt_stem
- gt_basename

Output layout
-------------
  <out_root>/compression/<model_dir>/<speaker>/<generated wavs...>
  <out_root>/compression/VCTK/<speaker>/<reference wavs...>

Example
-------
python build_vctk_compression_dataset_from_manifest.py \
  --manifest_csv /path/to/top10_longest_by_speaker_manifest.csv \
  --logs_root /home/xenial/scratch/audiobench_logs \
  --vctk_root /home/xenial/scratch/audiobench_logs/VCTK \
  --out_root /home/xenial/scratch \
  --pattern "*_ots_on_vctk" \
  --speakers p363 p283 p316 p288 \
  --dry_run
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


AUDIO_EXTS = [".wav", ".flac", ".mp3", ".m4a"]


@dataclass(frozen=True)
class Utterance:
    speaker_id: str
    gen_path: str
    gen_basename: str
    gen_stem: str
    ground_truth_path: str
    gt_basename: str
    gt_stem: str


def read_manifest(manifest_csv: Path, speakers: Optional[List[str]] = None) -> List[Utterance]:
    df = pd.read_csv(manifest_csv)
    required = {"speaker_id", "ground_truth_path", "generated_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest_csv missing required columns: {sorted(missing)}")

    df["speaker_id"] = df["speaker_id"].astype(str)
    if speakers:
        df = df[df["speaker_id"].isin(speakers)].copy()

    items: List[Utterance] = []
    for _, r in df.iterrows():
        gt_path = str(r["ground_truth_path"])
        gen_path = str(r["generated_path"])

        gt_p = Path(gt_path)
        gen_p = Path(gen_path)

        gt_stem = str(r["gt_stem"]) if "gt_stem" in df.columns else gt_p.stem
        gt_basename = str(r["gt_basename"]) if "gt_basename" in df.columns else gt_p.name

        items.append(
            Utterance(
                speaker_id=str(r["speaker_id"]),
                gen_path=gen_path,
                gen_basename=gen_p.name,
                gen_stem=gen_p.stem,
                ground_truth_path=gt_path,
                gt_basename=gt_basename,
                gt_stem=gt_stem,
            )
        )

    # stable ordering
    items.sort(key=lambda x: (x.speaker_id, x.gen_basename))
    return items


def find_latest_timestamp_dir(model_dir: Path) -> Path:
    """model_dir: <logs_root>/cosyvoice_ots_on_vctk -> timestamp subdirs like 20251122-145159"""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    candidates = [p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not candidates:
        raise FileNotFoundError(f"No timestamp subdirs under: {model_dir}")
    # Lexicographic max works for YYYYMMDD-HHMMSS; mtime as secondary key
    candidates_sorted = sorted(candidates, key=lambda p: (p.name, p.stat().st_mtime))
    return candidates_sorted[-1]


def _prefer_audio_ext(paths: List[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    for ext in AUDIO_EXTS:
        for p in paths:
            if p.suffix.lower() == ext:
                return p
    return paths[0]


def normalize_target_stem_from_generated(gen_stem: str) -> str:
    """
    Derive the "target" stem for reference lookup from a generated filename stem.

    Examples:
      p363_155_mic1_cloned                     -> p363_155_mic1
      p363_181_mic1_to_p363_155_mic1_cloned    -> p363_155_mic1
    """
    s = gen_stem
    if "_to_" in s:
        s = s.split("_to_", 1)[1]  # keep target part
    if s.endswith("_cloned"):
        s = s[: -len("_cloned")]
    return s


def resolve_generated_file(gen_audio_root: Path, speaker: str, gen_basename: str, gen_stem: str) -> Path:
    """
    Resolve generated wav under:
      .../<timestamp>/generated_audio/<speaker>/

    Supports both:
      - <target>.wav
      - <src>_to_<target>.wav  (StyleTTS2-like)
    """
    spk_dir = gen_audio_root / speaker
    if not spk_dir.exists():
        raise FileNotFoundError(f"Speaker folder not found under generated_audio: {spk_dir}")

    # (a) exact target basename
    p1 = spk_dir / gen_basename
    if p1.exists():
        return p1

    # (b) *_to_<target basename>
    m2 = list(spk_dir.glob("*_to_" + gen_basename))
    if m2:
        return _prefer_audio_ext(m2)

    # (c) *_to_<target stem>.*  (extension differs)
    m3 = list(spk_dir.glob("*_to_" + gen_stem + ".*"))
    if m3:
        return _prefer_audio_ext(m3)

    # (d) <target stem>.*
    m4 = list(spk_dir.glob(gen_stem + ".*"))
    if m4:
        return _prefer_audio_ext(m4)

    raise FileNotFoundError(
        f"Cannot find generated file for {speaker}/{gen_basename} (stem={gen_stem}) under {spk_dir}"
    )


def resolve_vctk_reference(vctk_root: Path, u: Utterance) -> Path:
    """
    Resolve VCTK reference for this utterance.

    Priority:
      (0) manifest ground_truth_path if exists
      (1) vctk_root/<speaker>/<normalized_target_stem>.*   (derived from generated_path stem)
      (2) vctk_root/<speaker>/<gt_stem>.*
      (3) vctk_root/<speaker>/<gt_basename>
    """
    # (0) direct
    ref_src = Path(u.ground_truth_path)
    if ref_src.exists():
        return ref_src

    spk_dir = vctk_root / u.speaker_id
    if not spk_dir.exists():
        raise FileNotFoundError(f"VCTK speaker dir not found: {spk_dir}")

    # (1) by normalized target stem from generated
    target = normalize_target_stem_from_generated(u.gen_stem)
    matches = list(spk_dir.glob(target + ".*"))
    if matches:
        return _prefer_audio_ext(matches)

    # (2) by gt_stem
    matches = list(spk_dir.glob(u.gt_stem + ".*"))
    if matches:
        return _prefer_audio_ext(matches)

    # (3) by gt_basename
    candidate = spk_dir / u.gt_basename
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Cannot locate VCTK reference for speaker={u.speaker_id}, target={target}, gt_stem={u.gt_stem}"
    )


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=Path, required=True,
                    help="top10_longest_by_speaker_manifest.csv generated earlier")
    ap.add_argument("--logs_root", type=Path, default=Path("/home/xenial/scratch/audiobench_logs"),
                    help="Root containing xx_ots_on_vctk/timestamp/generated_audio")
    ap.add_argument("--vctk_root", type=Path, default=Path("/home/xenial/scratch/audiobench_logs/VCTK"),
                    help="Root of VCTK references")
    ap.add_argument("--out_root", type=Path, default=Path("."), help="Where to write compression/")
    ap.add_argument("--pattern", type=str, default="*_ots_on_vctk",
                    help="Glob pattern under logs_root to find model dirs")
    ap.add_argument("--speakers", nargs="*", default=None,
                    help="Optional: subset of speakers, e.g. p363 p283 p316 p288")
    ap.add_argument("--dry_run", action="store_true", help="Only print actions, do not copy")
    args = ap.parse_args()

    selected = read_manifest(args.manifest_csv, args.speakers if args.speakers else None)

    # Save manifest used for this run
    out_manifest = args.out_root / "compression" / "manifest_used.jsonl"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        for u in selected:
            f.write(json.dumps(u.__dict__, ensure_ascii=False) + "\n")

    # 1) Copy VCTK references (anchored to manifest selected generated_path list)
    for u in selected:
        ref_src = resolve_vctk_reference(args.vctk_root, u)
        ref_dst = args.out_root / "compression" / "VCTK" / u.speaker_id / ref_src.name
        if args.dry_run:
            print("[DRY] copy ref", ref_src, "->", ref_dst)
        else:
            copy_file(ref_src, ref_dst)

    # 2) Copy generated audios for every model (latest timestamp)
    model_dirs = sorted([p for p in args.logs_root.glob(args.pattern) if p.is_dir()])
    if not model_dirs:
        raise FileNotFoundError(f"No model dirs matching pattern '{args.pattern}' under {args.logs_root}")

    for md in model_dirs:
        latest = find_latest_timestamp_dir(md)
        gen_audio_root = latest / "generated_audio"
        if not gen_audio_root.exists():
            print("[WARN] missing generated_audio:", gen_audio_root)
            continue

        rel_model = md.name  # e.g., cosyvoice_ots_on_vctk
        for u in selected:
            try:
                gen_src = resolve_generated_file(gen_audio_root, u.speaker_id, u.gen_basename, u.gen_stem)
            except FileNotFoundError as e:
                print("[WARN]", e)
                continue

            gen_dst = args.out_root / "compression" / rel_model / u.speaker_id / gen_src.name
            if args.dry_run:
                print("[DRY] copy gen", gen_src, "->", gen_dst)
            else:
                copy_file(gen_src, gen_dst)

    print("Done.")
    print("Manifest used:", out_manifest)


if __name__ == "__main__":
    main()

    '''
    python src/datasets/build_vctk_compression_dataset_from_manifest.py   --manifest_csv src/datasets/top10_longest_by_speaker_manifest.csv   --logs_root /home/xenial/scratch/audiobench_logs   --vctk_root /home/xenial/scratch/audiobench_logs/VCTK/audios   --out_root /home/xenial/scratch   --pattern "playdiffusion_ots_on_vctk" 
    '''