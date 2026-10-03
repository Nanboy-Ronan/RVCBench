# Changelog

Notable changes to the RVCBench codebase. Paper results are not affected by entries here unless stated.

## Unreleased (2.0.0.dev0, branch `main`)

### Added

- Installable `rvcbench` package under `src/rvcbench/`, with Hydra configs shipped inside the package.
- Commands `rvcbench run`, `run-protected`, `protect` and `denoise`, next to `doctor`, `smoke`, `status`, `report`, `audit-source`, `compare-check` and `compare-timing`.
- `rvcbench prompts` and `rvcbench score`: export a versioned suite's references and texts, then score audio generated anywhere into `submission.json`. The `onboarding-v1` preview suite ships with the package.
- `core-v1` suite: 480 utterances in 22 paired tasks plus 7 post-processing tasks, covering 16 of the paper's 18 robustness evaluations (deepfake detectability is planned for `core-v1.1`). Built by `scripts/build_core_suite.py`; its protected references are published under `Protected_LibriTTS/` in the Hub dataset. Not yet a leaderboard suite.
- `full-v1` suite: the `core-v1` tasks with every pair of their datasets, 12,724 utterances (the protection tasks are not yet included). `core-v1` is a subset of it. Built by `scripts/build_core_suite.py --full`; its per-task files are gzipped.
- Suite features: per-task metrics, clean anchors with relative change, group means, post-processing tasks (MP3, AAC, Opus, telephone band) and a STOI metric.
- `rvcbench setup-scorers` downloads and verifies the metric models (speaker, Whisper, SpeechMOS, emotion).
- `rvcbench prompts` also writes `prompts.tsv` (ZipVoice `--test-list`) and `prompts.lst` (Seed-TTS-eval lists, read by F5-TTS, CosyVoice and others) with absolute reference paths. Every utterance has an `id`, and `rvcbench score` reads outputs saved flat as `<id>.wav`.
- `rvcbench score` scores several output directories at once into one directory per model plus `comparison.md`, `.csv` and `.json`; `rvcbench compare` builds the same comparison from existing `submission.json` files of one suite version.
- Documentation: a shorter README organized around evaluating a model, and new pages for the Core suite, running the built-in models, datasets and codebase versions.
- External adapters: `vc.adapter=package.module:ClassName` and the `rvcbench.VoiceCloningAdapter` base class evaluate a model without changing the package.
- Per-sample run records with input and output hashes, seeds, failures, metric coverage and source provenance; resume and retry.
- Direct per-sample backends with request seed validation for Qwen3-TTS, F5-TTS, XTTS, ZipVoice, SparkTTS, CosyVoice, OpenVoice, StyleTTS2, Bark, FireRedTTS2, VoxCPM, IndexTTS and MaskGCT.
- Frozen reproduction subsets, pinned Hub revisions and model asset hashing.
- Lint baseline, pre-commit hooks, CI on Python 3.10 and 3.12 with a wheel build and install check.

### Changed

- `main` now holds v2. The code released with the paper is preserved on branch `v1` (tag `v1.0`); use it to
  reproduce the paper.
- The Python package is named `rvcbench`; it was importable as `src`.
- Configs moved from `configs/` to `src/rvcbench/configs/`. Config names passed to `--config-name` are unchanged.
- Malformed inputs and incomplete checkpoints raise errors instead of being skipped or patched.
- Python 3.10 or newer is required.
- Run records list `stoi` in `coverage.metric_valid`.
- `rvcbench score` loads each metric model once per call instead of once per task, and fails before writing results when the directory holds none of the suite's files. `--model` is optional and defaults to the directory name.
- Scorer model files live in `$RVCBENCH_ASSET_DIR/<name>`, else `./checkpoints/<name>` when present, else `~/.cache/rvcbench/<name>`; a speaker model fetched from the Hub is copied there.
- Maintainer files moved out of the repository root: `upload_data.py` to `scripts/upload_hf_dataset.py`, `README_HF_DATASET.md` to `docs/hf_dataset_card.md`, and `data/dataset.md` to `docs/dataset_preprocessing.md`.
- The evaluation extras require `pysptk>=1.0`, `speechbrain>=1.0.3` and `transformers` (for the emotion metric).

### Removed

- The root `requirements.txt`, a legacy dependency list that contradicted `pyproject.toml`.

### Fixed

- FireRedTTS2 output was written at the prompt sample rate (16 kHz) instead of the codec rate (24 kHz).
- Debugger breakpoints removed from the legacy OZSpeech wrapper.

## 1.0 (tag `v1.0`, branch `v1`)

The code released with the paper, before the architecture refactor. Use it to reproduce the paper's results.
