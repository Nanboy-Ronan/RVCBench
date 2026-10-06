# Changelog

Notable changes to the RVCBench codebase. Paper results are not affected by entries here unless stated.

## Unreleased

- Cap PyTorch below 2.10 in the core dependencies, matching the evaluation extra, so that installing
  `rvcbench` and then `rvcbench[eval]` no longer downgrades PyTorch.
- The `qwen3` and `enkidu` extras use version ranges instead of exact pins and no longer replace an installed
  PyTorch build. `envs/qwen3-tts.yml` and the new `envs/enkidu.yml` keep the exact versions used for the
  published runs. `rvcbench[enkidu,eval]` can now be installed together.
- Stop shipping the 6 MB CMU dictionary pickle; it is built from `cmudict.rep` in the user cache when needed
  instead of being written into the package directory.
- Support Python 3.13 (tested in CI; the scoring regression also passes on 3.13) and mark the package as
  Production/Stable. Scoring behavior is unchanged.

## 2.2.0 (2026-10-05)

- Add `rvcbench tasks --suite ...` with paper scenario mappings, metrics and dependencies, plus JSON output.
- Add `--tasks` to prompt export and single/multi-model scoring; automatically include clean anchors
  and source tasks for derived evaluations. Unselected tasks do not download or score data.
- Record selection provenance and bind resume/comparison to the effective task set. Whole-suite
  commands retain their existing behavior and identity.
- Document scenario-specific commands and correct the core/full coverage distinction.

## 2.1.1 (2026-10-05)

- Publish an English documentation website with searchable guides, Python API and CLI references.
- Align package discovery with automatic speech metrics and dataset-backed voice cloning evaluation.
- Add page-specific search summaries, social previews, linked software/dataset metadata, complete sitemaps
  and a plain-text documentation export. Check these artifacts before website deployment.
- Point PyPI documentation links to the documentation website and remove the Chinese quickstart from
  current documentation. Scoring behavior is unchanged from 2.1.0.

## 2.1.0 (2026-10-05)

- Present RVCBench as a general-purpose voice cloning evaluation package, with two documented pip workflows:
  automatic metrics for your audio, and automatic scoring with the packaged datasets.
- Add English and Chinese step-by-step quickstarts and a user documentation index covering installation,
  audio inputs, inference, result inspection, comparison and recovery.
- Pin and hash-verify SpeechMOS source/weights, ECAPA speaker assets and the emotion base initializer;
  verify cached Whisper weights during setup. Check-only setup does not download missing assets.
- Require matching per-task scoring fingerprints for comparison. `--allow-incompatible` produces an
  explicitly unranked inspection; it cannot create leaderboard results.
- Add `rvcbench score --resume` for one or several models, with input validation, per-metric cache reuse,
  failed-sample retry, a durable request journal and protection against concurrent writes per model.
- Preserve caller RNG states and cuDNN flags when using the metrics API, including on failures, without
  initializing unrelated CUDA devices. Add `Evaluator("all")` and a separate same-text `target` for MCD/STOI.
- Require SpeechBrain 1.1.1 or newer in the evaluation extra for its offline fetching interface.
- Gate publishing on the CPU checks and a real-speech scoring regression through the installed wheel's
  API and dataset workflows. Run the same scoring regression weekly.

Scoring fingerprints change with these fixes. Existing reports remain readable; rescore models in the
same environment to compare them. Resume applies to runs with the new `score_request.json` journal.

## 2.0.0 (2026-10-03)

First release on PyPI: `pip install rvcbench`.

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
- `rvcbench.metrics`: the benchmark's metrics (SIM, SVA, WER, SpeechMOS, MCD, STOI, emotion) as a Python API, with one-line functions and an `Evaluator` that loads each metric model once.
- `rvcbench --version`, and a release workflow that publishes to PyPI through trusted publishing when a version tag is pushed.
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
- The evaluation extras use version ranges instead of exact pins: scores were checked to be identical up to float rounding with torch 2.6 and 2.9 and numpy 1.26 and 2.2. They require `pysptk>=1.0`, `speechbrain>=1.0.3`, `transformers` (for the emotion metric) and `setuptools<81` (pysptk imports `pkg_resources`).
- Scorers read audio with soundfile instead of `torchaudio.load`, which needs `torchcodec` from torchaudio 2.9; the samples, and therefore the scores, are unchanged.
- README links and images are absolute, so the page also renders on PyPI.

### Removed

- The root `requirements.txt`, a legacy dependency list that contradicted `pyproject.toml`.

### Fixed

- FireRedTTS2 output was written at the prompt sample rate (16 kHz) instead of the codec rate (24 kHz).
- Debugger breakpoints removed from the legacy OZSpeech wrapper.

## 1.0 (tag `v1.0`, branch `v1`)

The code released with the paper, before the architecture refactor. Use it to reproduce the paper's results.
