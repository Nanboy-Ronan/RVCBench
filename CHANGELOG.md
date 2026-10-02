# Changelog

Notable changes to the RVCBench codebase. Paper results are not affected by entries here unless stated.

## Unreleased (2.0.0.dev0, branch `v2`)

### Added

- Installable `rvcbench` package under `src/rvcbench/`, with Hydra configs shipped inside the package.
- Commands `rvcbench run`, `run-protected`, `protect` and `denoise`, next to `doctor`, `smoke`, `status`, `report`, `audit-source`, `compare-check` and `compare-timing`.
- External adapters: `vc.adapter=package.module:ClassName` and the `rvcbench.VoiceCloningAdapter` base class evaluate a model without changing the package.
- Per-sample run records with input and output hashes, seeds, failures, metric coverage and source provenance; resume and retry.
- Direct per-sample backends with request seed validation for Qwen3-TTS, F5-TTS, XTTS, ZipVoice, SparkTTS, CosyVoice, OpenVoice, StyleTTS2, Bark, FireRedTTS2, VoxCPM, IndexTTS and MaskGCT.
- Frozen reproduction subsets, pinned Hub revisions and model asset hashing.
- Lint baseline, pre-commit hooks, CI on Python 3.10 and 3.12 with a wheel build and install check.

### Changed

- The Python package is named `rvcbench`; it was importable as `src`.
- Configs moved from `configs/` to `src/rvcbench/configs/`. Config names passed to `--config-name` are unchanged.
- Malformed inputs and incomplete checkpoints raise errors instead of being skipped or patched.
- Python 3.10 or newer is required.

### Fixed

- FireRedTTS2 output was written at the prompt sample rate (16 kHz) instead of the codec rate (24 kHz).
- Debugger breakpoints removed from the legacy OZSpeech wrapper.

## 1.0 (tag `v1.0`, branch `main`)

The codebase before the architecture refactor, including the NeurIPS 2026 acceptance announcement.
