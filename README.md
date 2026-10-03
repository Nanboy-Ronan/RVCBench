# RVCBench: Benchmarking the Robustness of Voice Cloning
<img src="figs/logo.png" alt="RVCBench logo" width="40" style="vertical-align: middle; margin-right: 8px;">

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026%20Accepted-6842c2.svg)](https://arxiv.org/abs/2602.00443)
[![Website](https://img.shields.io/badge/Website-RVCBench-0d6ea8.svg)](https://nanboy-ronan.github.io/RVCBench/)
[![Paper](https://img.shields.io/badge/arXiv-2602.00443-b31b1b.svg)](https://arxiv.org/abs/2602.00443)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffcc00.svg)](https://huggingface.co/datasets/Nanboy/RVCBench)
[![Demo](https://img.shields.io/badge/HuggingFace-Demo%20Space-ff6f00.svg)](https://huggingface.co/spaces/Nanboy/RVCBench)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![GitHub stars](https://img.shields.io/github/stars/Nanboy-Ronan/RVCBench?style=social)](https://github.com/Nanboy-Ronan/RVCBench/stargazers)

**News:** RVCBench has been accepted to **NeurIPS 2026**!

RVCBench measures how robust voice cloning (zero-shot text-to-speech) is under realistic deployment
conditions: noisy, accented or overlapping reference audio; irregular or scam text; long-form and
multilingual generation; compression; and anti-cloning protection with and without denoising. The
[paper](https://arxiv.org/abs/2602.00443) defines 18 robustness evaluations over 225 speakers and 14,370
utterances and evaluates 18 open-source models.

[Evaluate your model](#evaluate-your-model) · [What it measures](#what-rvcbench-measures) · [Results](#results-from-the-paper) · [v1 and v2](#v1-and-v2) · [Installation](#installation) · [Built-in models](#built-in-models) · [Website](https://nanboy-ronan.github.io/RVCBench/) · [Dataset](https://huggingface.co/datasets/Nanboy/RVCBench) · [Citation](#citation)

> [!NOTE]
> **To reproduce the paper, use the [`v1`](https://github.com/Nanboy-Ronan/RVCBench/tree/v1) branch**, the code
> released with the paper. This branch, `main`, is v2: the same benchmark as an installable package for
> evaluating new models. See [v1 and v2](#v1-and-v2).

![RVCBench main figure](figs/main.png)

## Evaluate your model

Generate speech with your own code, in your own environment, and let RVCBench score it:

```bash
pip install "rvcbench[eval] @ git+https://github.com/Nanboy-Ronan/RVCBench@main"
rvcbench setup-scorers                                    # once: download the metric models
rvcbench prompts --suite core-v1 --output prompts/        # 480 reference clips and texts
# generate every line of prompts/prompts.tsv into outputs/my-model/<id>.wav
rvcbench score --suite core-v1 --generated outputs/my-model --output results/my-model --device cuda
```

`prompts/` lists the utterances in the formats that batch-inference scripts already read, so you
usually do not need to write a loop:

| File | Line format | Read by |
| --- | --- | --- |
| `prompts.tsv` | `id<TAB>reference_text<TAB>reference_audio<TAB>text` | ZipVoice `--test-list` |
| `prompts.lst` | `id\|reference_text\|reference_audio\|text` | Seed-TTS-eval scripts (F5-TTS, CosyVoice, ...) |
| `prompts.jsonl` | JSON, also with language and speaker | your own script |

For example, ZipVoice reads the TSV and writes `<id>.wav` files, which `rvcbench score` reads directly:

```bash
python3 -m zipvoice.bin.infer_zipvoice --model-name zipvoice --test-list prompts/prompts.tsv --res-dir outputs/zipvoice
rvcbench score --suite core-v1 --generated outputs/zipvoice --output results/zipvoice --device cuda
```

To compare several models or checkpoints, pass all their output directories at once. Each metric model
is loaded once for all of them, and `results/comparison.md` puts them side by side:

```bash
rvcbench score --suite core-v1 --generated outputs/zipvoice outputs/zipvoice_distill --output results/ --device cuda
rvcbench compare results/*/submission.json --output results/   # rebuild the table, e.g. after adding a model
```

Pick a suite by how much you want to generate. `core-v1` is a subset of `full-v1`, with the same tasks:

| Suite | Utterances | Use it to |
| --- | ---: | --- |
| `onboarding-v1` | 52 | check the workflow end to end |
| `core-v1` | 480 | evaluate quickly; 16 of the paper's 18 evaluations |
| `full-v1` | 12,724 | evaluate on every pair of the paper's datasets; protection tasks still in `core-v1` only |

Scoring is the slow step. On a shared RTX A6000, `core-v1` took about 35 minutes per model and 4 GB of
GPU memory. `full-v1` scores about 41 times as many files, so plan for about a day on similar hardware.
For `full-v1`, download the dataset once (12.6 GB) and pass it to both commands with `--data-root`; see
[Full suite](docs/core_suite.md#full-suite).

`submission.json` reports SIM, MOS, WER and MCD for every evaluation, and the change of each perturbed
condition against its clean counterpart. The suite pins the dataset revision and the hash of every input,
exports only reference audio and texts, and fails a sample whose file is missing or invalid. A model that
does not support a language leaves those tasks incomplete; the other tasks are still scored.

- [Core suite](docs/core_suite.md): which paper evaluation each task covers, its data and metrics.
- [Evaluate your own model](docs/adding_a_model.md): the prompt and output formats, and the alternative of
  a one-file adapter (`rvcbench.VoiceCloningAdapter`) that lets RVCBench drive your model.
- `core-v1` is not yet a leaderboard suite; baseline results for the paper's models are in progress.

## What RVCBench measures

| Dimension | Paper evaluations | In `core-v1` |
| --- | --- | --- |
| **Input robustness** | Reference-audio shifts across 12 accents, gender and age (RVC-AudioShift); hallucination-style and scam text prompts (RVC-TextShift) | Yes |
| **Generation robustness** | English, Chinese and cross-lingual cloning (RVC-Multilingual); long text and long references (RVC-LongContext); expressive and persuasive speech (RVC-Expression) | Yes; expression scored by emotion consistency |
| **Output robustness** | MP3, AAC, Opus and telephone-band post-processing (RVC-Compression); deepfake detectability (RVC-Detectability) | Compression yes; detectability planned |
| **Perturbation robustness** | Background noise and overlapping speakers (RVC-PassiveNoise); Gaussian noise, SPEC, SafeSpeech, POP and Enkidu protection (RVC-AdvNoise); denoising of protected references (RVC-AntiProtect) | Yes |

Metrics: speaker similarity and verification (SIM, SVA; ECAPA-TDNN), naturalness (MOS; UTMOS via
SpeechMOS), content (WER; Whisper medium), spectral distance (MCD), emotion consistency (EMC) and
intelligibility under compression (STOI). Real-time factor is recorded but not comparable across models.

## Results from the paper

To reproduce these results, use the [`v1`](https://github.com/Nanboy-Ronan/RVCBench/tree/v1) branch.

> [!NOTE]
> **Metric guide** — SIM: speaker cosine similarity ↑ · WER: word error rate ↓ · MOS: SpeechMOS perceptual score ↑ · MCD: mel cepstral distortion ↓ · RTF: real-time factor (< 1 = faster-than-real-time) ↓ · SVA: speaker verification accuracy ↑ · Emo: emotion match rate ↑
>
> **Timing warning:** Historical RTF values have unverified measurement boundaries and execution conditions. They are raw records, cannot rank model speed, and must not inherit current backend scopes. Rank is by SIM.
>
> **Bold** marks the best value per quality column. Reported in the [paper](https://arxiv.org/abs/2602.00443) (arXiv v2) on clean prompts, averaged over each dataset's full speaker set.

### Leaderboard — LibriTTS

| Rank | Model | SIM ↑ | WER ↓ | MOS ↑ | MCD ↓ | RTF (raw, incomparable) | SVA ↑ | Emo ↑ |
|:----:|-------|------:|------:|------:|------:|------:|------:|------:|
| 1 | **Qwen3-TTS** | **0.614** | 0.052 | **4.39** | **5.79** | 2.02 | **0.974** | **0.731** |
| 2 | **IndexTTS** | 0.606 | 0.052 | 4.06 | 6.61 | 2.23 | 0.972 | 0.693 |
| 3 | **CosyVoice 2** | 0.602 | 0.175 | **4.39** | 6.17 | 4.58 | **0.974** | 0.729 |
| 4 | ZipVoice | 0.579 | 0.053 | 4.13 | 7.09 | 1.46 | 0.952 | 0.675 |
| 5 | MaskGCT | 0.570 | 0.088 | 3.93 | 6.91 | 1.36 | 0.939 | 0.682 |
| 6 | GLM-TTS | 0.570 | 0.087 | 4.08 | 6.41 | 1.74 | 0.951 | 0.678 |
| 7 | F5-TTS | 0.559 | 0.116 | 3.99 | 6.96 | 0.61 | 0.937 | 0.676 |
| 8 | Higgs Audio | 0.559 | 0.250 | 4.30 | 6.06 | 1.42 | 0.941 | 0.717 |
| 9 | MGM-Omni | 0.539 | 0.095 | 4.28 | 5.82 | 0.84 | 0.933 | 0.676 |
| 10 | PlayDiffusion | 0.506 | 0.055 | 4.15 | 8.06 | 0.73 | 0.936 | 0.681 |
| 11 | MOSS-TTSD | 0.492 | 0.383 | 4.10 | 7.09 | — | 0.876 | 0.667 |
| 12 | VibeVoice | 0.480 | 0.228 | 3.83 | 6.76 | 1.86 | 0.852 | 0.624 |
| 13 | FishSpeech | 0.472 | 0.166 | 4.37 | 6.47 | 3.61 | 0.907 | 0.682 |
| 14 | XTTS-v2 | 0.454 | 0.073 | 3.81 | 8.62 | 0.62 | 0.908 | 0.639 |
| 15 | SparkTTS | 0.408 | 0.326 | 4.06 | 5.83 | 1.56 | 0.764 | 0.672 |
| 16 | OZSpeech | 0.388 | 0.060 | 3.21 | 6.87 | 8.75 | 0.840 | 0.636 |
| 17 | OpenVoice V2 | 0.244 | 0.075 | 4.30 | 7.06 | 0.08 | 0.474 | 0.601 |
| 18 | StyleTTS 2 | 0.228 | **0.049** | 4.30 | 6.81 | 0.11 | 0.388 | 0.589 |

### Protection Robustness — SIM on LibriTTS

Speaker similarity under each audio protection method. Models sorted by clean SIM. A larger drop from **Clean** indicates more effective protection. **Bold** marks the lowest protected SIM per column (most effectively protected model per method).

| Model | Clean | SafeSpeech | Enkidu | SPEC | Gaussian | POP |
|-------|------:|-----------:|-------:|---------:|---------:|---------:|
| Qwen3-TTS | 0.614 | 0.384 | 0.502 | 0.363 | 0.408 | 0.582 |
| IndexTTS | 0.606 | 0.346 | 0.475 | 0.318 | 0.392 | 0.572 |
| CosyVoice 2 | 0.602 | 0.321 | 0.447 | 0.301 | 0.384 | 0.549 |
| ZipVoice | 0.579 | 0.287 | 0.435 | 0.262 | 0.258 | 0.543 |
| MaskGCT | 0.570 | 0.303 | 0.407 | 0.281 | 0.312 | 0.530 |
| GLM-TTS | 0.570 | 0.330 | 0.445 | 0.311 | 0.388 | 0.532 |
| F5-TTS | 0.559 | 0.207 | 0.431 | 0.176 | 0.137 | 0.520 |
| Higgs Audio | 0.559 | 0.264 | 0.435 | 0.236 | 0.272 | 0.521 |
| MGM-Omni | 0.539 | 0.184 | 0.316 | 0.166 | 0.229 | 0.491 |
| PlayDiffusion | 0.506 | 0.173 | — | 0.149 | 0.162 | 0.466 |
| MOSS-TTSD | 0.492 | 0.242 | 0.335 | 0.216 | 0.247 | 0.453 |
| VibeVoice | 0.480 | 0.272 | 0.367 | 0.253 | 0.280 | 0.442 |
| FishSpeech | 0.472 | 0.238 | 0.334 | 0.212 | 0.235 | 0.439 |
| XTTS-v2 | 0.454 | 0.260 | 0.308 | 0.241 | 0.237 | 0.414 |
| SparkTTS | 0.408 | 0.129 | 0.137 | 0.108 | 0.062 | 0.359 |
| OZSpeech | 0.388 | 0.156 | 0.187 | 0.147 | 0.148 | 0.337 |
| OpenVoice V2 | 0.244 | 0.185 | 0.188 | 0.180 | 0.175 | 0.236 |
| StyleTTS 2 | **0.228** | **0.089** | **0.125** | **0.081** | **0.030** | **0.207** |

<details>
<summary><strong>Cross-Dataset Generalisation — SIM across all 10 datasets (click to expand)</strong></summary>

Speaker similarity (SIM) on clean prompts across all benchmark datasets. — indicates the model was not evaluated on that dataset.

| Model | LibriTTS | VCTK | Multi-spk | Long | AISHELL | French | Bilingual | BG-clean | BG-noise | Hallucin. |
|-------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| Qwen3-TTS | 0.614 | 0.618 | 0.495 | 0.561 | **0.721** | **0.536** | **0.673** | **0.689** | **0.572** | 0.515 |
| IndexTTS | 0.606 | 0.567 | 0.473 | **0.775** | **0.721** | 0.397 | **0.673** | 0.589 | 0.528 | 0.529 |
| CosyVoice 2 | 0.602 | **0.582** | 0.448 | 0.530 | 0.717 | 0.378 | 0.653 | 0.626 | 0.515 | 0.518 |
| ZipVoice | 0.579 | 0.554 | **0.531** | 0.729 | 0.712 | 0.363 | 0.322 | 0.625 | 0.462 | 0.509 |
| MaskGCT | 0.570 | 0.555 | 0.431 | 0.194 | 0.674 | 0.494 | — | 0.610 | 0.487 | 0.499 |
| GLM-TTS | 0.570 | 0.573 | 0.445 | 0.757 | 0.690 | 0.398 | 0.657 | 0.622 | 0.528 | **0.533** |
| F5-TTS | 0.559 | 0.537 | 0.507 | 0.607 | 0.696 | 0.304 | 0.653 | 0.582 | 0.414 | 0.455 |
| Higgs Audio | 0.559 | 0.516 | 0.418 | 0.520 | 0.581 | 0.349 | 0.543 | 0.592 | 0.421 | 0.425 |
| MGM-Omni | 0.539 | 0.447 | 0.370 | 0.442 | 0.713 | 0.227 | 0.630 | 0.523 | 0.332 | 0.396 |
| PlayDiffusion | 0.506 | 0.426 | 0.360 | 0.637 | 0.441 | 0.283 | 0.465 | 0.433 | 0.305 | 0.408 |
| MOSS-TTSD | 0.492 | 0.440 | 0.379 | 0.644 | 0.437 | 0.327 | 0.471 | 0.494 | **0.488** | 0.416 |
| VibeVoice | 0.480 | 0.436 | 0.348 | 0.625 | 0.564 | 0.343 | 0.531 | 0.513 | 0.364 | 0.408 |
| FishSpeech | 0.472 | 0.430 | 0.383 | 0.572 | 0.611 | 0.374 | 0.566 | 0.495 | 0.387 | 0.351 |
| XTTS-v2 | 0.454 | 0.454 | 0.328 | 0.613 | 0.569 | **0.445** | 0.506 | **0.546** | 0.394 | **0.488** |
| SparkTTS | 0.408 | **0.532** | 0.228 | 0.345 | 0.569 | 0.164 | 0.480 | 0.588 | 0.332 | 0.336 |
| OZSpeech | 0.388 | 0.253 | 0.271 | — | — | 0.109 | — | 0.272 | 0.164 | 0.281 |
| OpenVoice V2 | 0.244 | 0.392 | 0.192 | 0.278 | 0.431 | 0.271 | 0.298 | 0.484 | 0.358 | 0.365 |
| StyleTTS 2 | 0.228 | 0.236 | 0.162 | — | — | — | 0.213 | 0.196 | 0.166 | 0.184 |

</details>

## v1 and v2

v1 and v2 are two versions of the code for the same benchmark: the datasets, metrics and paper results are
the same. v1 is the code released with the paper; v2 rebuilds it as an installable package for evaluating
new models.

| | v1 ([`v1`](https://github.com/Nanboy-Ronan/RVCBench/tree/v1) branch) | v2 (this branch, `main`) |
| --- | --- | --- |
| Use it to | **Reproduce the paper** | Evaluate a new model |
| Status | Frozen at the paper release (tag `v1.0`) | Under active development |
| Install | Clone, then `pip install` a list of packages | `pip install` the `rvcbench` package from GitHub |
| Run a built-in model | `python run_vc.py --config-name ...` | `rvcbench run --config-name ...` (same config names) |
| Evaluate your own model | Add an adapter to the codebase | Score audio generated anywhere (`rvcbench prompts`, `rvcbench score`), or a one-file adapter |
| Evaluation data | Full datasets | Full datasets, plus the `core-v1` suite: 480 pinned utterances covering 16 of the paper's 18 evaluations |
| Run output | `metrics.json` per run | Per-sample `run_manifest.json` with input and output hashes, failures and metric coverage |

```bash
git clone --branch v1 https://github.com/Nanboy-Ronan/RVCBench.git RVCBench-v1   # reproduce the paper
```

v1 and v2 name versions of this code. They are unrelated to the paper's arXiv versions and to suite names such
as `core-v1`. [Codebase versions](docs/versions.md) lists what changed in v2.

## Installation

Python 3.10 or newer on Linux. A GPU is recommended for scoring.

```bash
# package only
pip install "rvcbench[eval] @ git+https://github.com/Nanboy-Ronan/RVCBench@main"

# or a source checkout, needed to run the built-in models
git clone https://github.com/Nanboy-Ronan/RVCBench.git
cd RVCBench
python -m pip install -e '.[eval]'
rvcbench doctor                               # check dependencies
rvcbench smoke --output results/smoke         # synthetic CPU pipeline check, no downloads
```

| Extra | Adds |
| --- | --- |
| *(none)* | Runner, run records, suites and the `rvcbench` command |
| `eval` | Metrics: Whisper, SpeechBrain ECAPA and emotion, SpeechMOS, MCD, STOI |
| `qwen3` | Qwen3-TTS runtime for the quickstart |
| `http` | Clients for server-backed models |
| `enkidu` | Enkidu protection |
| `dev` | Tests, lint, pre-commit and build tools |

- **FFmpeg** is needed for the compression tasks and by several models.
- **Hugging Face login** (`hf auth login`) is recommended: anonymous downloads are rate-limited.
- **Scorer models** are stored in `$RVCBENCH_ASSET_DIR`, else `./checkpoints/` when it exists, else
  `~/.cache/rvcbench/`; see [model environments](docs/model_environments.md).
- If installing `[eval]` reports that no `pysptk` version matches, see
  [building the evaluation extras](docs/model_environments.md#building-the-evaluation-extras-from-source).

## Built-in models

RVCBench includes adapters for 32 integrations. Each needs its own environment from [`envs/`](envs/), the
upstream inference code and checkpoints; [Running the built-in models](docs/models.md) covers setup,
per-model notes, server-backed models and the protection pipeline.

```bash
rvcbench run --config-name ots_vc/clean/libritts/qwen3_tts_ots dataset.speaker_id=1089 adversary.max_samples=5
```

`rvcbench run`, `run-protected`, `protect` and `denoise` take Hydra arguments (`--config-name NAME
key=value ...`); in a source checkout `python run_vc.py`, `run_vc_protect.py`, `run_protect.py` and
`run_denoiser.py` do the same. Configs ship inside the package under
[`src/rvcbench/configs/`](src/rvcbench/configs/); add your own with `--config-dir`.

<details>
<summary><strong>Supported models (32)</strong></summary>

"In paper" marks the 18 models of the paper. "v2 subset run" means real generation and core scoring were
recorded on a fixed subset during the v2 refactor; see [validation coverage](docs/validation.md).

| Model | `vc.model` | In paper | v2 subset run |
| --- | --- | :---: | :---: |
| BertVITS2 | `bertvits2` |  | pending |
| Qwen3-TTS | `qwen3_tts` | ✓ | ✓ |
| Qwen3-Omni | `qwen3_omni` |  | pending |
| FireRedTTS-2 | `fireredtts2` |  | ✓ |
| VoxCPM | `voxcpm` |  | ✓ |
| F5-TTS | `f5_tts` | ✓ | ✓ |
| MaskGCT | `maskgct` | ✓ | ✓ |
| OpenVoice V2 | `openvoice` | ✓ | ✓ |
| Coqui XTTS-v2 | `xtts` | ✓ | ✓ |
| IndexTTS | `index_tts` | ✓ | ✓ |
| ZipVoice | `zipvoice` | ✓ | ✓ |
| FishSpeech | `fishspeech` | ✓ | ✓ |
| Fish Audio S2 (in-proc) | `fishspeech_s2` |  | ✓ |
| Fish Audio S2 (server) | `fish_audio_s2` |  | ✓ |
| CosyVoice / 2 | `cosyvoice` | ✓ | ✓ |
| Higgs Audio | `higgs_audio` | ✓ | ✓ |
| Higgs TTS 3 | `higgs_tts_3` |  | pending |
| SparkTTS | `sparktts` | ✓ | ✓ |
| VALL-E | `vall_e` |  | pending |
| StyleTTS 2 | `styletts2` | ✓ | ✓ |
| GLM-TTS | `glm_tts` | ✓ | ✓ |
| GlowTTS | `glowtts` |  | pending |
| Kimi Audio | `kimi_audio` |  | ✓ |
| MGM-Omni | `mgm_omni` | ✓ | ✓ |
| MOSS TTSD | `moss_ttsd` | ✓ | ✓ |
| MOSS-TTS | `moss_tts` |  | ✓ |
| dots.tts | `dots_tts` |  | ✓ |
| ZONOS2 | `zonos2` |  | ✓ |
| PlayDiffusion | `playdiffusion` | ✓ | ✓ |
| Bark Voice Clone | `bark_voice_clone` |  | ✓ |
| OZSpeech | `ozspeech` | ✓ | ✓ |
| VibeVoice | `vibevoice` | ✓ | ✓ |

</details>

## Datasets

The data is on Hugging Face at [Nanboy/RVCBench](https://huggingface.co/datasets/Nanboy/RVCBench) and is
downloaded on demand. [Datasets](docs/datasets.md) lists each folder with its source and paper evaluation,
the manifest format, and how to use a local copy.

## Run records and reproducibility

Every run writes `run_manifest.json`: per-sample status, input and output hashes, seeds, the resolved
config, environment and source provenance, and metric coverage over all requested samples. Missing outputs
or metrics make a run `partial`.

```bash
rvcbench status results/<run_name>/<timestamp>      # coverage and errors
rvcbench report results/<run_name>/<timestamp> --output report.json   # refuses incomplete runs
rvcbench compare-check <run_a> <run_b>              # are two runs comparable?
```

Retries (`+vc.retries=1`), resuming into a new run (`+vc.resume_from=...`), scoring saved audio in a
separate environment, frozen subsets and timing are covered in the [run guide](docs/run_protocol.md).
[Validation coverage](docs/validation.md) states what has been verified for each model; subset runs do not
establish reproduction of the paper's tables.

## Repository layout

```text
src/rvcbench/
├── benchmark/     runner, run records, suites, provenance and the rvcbench command
├── suites/        versioned suites (onboarding-v1, core-v1, full-v1)
├── configs/       Hydra configs: datasets, models, protection, denoising
├── adversary/     voice cloning adapters
├── models/        model wrappers
├── protection/    protection methods
├── evaluation/    metrics
└── datasets/      dataset loading and manifests
envs/              one environment file per model
scripts/           quickstarts, model workers and maintenance tools
docs/              guides
```

## Contributing

Contributions are welcome, especially new models, protection methods, datasets and metrics. Development
setup, checks and rules are in [CONTRIBUTING.md](CONTRIBUTING.md); changes are listed in
[CHANGELOG.md](CHANGELOG.md). Open an issue or pull request on GitHub, or contact
**ruinanjin@alumni.ubc.ca**.

## Citation

```bibtex
@inproceedings{jin2026rvcbench,
  title     = {RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models},
  author    = {Jin, Ruinan and Liao, Xinting and Yu, Hanlin and Pandya, Deval and Li, Xiaoxiao},
  booktitle = {Advances in Neural Information Processing Systems},
  url       = {https://arxiv.org/abs/2602.00443},
  year      = {2026}
}
```

## License

[CC0-1.0](LICENSE). Model checkpoints, upstream code and source corpora keep their own licenses.
