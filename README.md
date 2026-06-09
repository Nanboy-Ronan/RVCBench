# RVCBench — Voice Cloning Benchmark
<img src="figs/logo.png" alt="RVCBench logo" width="40" style="vertical-align: middle; margin-right: 8px;">

[![Paper](https://img.shields.io/badge/arXiv-2602.00443-b31b1b.svg)](https://arxiv.org/abs/2602.00443)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffcc00.svg)](https://huggingface.co/datasets/Nanboy/RVCBench)
[![Demo](https://img.shields.io/badge/HuggingFace-Demo%20Space-ff6f00.svg)](https://huggingface.co/spaces/Nanboy/RVCBench)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#requirements)

**RVCBench** is the first large-scale benchmark for **voice cloning robustness**, **voice clone evaluation**, **speaker privacy**, and **audio deepfake protection** — covering **26 TTS/VC models**, **10 datasets**, and **5 audio protection methods**.

RVCBench provides a unified, reproducible pipeline covering the full attack-defense cycle: source-audio protection, zero-shot or fine-tuning voice cloning, optional denoising, and evaluation of speaker similarity, intelligibility, perceptual quality, and runtime.

At a glance, this release supports **26 VC/TTS adversary models**, **5 audio protection methods**, **10 public benchmark dataset configurations**, and both fidelity and generation-quality metrics.

**Canonical resources:** [paper](https://arxiv.org/abs/2602.00443) · [Hugging Face dataset](https://huggingface.co/datasets/Nanboy/RVCBench) · [interactive demo](https://huggingface.co/spaces/Nanboy/RVCBench) · [quickstart notebooks](notebooks/) · [model environments](docs/model_environments.md) · [citation](#citation)

**Contents:** [Results](#benchmark-results) · [Models](#supported-models) · [Getting Started](#getting-started) · [Quickstart](#quickstart-path) · [Full Pipeline](#full-benchmark-path) · [Data & Checkpoints](#data--checkpoints) · [Citation](#citation)

![RVCBench main figure](figs/main.png)

---

## Overview

Voice cloning technology poses a growing threat to speaker privacy. Audio protection methods—such as adversarial perturbations—aim to make cloned speech recognizably degraded, but their effectiveness varies widely across different VC models and datasets. RVCBench closes this gap by offering a single framework that:

- applies protection algorithms to source audio (SafeSpeech, Enkidu, EM, Gaussian noise, spectral perturbations),
- runs a broad suite of zero-shot and fine-tuning VC adversaries on clean or protected inputs,
- optionally denoises protected audio and re-evaluates,
- computes standardised fidelity and generation-quality metrics with bootstrap confidence intervals.

RVCBench is intended for researchers and engineers working on voice cloning benchmarks, audio deepfake robustness, speaker verification resilience, anti-spoofing, synthetic speech detection, TTS safety, and privacy-preserving speech generation.

---

## Benchmark Results

> [!NOTE]
> **Metric guide** — SIM: speaker cosine similarity ↑ · WER: word error rate ↓ · MOS: SpeechMOS perceptual score ↑ · MCD: mel cepstral distortion ↓ · RTF: real-time factor (< 1 = faster-than-real-time) ↓ · SVA: speaker verification accuracy ↑ · Emo: emotion match rate ↑
>
> **Bold** marks the best value per column. All results on clean (unprotected) prompts, averaged over the full speaker set for each dataset.

### Leaderboard — LibriTTS

| Rank | Model | SIM ↑ | WER ↓ | MOS ↑ | MCD ↓ | RTF ↓ | SVA ↑ | Emo ↑ |
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
| 17 | OpenVoice V2 | 0.244 | 0.075 | 4.30 | 7.06 | **0.08** | 0.474 | 0.601 |
| 18 | StyleTTS 2 | 0.228 | **0.049** | 4.30 | 6.81 | 0.11 | 0.388 | 0.589 |

### Protection Robustness — SIM on LibriTTS

Speaker similarity under each audio protection method. Models sorted by clean SIM. A larger drop from **Clean** indicates more effective protection. **Bold** marks the lowest protected SIM per column (most effectively protected model per method).

| Model | Clean | SafeSpeech | Enkidu | Spectral | GR-Noise | AntiFake |
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

---

## Supported Models

### Voice Cloning Adversaries (Zero-Shot OTS)

RVCBench currently includes wrappers or configs for **26 VC/TTS adversary models**:

| Model | Key |
|---|---|
| BertVITS2 | `bert` |
| Qwen3-TTS | `qwen3_tts` |
| Qwen3-Omni | `qwen3_omni` |
| FireRedTTS-2 | `fireredtts2` |
| VoxCPM | `voxcpm` |
| F5-TTS | `f5_tts` |
| MaskGCT | `maskgct` |
| OpenVoice V2 | `openvoice` |
| Coqui XTTS-v2 | `xtts` |
| IndexTTS | `index_tts` |
| ZipVoice | `zipvoice` |
| FishSpeech | `fishspeech` |
| CosyVoice / CosyVoice 2 | `cosyvoice` |
| Higgs Audio | `higgs_audio` |
| SparkTTS | `sparktts` |
| VALL-E | `vall_e` |
| StyleTTS 2 | `styletts2` |
| GLM-TTS | `glm_tts` |
| GlowTTS | `glowtts` |
| Kimi Audio | `kimi_audio` |
| MGM-Omni | `mgm_omni` |
| MOSS TTSD | `moss_ttsd` |
| PlayDiffusion | `playdiffusion` |
| Bark Voice Clone | `bark_voice_clone` |
| OZSpeech | `ozspeech` |
| VibeVoice | `vibevoice` |

### Protection Methods

RVCBench currently supports **5 audio protection methods**:

| Method | Description |
|---|---|
| SafeSpeech | Adversarial perturbation optimised against a surrogate VC model |
| Enkidu | Perceptual-loss adversarial perturbation |
| EM | Expectation–Maximisation perturbation |
| GRNoise | Gaussian random noise (no surrogate model required) |
| AntiFake | AntiFake adversarial perturbation |

## Supported Datasets

The public Hugging Face dataset release exposes **10 benchmark dataset configurations**:

| Dataset config | Typical use |
|---|---|
| `Libritts` | English zero-shot VC/TTS benchmark prompts |
| `VCTK` | Multi-speaker English voice cloning |
| `Multispeaker_libri` | Multi-speaker LibriSpeech-style evaluation |
| `Long_context` | Longer-context voice cloning prompts |
| `AISHELL1_dev` | Mandarin speech evaluation |
| `CommonVoiceFR_dev` | French speech evaluation |
| `Bilingual_uedin` | Bilingual speech evaluation |
| `Background_noise` | Noisy prompt robustness |
| `robotcall` | Robocall-style speech robustness |
| `vctk_text_robust` | Text robustness on VCTK-style prompts |

## Evaluated Metrics

RVCBench reports fidelity metrics for protection/denoising runs and generation-quality metrics for VC/TTS runs.

| Stage | Metrics |
|---|---|
| Protection and denoising fidelity | SNR, STOI, MCD, WER, SpeechMOS, DNSMOS, speaker similarity |
| Voice cloning / TTS generation | MCD, WER, speaker similarity, SpeechMOS, DNSMOS, emotion match rate, real-time factor (RTF) |

---

## Getting Started

### Requirements

- Python 3.9+
- PyTorch ≥ 2.0 with CUDA
- `hydra-core`, `omegaconf`, `pandas`, `pyarrow`, `soundfile`, `librosa`
- Model-specific packages. Many supported VC/TTS models need mutually
  incompatible dependency stacks, so launch each model from its matching
  environment in [`envs/`](envs/); see
  [docs/model_environments.md](docs/model_environments.md).

### Installation

```bash
git clone https://github.com/Nanboy-Ronan/RVCBench.git
cd RVCBench
pip install hydra-core omegaconf pandas pyarrow soundfile librosa \
            jiwer openai-whisper pymcd huggingface_hub
```

For model-specific runs, prefer the checked-in Conda environment files:

```bash
cd envs
conda env create -f qwen3-tts.yml
conda activate qwen3
cd ..
```

Model checkpoints and third-party inference code are **not bundled**. Download instructions are provided in the [Data & Checkpoints](#data--checkpoints) section.

---

## Quickstart Path

> [!TIP]
> If you only want to evaluate a single model, skip directly to [Running Specific VC Models](#running-specific-vc-models) — you do not need to download every checkpoint bundle.

Use this path when you want to run a small, end-to-end example with automatic data download. All outputs — data, generated audio, and metrics — are written **inside the repository directory**. The Qwen3-TTS examples assume the `qwen3` environment is active and the Qwen checkpoint is available from Hugging Face or a local path.

### Fastest quickstart

Simplest voice-cloning run:

```bash
conda activate qwen3
python scripts/run_qwen3tts_quickstart.py --max-samples 5
```

Protection plus voice-cloning run:

```bash
conda activate qwen3
python scripts/run_protect_qwen3tts_quickstart.py --max-samples 5
```

Both commands download the selected LibriTTS speaker data from `Nanboy/RVCBench` unless `--no-hf-download` is passed.

### Quickstart examples

Three end-to-end examples are provided as both Jupyter notebooks and standalone Python scripts.

| Example | Notebook | Script |
|---|---|---|
| FishSpeech zero-shot VC on VCTK | `notebooks/rvcbench_fishspeech_quickstart.ipynb` | `scripts/run_fishspeech_quickstart.py` |
| Qwen3-TTS zero-shot VC on LibriTTS | `notebooks/rvcbench_qwen3tts_quickstart.ipynb` | `scripts/run_qwen3tts_quickstart.py` |
| Protection (GRNoise / SafeSpeech) + Qwen3-TTS | `notebooks/rvcbench_safespeech_qwen3tts_quickstart.ipynb` | `scripts/run_protect_qwen3tts_quickstart.py` |

Model setup for the quickstarts, including gated downloads and exact commands:
[docs/quickstart_model_setup.md](docs/quickstart_model_setup.md)

### Additional script commands

```bash
conda activate qwen3

# Zero-shot voice cloning only
python scripts/run_qwen3tts_quickstart.py                    # auto-downloads data from HF
python scripts/run_qwen3tts_quickstart.py \
    --no-hf-download --speaker-id 1089 --max-samples 10
python scripts/run_qwen3tts_quickstart.py \
    --qwen-checkpoint-path checkpoints/Qwen3-TTS-12Hz-1.7B-Base

# Protection + voice clone attack (two-step pipeline)
python scripts/run_protect_qwen3tts_quickstart.py            # Gaussian noise protection (no checkpoints needed)
python scripts/run_protect_qwen3tts_quickstart.py \
    --protect-config safespeech_on_libritts                    # SafeSpeech (requires surrogate-model checkpoints)

# FishSpeech zero-shot voice cloning
python scripts/run_fishspeech_quickstart.py

# Smoke-test the notebook companion scripts against local HF-formatted data only
python scripts/validate_quickstarts.py
```

Pass `--help` to either script for the full list of options.

---

## Full Benchmark Path

Use this path when you want to run the configurable benchmark entry points directly. All full benchmark entry points use [Hydra](https://hydra.cc) for configuration, and config values can be overridden on the command line.

### Experiment flow

```
source audio → [protection] → [denoising] → voice cloning → evaluation
```

### 1. Apply protection and evaluate fidelity

```bash
python run_protect.py --config-name safespeech_on_libritts
```

### 2. Run zero-shot voice cloning on clean prompts

```bash
# Using a specific model and dataset
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots

# With command-line overrides
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
    adversary.max_samples=50 dataset.speaker_id=1089
```

### 3. Run voice cloning on protected prompts

```bash
python run_vc_protect.py --config-name ots_vc/protection/safespeech/ozspeech_ots \
    protected_audio_dir=results/safespeech_on_libritts/<timestamp>/protected_audio
```

### 4. Optionally denoise protected audio and re-evaluate

```bash
python run_denoiser.py --config-name denoise/denoiser_dns64_on_protected_libritts_spec
```

---

## Running Specific VC Models

The zero-shot VC configs are under `configs/ots_vc/clean/`. Most older examples use `configs/ots_vc/clean/libritts/`, while some newer model integrations currently live under `configs/ots_vc/clean/libratts/`. Despite the directory name difference, the added configs here still default to the `libritts` dataset internally.

Before launching a model, activate the matching environment from
[`envs/`](envs/). See [docs/model_environments.md](docs/model_environments.md) for the full map.

### If You Only Want One Model

You do not need to download every checkpoint bundle. For a single model, the workflow is:

1. Pick the Hydra config for that model under `configs/ots_vc/clean/...`.
2. Create and activate that model's environment from `envs/`.
3. Download or install only that model's runtime and checkpoints.
4. Point any local paths with Hydra overrides such as `adversary.code_path=...` or `adversary.checkpoint_path=...`.
5. Run `run_vc.py` against the dataset/config you want.

Generic pattern:

```bash
python run_vc.py --config-name <model_config> \
  dataset.speaker_id=<speaker_id> \
  adversary.max_samples=<n>
```

If the model needs a local repo checkout or checkpoint directory:

```bash
python run_vc.py --config-name <model_config> \
  dataset.speaker_id=<speaker_id> \
  adversary.code_path=/path/to/model_repo \
  adversary.checkpoint_path=/path/to/checkpoint_or_hf_id
```

Concrete examples:

```bash
# Qwen3-TTS only
python -m pip install -U qwen-tts
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir checkpoints/Qwen3-TTS-12Hz-1.7B-Base
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  dataset.speaker_id=1089 \
  adversary.checkpoint_path=checkpoints/Qwen3-TTS-12Hz-1.7B-Base

# FishSpeech only
git clone https://github.com/fishaudio/fish-speech.git checkpoints/fish_speech
huggingface-cli download fishaudio/s1-mini \
  --local-dir checkpoints/fish_speech/openaudio-s1-mini
python run_vc.py --config-name ots_vc/clean/vctk/fishspeech_ots \
  dataset.speaker_id=p226 \
  adversary.code_path=checkpoints/fish_speech \
  adversary.llama_checkpoint_path=checkpoints/fish_speech/openaudio-s1-mini \
  adversary.decoder_checkpoint_path=checkpoints/fish_speech/openaudio-s1-mini/codec.pth
```

For the full command list for the quickstart models, see
[docs/quickstart_model_setup.md](docs/quickstart_model_setup.md).

### Additional Examples

```bash
# FireRedTTS-2
python run_vc.py --config-name ots_vc/clean/libratts/fireredtts2_ots

# VoxCPM
python run_vc.py --config-name ots_vc/clean/libratts/voxcpm_ots
```

### Model-specific setup notes

- `Qwen3-TTS` can use either the Hugging Face model ID `Qwen/Qwen3-TTS-12Hz-1.7B-Base` directly or a local directory passed via `adversary.checkpoint_path=...`. It also needs the `qwen-tts` Python package.
- `FishSpeech` needs both a local checkout of `fishaudio/fish-speech` and the `fishaudio/s1-mini` checkpoint directory. Pass them with `adversary.code_path=...`, `adversary.llama_checkpoint_path=...`, and `adversary.decoder_checkpoint_path=...`.
- `FireRedTTS-2` expects a local upstream checkout at `checkpoints/FireRedTTS2` and pretrained weights under `checkpoints/FireRedTTS2/pretrained_models/FireRedTTS2` by default.
- `VoxCPM` defaults to the Hugging Face model ID `openbmb/VoxCPM2`. If you want to force offline/local loading, override `adversary.local_files_only=true` and optionally set `adversary.cache_dir=/path/to/cache`.
- All model-specific paths and generation knobs can be overridden at launch time with Hydra, for example: `adversary.code_path=/path/to/model_repo` or `adversary.max_samples=20`.

### Example overrides

```bash
python run_vc.py --config-name ots_vc/clean/libratts/fireredtts2_ots \
    adversary.max_samples=20 \
    dataset.speaker_id=1089

python run_vc.py --config-name ots_vc/clean/libratts/voxcpm_ots \
    adversary.local_files_only=true \
    adversary.cache_dir=/path/to/hf-cache
```

---

## Data & Checkpoints

### Benchmark dataset

The benchmark dataset is publicly available on Hugging Face. The data loader fetches it automatically when `use_hf_dataset: true` (the default in all dataset configs).

👉 **[Nanboy/RVCBench on Hugging Face](https://huggingface.co/datasets/Nanboy/RVCBench)**

A static snapshot is also available for offline use:

👉 **[Download via Google Drive](https://drive.google.com/file/d/1ZDOMorDGV8i5oVNtA5BaJLbFj2dVo5AU/view?usp=drive_link)**

### Model checkpoints

Each VC adversary requires its own inference code and pretrained weights. Clone the relevant repository into `checkpoints/` and verify the paths in the corresponding config under `configs/`.

A bundled archive with all supported model code and checkpoints is available here:

👉 **[Download pretrained checkpoints](https://arbutus.cloud.alliancecan.ca/api/swift/containers/rjin/object/AudioBench/checkpoint.zip)**

If you only need a single model, cloning just that repository is the fastest option.

---

## Dataset Format

Each dataset follows a canonical layout:

```text
data/<dataset>/
├── audios/
│   └── <speaker_id>/*.wav
├── filelists/          # legacy per-speaker JSON manifests (kept for compatibility)
└── metadata.parquet    # canonical manifest used by all loaders
```

`metadata.parquet` stores one row per benchmark pair with the following columns:

| Column | Description |
|---|---|
| `speaker_id` | Target speaker identifier |
| `prompt_file_name` | Path to the prompt (reference) audio |
| `prompt_text`, `prompt_language` | Transcript and language of the prompt |
| `target_file_name` | Path to the ground-truth target audio |
| `target_text`, `target_language` | Transcript and language of the target |
| `pair_id`, `dataset_name`, `split` | Provenance fields |

Training-oriented phoneme and alignment annotations (`prompt_phonemes`, `prompt_tone`, `prompt_word2ph`, and their `target_*` counterparts) are preserved when available. Dataset-specific fields (e.g., `spam_type` in `robotcall`) are carried as additional columns.

To rebuild canonical manifests from legacy per-speaker JSON files:

```bash
python src/datasets/build_canonical_manifests.py --force
```

Dataset selection and `speaker_id` filtering continue to work through `configs/dataset/` as before.

---

## Repository Structure

```text
RVCBench/
├── run_vc.py                  # voice cloning on clean prompts
├── run_protect.py             # apply protection + fidelity evaluation
├── run_vc_protect.py          # voice cloning on protected prompts
├── run_denoiser.py            # denoise protected audio + re-evaluate
├── configs/
│   ├── dataset/               # dataset configs (root_path, sampling_rate, …)
│   ├── model/                 # surrogate model configs for protection
│   ├── ots_vc/                # zero-shot VC configs (clean / protected)
│   └── denoise/               # denoiser configs
├── src/
│   ├── adversary/             # VC adversary wrappers
│   ├── protection/            # protection algorithm implementations
│   ├── datasets/              # dataset loaders and manifest utilities
│   ├── evaluation/            # fidelity and generation-quality metrics
│   ├── models/                # model generator wrappers
│   └── workflows/             # end-to-end pipeline orchestration
├── notebooks/                 # quickstart notebooks and runnable example scripts
└── data/                      # local dataset folders (populated at runtime)
```

---

## Outputs

Each run writes a timestamped directory under `results/`:

```text
results/<run_name>/<timestamp>/
├── generated_audio/            # cloned audio files (VC runs)
├── protected_audio/            # perturbed audio files (protection runs)
├── perturbed_noise/            # saved perturbation tensors
├── generation_sample_metrics.csv
├── fidelity_sample_metrics.csv
└── metrics.json                # aggregated metrics with bootstrap CIs
```

`metrics.json` contains fidelity metrics (SNR, STOI, MCD, WER, MOS, speaker similarity) for protection and denoising runs, and generation-quality metrics (MCD, WER, speaker similarity, MOS, emotion match rate, RTF) for VC runs.

---

## Configuration

All configs use [Hydra](https://hydra.cc). Any field can be overridden from the command line:

```bash
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
    device=cuda:1 \
    adversary.max_samples=100 \
    dataset.speaker_id=1089 \
    dataset.use_hf_dataset=false \
    dataset.root_path=/path/to/local/data
```

Key config locations:

| Path | Controls |
|---|---|
| `configs/dataset/` | Dataset root, sampling rate, speaker selection |
| `configs/ots_vc/` | VC model, generation hyperparameters, evaluation settings |
| `configs/model/` | Surrogate model used during protection |
| `configs/denoise/` | Denoiser model and paths |

---

## Contributing

Contributions are welcome. Areas of particular interest include:

- New protection or defense methods
- New VC adversary wrappers
- Dataset adapters and additional evaluation metrics
- Reproducibility and documentation improvements

Please open an issue or pull request on GitHub. For questions, contact:

**ruinanjin@alumni.ubc.ca**

---

## Citation

If you use RVCBench in your research, please cite:

```bibtex
@article{liao2026rvcbench,
  title   = {RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models},
  author  = {Liao, Xinting and Jin, Ruinan and Yu, Hanlin and Pandya, Deval and Li, Xiaoxiao},
  journal = {arXiv preprint arXiv:2602.00443},
  year    = {2026}
}
```

## License

See `LICENSE`.
