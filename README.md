<div align="center">

<img src="https://raw.githubusercontent.com/Nanboy-Ronan/RVCBench/main/figs/logo.png" alt="RVCBench logo" width="110">

# RVCBench

### Benchmarking the Robustness of Voice Cloning

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-6842c2.svg)](https://arxiv.org/abs/2602.00443)
[![arXiv](https://img.shields.io/badge/arXiv-2602.00443-b31b1b.svg)](https://arxiv.org/abs/2602.00443)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffcc00.svg)](https://huggingface.co/datasets/Nanboy/RVCBench)
[![Website](https://img.shields.io/badge/Website-RVCBench-0d6ea8.svg)](https://nanboy-ronan.github.io/RVCBench/)
[![CI](https://github.com/Nanboy-Ronan/RVCBench/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Nanboy-Ronan/RVCBench/actions/workflows/ci.yml)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-lightgrey.svg)](https://github.com/Nanboy-Ronan/RVCBench/blob/main/LICENSE)

[**Paper**](https://arxiv.org/abs/2602.00443) · [**Website**](https://nanboy-ronan.github.io/RVCBench/) · [**Dataset**](https://huggingface.co/datasets/Nanboy/RVCBench) · [**Demo**](https://huggingface.co/spaces/Nanboy/RVCBench) · [**Evaluate your model**](#evaluate-your-model) · [**Reproduce the paper**](#reproduce-the-paper)

</div>

> **News**
>
> - **2026-10** · Score your own model straight from ZipVoice- or Seed-TTS-style batch lists, and compare several models in one table.
> - **2026-09** · RVCBench is accepted to **NeurIPS 2026**.

Voice cloning models sound convincing in clean demos. RVCBench measures how they hold up in deployment:
noisy, accented or overlapping reference audio; irregular or scam text; long-form and multilingual speech;
compression; and anti-cloning protection, with and without denoising. The
[paper](https://arxiv.org/abs/2602.00443) defines **18 robustness evaluations** over **14,370 utterances**
from **204 speakers** and evaluates **18 open-source models**.

<p align="center"><img src="https://raw.githubusercontent.com/Nanboy-Ronan/RVCBench/main/figs/main.png" alt="RVCBench overview" width="100%"></p>

## Evaluate your model

Generate speech with your own code, then let RVCBench score it. With ZipVoice, for example:

```bash
pip install "rvcbench[eval] @ git+https://github.com/Nanboy-Ronan/RVCBench@main"
rvcbench setup-scorers                                # once: download the metric models
rvcbench prompts --suite core-v1 --output prompts/    # reference clips and texts to synthesize
python3 -m zipvoice.bin.infer_zipvoice --model-name zipvoice \
  --test-list prompts/prompts.tsv --res-dir outputs/zipvoice
rvcbench score --suite core-v1 --generated outputs/zipvoice --output results/zipvoice --device cuda
```

- **Works with batch scripts.** `prompts/` has `prompts.tsv` (ZipVoice), `prompts.lst` (Seed-TTS-eval
  scripts such as F5-TTS and CosyVoice) and `prompts.jsonl`. Save each output as `<id>.wav`.
- **Compare models in one run.** Pass several output folders to `rvcbench score` and get `comparison.md`.
- **Pinned inputs.** Every input is checked by hash against a fixed dataset revision, and target recordings
  are never exported.

| Suite | Utterances | Use it for |
| --- | ---: | --- |
| `onboarding-v1` | 52 | checking the workflow end to end |
| `core-v1` | 480 | a quick evaluation covering 16 of the 18 evaluations (about 35 min of scoring) |
| `full-v1` | 12,724 | every pair of the paper's datasets, without the protection tasks for now (about a day of scoring) |

The same metrics are available in your own code, with the same models and definitions:

```python
from rvcbench import metrics

with metrics.Evaluator(["sim", "wer", "speechmos"], device="cuda") as evaluator:
    evaluator.score("generated.wav", reference="reference.wav", text="Hello there.", language="en")
```

See the [metrics API](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/metrics.md) for every metric and the one-line functions.

Requires Linux, Python 3.10+ and FFmpeg; a GPU is recommended for scoring. More in
[Evaluate your own model](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/adding_a_model.md) and the [Core suite](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/core_suite.md) guide.

## What it measures

| Dimension | Paper evaluations |
| --- | --- |
| **Input** | Reference audio across 12 accents, gender and age; hallucination-style and scam text prompts |
| **Generation** | English, Chinese and cross-lingual cloning; long text and long references; expressive and persuasive speech |
| **Output** | MP3, AAC, Opus and telephone-band compression; deepfake detectability |
| **Perturbation** | Background noise and overlapping speakers; Gaussian noise, SPEC, SafeSpeech, POP and Enkidu protection; denoising of protected references |

Metrics: speaker similarity (SIM, ECAPA-TDNN), naturalness (MOS, UTMOS), word error rate (WER, Whisper
medium), mel-cepstral distortion (MCD), emotion consistency and STOI.

## Results

From the [paper](https://arxiv.org/abs/2602.00443) (arXiv v3). SIM, MOS, WER and MCD are on clean LibriTTS
prompts; the last three columns are speaker similarity on other datasets. **Bold** marks the best value
per column, and † marks models reported in the paper's appendix.

| Model | SIM ↑ | MOS ↑ | WER ↓ | MCD ↓ | VCTK SIM ↑ | Chinese SIM ↑ | Cross-lingual SIM ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-TTS | **0.61** | **4.39** | 0.05 | **5.79** | **0.62** | **0.72** | **0.67** |
| IndexTTS | **0.61** | 4.06 | 0.05 | 6.61 | 0.57 | **0.72** | **0.67** |
| dots.tts † | 0.60 | 4.17 | 0.06 | 6.11 | 0.57 | 0.68 | 0.61 |
| CosyVoice 2 | 0.58 | 4.37 | 0.05 | 6.02 | 0.58 | **0.72** | 0.65 |
| ZipVoice | 0.58 | 4.13 | 0.05 | 7.09 | 0.55 | 0.71 | 0.63 |
| MOSS-TTS v1.5 † | 0.57 | 4.32 | 0.06 | 6.66 | 0.51 | 0.68 | 0.65 |
| GLM-TTS | 0.57 | 4.08 | 0.09 | 6.41 | 0.57 | 0.69 | 0.66 |
| MaskGCT | 0.57 | 3.93 | 0.09 | 6.91 | 0.56 | 0.67 | 0.63 |
| Higgs TTS 3 † | 0.56 | 4.23 | 0.05 | 6.32 | 0.45 | 0.63 | 0.30 |
| F5-TTS | 0.56 | 3.99 | 0.12 | 6.96 | 0.54 | 0.70 | 0.65 |
| Higgs Audio | 0.56 | 4.30 | 0.25 | 6.06 | 0.52 | 0.58 | 0.54 |
| Fish Audio S2 † | 0.54 | 4.37 | **0.04** | 6.16 | 0.51 | 0.66 | 0.62 |
| MGM-Omni | 0.54 | 4.28 | 0.09 | 5.82 | 0.45 | 0.71 | 0.63 |
| PlayDiffusion | 0.51 | 4.15 | 0.05 | 8.06 | 0.43 | 0.44 | 0.46 |
| MOSS-TTSD | 0.49 | 4.10 | 0.38 | 7.09 | 0.44 | 0.44 | 0.44 |
| VibeVoice | 0.48 | 3.83 | 0.23 | 6.76 | 0.44 | 0.56 | 0.53 |
| FishSpeech | 0.47 | 4.37 | 0.17 | 6.47 | 0.43 | 0.61 | 0.57 |
| XTTS-v2 | 0.45 | 3.81 | 0.07 | 8.62 | 0.45 | 0.57 | 0.51 |
| Spark-TTS | 0.41 | 4.06 | 0.33 | 5.83 | 0.53 | 0.57 | 0.48 |
| OZSpeech | 0.39 | 3.21 | 0.06 | 6.87 | 0.25 | 0.00 | 0.17 |
| OpenVoice V2 | 0.24 | 4.30 | 0.07 | 7.06 | 0.39 | 0.43 | 0.30 |
| StyleTTS 2 | 0.23 | 4.30 | 0.05 | 6.81 | 0.24 | 0.11 | 0.21 |

<details>
<summary><b>Speaker similarity under anti-cloning protection</b> (LibriTTS)</summary>

<br>

A lower SIM under protection means the protection hides the speaker's voice better.

| Model | Clean | SafeSpeech | SPEC | Enkidu | Gaussian | POP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-TTS | 0.61 | 0.38 | 0.36 | 0.50 | 0.41 | 0.58 |
| IndexTTS | 0.61 | 0.35 | 0.32 | 0.47 | 0.39 | 0.57 |
| dots.tts † | 0.60 | 0.41 | 0.39 | 0.49 | 0.44 | 0.57 |
| CosyVoice 2 | 0.58 | 0.32 | 0.30 | 0.45 | 0.38 | 0.55 |
| ZipVoice | 0.58 | 0.29 | 0.26 | 0.44 | 0.26 | 0.54 |
| MOSS-TTS v1.5 † | 0.57 | 0.33 | 0.31 | 0.43 | 0.33 | 0.53 |
| GLM-TTS | 0.57 | 0.33 | 0.31 | 0.44 | 0.39 | 0.53 |
| MaskGCT | 0.57 | 0.30 | 0.28 | 0.41 | 0.31 | 0.53 |
| Higgs TTS 3 † | 0.56 | 0.48 | 0.48 | 0.49 | 0.34 | 0.53 |
| F5-TTS | 0.56 | 0.21 | 0.18 | 0.43 | 0.14 | 0.52 |
| Higgs Audio | 0.56 | 0.26 | 0.24 | 0.43 | 0.27 | 0.52 |
| Fish Audio S2 † | 0.54 | 0.32 | 0.30 | 0.43 | 0.34 | 0.52 |
| MGM-Omni | 0.54 | 0.18 | 0.17 | 0.32 | 0.23 | 0.49 |
| PlayDiffusion | 0.51 | 0.17 | 0.15 | 0.34 | 0.16 | 0.47 |
| MOSS-TTSD | 0.49 | 0.24 | 0.22 | 0.34 | 0.25 | 0.08 |
| VibeVoice | 0.48 | 0.27 | 0.25 | 0.37 | 0.28 | 0.45 |
| FishSpeech | 0.47 | 0.24 | 0.21 | 0.33 | 0.23 | **0.01** |
| XTTS-v2 | 0.45 | 0.26 | 0.24 | 0.31 | 0.24 | 0.41 |
| Spark-TTS | 0.41 | 0.13 | 0.11 | 0.14 | 0.06 | 0.36 |
| OZSpeech | 0.39 | 0.16 | 0.15 | 0.19 | 0.15 | 0.34 |
| OpenVoice V2 | 0.24 | 0.18 | 0.18 | 0.19 | 0.18 | 0.24 |
| StyleTTS 2 | 0.23 | **0.09** | **0.08** | **0.12** | **0.03** | 0.21 |

</details>

Results for compression, deepfake detectability, long-form and expressive speech are in the paper and on
the [website](https://nanboy-ronan.github.io/RVCBench/).

## Reproduce the paper

To reproduce the paper, use the code released with it, kept on the
[`v1`](https://github.com/Nanboy-Ronan/RVCBench/tree/v1) branch (tag `v1.0`). This branch, `main`, is v2:
the same benchmark rebuilt as an installable package for evaluating new models.

```bash
git clone --branch v1 https://github.com/Nanboy-Ronan/RVCBench.git RVCBench-v1
```

| | v1 (branch `v1`) | v2 (`main`) |
| --- | --- | --- |
| Use it to | **Reproduce the paper** | Evaluate a new model |
| Status | Frozen at the paper release | Under active development |
| Install | Clone, then `pip install` a list of packages | `pip install` the `rvcbench` package |
| Run a built-in model | `python run_vc.py --config-name ...` | `rvcbench run --config-name ...` |
| Evaluate your own model | Add an adapter to the codebase | Score audio generated anywhere, or a one-file adapter |
| Evaluation data | Full datasets | Full datasets, plus the `core-v1` and `full-v1` suites |

## Run the built-in models

RVCBench includes adapters for 32 voice cloning models, 5 protection methods and a denoising stage.
Each model runs in its own environment from [`envs/`](https://github.com/Nanboy-Ronan/RVCBench/tree/main/envs).

```bash
git clone https://github.com/Nanboy-Ronan/RVCBench.git && cd RVCBench
python -m pip install -e '.[eval]'
rvcbench run --config-name ots_vc/clean/libritts/qwen3_tts_ots dataset.speaker_id=1089 adversary.max_samples=5
```

Every run writes a per-sample `run_manifest.json` with input and output hashes, seeds, failures and metric
coverage. See [Running the built-in models](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/models.md) and the [run guide](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/run_protocol.md).

## Documentation

| Guide | Covers |
| --- | --- |
| [Evaluate your own model](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/adding_a_model.md) | Prompt and output formats, batch lists, several models, adapters |
| [Core and full suites](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/core_suite.md) | Tasks, data, metrics and scoring time |
| [Metrics API](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/metrics.md) | Speaker similarity, WER, MOS, MCD, STOI and emotion in your own code |
| [Running the built-in models](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/models.md) | Installation options, supported models, protection and denoising |
| [Datasets](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/datasets.md) | Hub folders, manifest format, preprocessing |
| [Run guide](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/run_protocol.md) | Run records, resuming, scoring saved audio, timing |
| [Codebase versions](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/versions.md) | What changed between v1 and v2 |
| [Contributing](https://github.com/Nanboy-Ronan/RVCBench/blob/main/CONTRIBUTING.md) | Development setup, checks and repository layout |

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

[CC0-1.0](https://github.com/Nanboy-Ronan/RVCBench/blob/main/LICENSE). Model checkpoints, upstream code and source corpora keep their own licenses. Questions and
contributions are welcome through issues and pull requests, or at **ruinanjin@alumni.ubc.ca**.
