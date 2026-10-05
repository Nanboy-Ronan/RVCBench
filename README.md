<div align="center">

<img src="https://raw.githubusercontent.com/Nanboy-Ronan/RVCBench/main/figs/logo.png" alt="RVCBench logo" width="110">

# RVCBench

### Comprehensive Voice Cloning Evaluation

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-6842c2.svg)](https://arxiv.org/abs/2602.00443)
[![arXiv](https://img.shields.io/badge/arXiv-2602.00443-b31b1b.svg)](https://arxiv.org/abs/2602.00443)
[![PyPI](https://img.shields.io/pypi/v/rvcbench.svg)](https://pypi.org/project/rvcbench/)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffcc00.svg)](https://huggingface.co/datasets/Nanboy/RVCBench)
[![Website](https://img.shields.io/badge/Website-RVCBench-0d6ea8.svg)](https://nanboy-ronan.github.io/RVCBench/)
[![CI](https://github.com/Nanboy-Ronan/RVCBench/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Nanboy-Ronan/RVCBench/actions/workflows/ci.yml)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-lightgrey.svg)](https://github.com/Nanboy-Ronan/RVCBench/blob/main/LICENSE)

[**Documentation**](https://nanboy-ronan.github.io/RVCBench/docs/) · [**Paper**](https://arxiv.org/abs/2602.00443) · [**Website**](https://nanboy-ronan.github.io/RVCBench/) · [**Dataset**](https://huggingface.co/datasets/Nanboy/RVCBench) · [**Demo**](https://huggingface.co/spaces/Nanboy/RVCBench) · [**Evaluate your model**](#evaluate-your-model) · [**Reproduce the paper**](#reproduce-the-paper)

</div>

> **News**
>
> - **2026-10** · `pip install "rvcbench[eval]"`: score your own model from ZipVoice- or Seed-TTS-style batch lists, compare models in one table, or use the metrics in your own code.
> - **2026-09** · RVCBench is accepted to **NeurIPS 2026**.

RVCBench is a general-purpose package for evaluating voice cloning. It brings speaker similarity,
speech quality, intelligibility, pronunciation accuracy and emotion consistency into one scoring API,
and provides ready-to-use datasets for evaluating a model across languages, speakers and recording conditions.

| What you want to do | Start here |
| --- | --- |
| **Score your own audio** with automatic speech metrics | [Python metrics API](#score-your-own-audio) — use your own files and data |
| **Evaluate your model with our data** and get a complete report | [Dataset evaluation](#evaluate-your-model) — export prompts, generate, score |

Both workflows are included in the pip package. Your model can run in its own environment or through an API;
RVCBench scores the audio it produces.

**First time here?** Follow the complete [getting started guide](https://nanboy-ronan.github.io/RVCBench/docs/quickstart/)
or [中文入门指南](https://nanboy-ronan.github.io/RVCBench/docs/quickstart_zh/).
They walk through installation, required files, both workflows and reading the results.

## Install

```bash
python -m pip install "rvcbench[eval]"
# Linux: install FFmpeg if it is not already available (e.g. sudo apt-get install ffmpeg).
rvcbench setup-scorers
```

Python 3.10+ and Linux are supported. `rvcbench[eval]` installs the package **with all seven public metrics**;
plain `rvcbench` installs the runner and datasets without the optional scoring dependencies.
`setup-scorers` downloads and verifies the metric models once. They are reused from your local cache.
A GPU is optional; choose `device="cpu"` or `--device cpu` to score on CPU.

For a smaller initial download, select only the metrics you need:
`rvcbench setup-scorers --metrics sim wer speechmos`.
See the [installation guide](https://nanboy-ronan.github.io/RVCBench/docs/installation/)
for CPU/GPU installation, caches and troubleshooting.

## Score your own audio

```python
from rvcbench import metrics

with metrics.Evaluator(["sim", "wer", "speechmos"], device="cpu") as evaluator:
    scores = evaluator.score(
        "generated.wav",
        reference="speaker_reference.wav",  # a recording of the intended speaker
        text="Hello there.",                 # what the generated audio should say
        language="en",
    )
    print(scores)  # dictionary with sim, wer and speechmos
```

Reuse the evaluator for a whole dataset: call `score()` for each file, and each metric model loads once.
For every available metric, use `metrics.Evaluator("all")` and also pass
`target="same_text_recording.wav"` for MCD and STOI. The target is a recording of the **same text** as the
synthesized audio; the speaker reference may contain different words.

| Metric | Evaluates | Required input besides generated audio |
| --- | --- | --- |
| `sim`, `sva` | Speaker similarity and speaker verification | Speaker reference recording |
| `wer` | Pronunciation/content accuracy via ASR word error rate | Expected text; optional language |
| `speechmos` | Predicted perceptual quality (UTMOS) | None |
| `mcd`, `stoi` | Acoustic distortion and intelligibility | Same-text target recording |
| `emotion` | Emotion consistency | Reference recording |

The [metrics guide](https://nanboy-ronan.github.io/RVCBench/docs/metrics/)
includes one-line functions, scoring all metrics, a batch example and metric definitions.
No RVCBench dataset or model adapter is needed for this workflow.

## Evaluate your model

RVCBench downloads the selected data, prepares reference clips and texts, and scores the audio your model generates.
Start with the 52-utterance onboarding suite:

```bash
rvcbench prompts --suite onboarding-v1 --output prompts/
# Generate every prompt with your model. Save outputs as <id>.wav in outputs/my-model/.
rvcbench score --suite onboarding-v1 --generated outputs/my-model --output results/my-model --device cpu
```

`prompts/` contains `prompts.jsonl`, `prompts.tsv` (ZipVoice) and `prompts.lst` (Seed-TTS-style scripts
such as F5-TTS and CosyVoice). For example, a model environment with ZipVoice installed can run:

```bash
python3 -m zipvoice.bin.infer_zipvoice --model-name zipvoice \
  --test-list prompts/prompts.tsv --res-dir outputs/my-model
```

The [model evaluation guide](https://nanboy-ronan.github.io/RVCBench/docs/adding_a_model/)
shows the exact file format and an example loop for your own inference function.
Model inference is the step supplied by you; data preparation, metric scoring and report generation are automatic.

| Suite | Utterances to generate | Use it for |
| --- | ---: | --- |
| `onboarding-v1` | 52 | Check the complete workflow |
| `core-v1` | 480 | Broad evaluation across languages, speaker groups, long speech, noise, compression and protection |
| `full-v1` | 12,724 | The larger evaluation datasets; protection tasks are currently covered by `core-v1` |

Use the same suite name for `prompts` and `score`. Every input is checked against a fixed dataset revision;
target recordings stay in the scoring data and are not exported with the prompts.
`submission.json` contains per-task scores, coverage and failures. Task directories include per-sample scores
and bootstrap confidence intervals. These suites are currently previews; published paper results use a separate protocol.

Resume an interrupted or partial evaluation with the original command plus `--resume`:

```bash
rvcbench score --suite onboarding-v1 --generated outputs/my-model --output results/my-model --device cpu --resume
```

Compare several models in one call, or compare reports you already have:

```bash
rvcbench score --suite core-v1 --generated outputs/model-a outputs/model-b --output results/comparison --device cuda
rvcbench compare results/model-a results/model-b --output results/compare
```

Comparison checks both the suite and scoring fingerprints. Different scoring environments must be rescored
together, or explicitly inspected with `--allow-incompatible`, which disables ranking.
The [suite guide](https://nanboy-ronan.github.io/RVCBench/docs/core_suite/)
lists tasks, coverage and approximate scoring costs.

## Evaluation coverage

The package covers clean voice cloning, multilingual and cross-lingual speech, speaker demographics,
long-form speech, unusual text, background noise, overlapping speakers, compression, and protected references.
Its seven public metrics cover identity, content, quality, intelligibility and emotion.
The paper additionally studies deepfake detectability and an audio-LLM expression judge; these two components
are not yet available through the packaged suites.

The [paper](https://arxiv.org/abs/2602.00443) defines 18 evaluations over 14,370 utterances from 204 speakers.
The package also includes adapters for running existing models and tools for protection and denoising research.

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
coverage. See [Running the built-in models](https://nanboy-ronan.github.io/RVCBench/docs/models/) and the [run guide](https://nanboy-ronan.github.io/RVCBench/docs/run_protocol/).

## Documentation

| Guide | Covers |
| --- | --- |
| [Getting started](https://nanboy-ronan.github.io/RVCBench/docs/quickstart/) · [中文](https://nanboy-ronan.github.io/RVCBench/docs/quickstart_zh/) | Your first evaluation, from installation to results |
| [Evaluate your own model](https://nanboy-ronan.github.io/RVCBench/docs/adding_a_model/) | Prompt and output formats, batch lists, several models, adapters |
| [Core and full suites](https://nanboy-ronan.github.io/RVCBench/docs/core_suite/) | Tasks, data, metrics and scoring time |
| [Install the package](https://nanboy-ronan.github.io/RVCBench/docs/installation/) | pip installation, CPU/GPU, downloads and troubleshooting |
| [Metrics API](https://nanboy-ronan.github.io/RVCBench/docs/metrics/) | Speaker similarity, WER, MOS, MCD, STOI and emotion in your own code |
| [Running the built-in models](https://nanboy-ronan.github.io/RVCBench/docs/models/) | Installation options, supported models, protection and denoising |
| [Datasets](https://nanboy-ronan.github.io/RVCBench/docs/datasets/) | Hub folders, manifest format, preprocessing |
| [Run guide](https://nanboy-ronan.github.io/RVCBench/docs/run_protocol/) | Run records, resuming, scoring saved audio, timing |
| [Codebase versions](https://nanboy-ronan.github.io/RVCBench/docs/versions/) | What changed between v1 and v2 |
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
