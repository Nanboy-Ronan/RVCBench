---
title: Voice cloning evaluation FAQ
description: Learn how RVCBench evaluates voice cloning, which speech metrics to use, what inputs are required and how to benchmark a new model with provided datasets.
---

# Voice cloning evaluation FAQ

RVCBench is a Python package for comprehensive voice cloning evaluation. It combines automatic speech
metrics with ready-to-use evaluation datasets. You can score your own audio or use benchmark prompts
to compare models across languages, speakers and recording conditions.

## Can I use the metrics without downloading benchmark data?

Yes. Install `rvcbench[eval]`, run `rvcbench setup-scorers`, and use `rvcbench.metrics.Evaluator` with
your own generated audio, references and text. The [metrics guide](metrics.md) includes single-file
and batch examples. Metric models are downloaded once and reused locally.

## Which aspects of voice cloning does RVCBench measure?

The public API exposes seven metrics across five aspects:

| Aspect | Metrics | Inputs beyond generated audio |
| --- | --- | --- |
| Speaker identity | `sim`, `sva` | Recording of the intended speaker |
| Content accuracy | `wer` | Expected text; optional language hint |
| Predicted naturalness | `speechmos` (UTMOS) | None |
| Acoustic fidelity and intelligibility | `mcd`, `stoi` | Recording of the same text |
| Emotion consistency | `emotion` | Reference recording with the intended expression |

These scores describe complementary properties. They are not a single universal quality score and do
not replace human listening tests. See [metric definitions](metrics.md#metric-definitions).

## How do I benchmark my model with the provided datasets?

Use `rvcbench prompts` to download selected data and export reference recordings and text. Generate
speech in your model's own environment or through its API, then run `rvcbench score` on the WAV files.
RVCBench handles data preparation, scoring and reports; you supply model inference. See the
[complete quickstart](quickstart.md#2b-evaluate-a-model-with-our-data).

## Does a new model need a built-in adapter?

No. Any model that can generate WAV files from the exported prompts can use the dataset scoring
workflow. Save outputs using the exported IDs. JSONL, ZipVoice-style TSV and Seed-TTS-style lists are
provided. Built-in adapters are an additional option, not the limit on supported model submissions.
See [model integration](adding_a_model.md).

## Which benchmark suite should I start with?

Use `onboarding-v1` with 52 outputs to check your integration. Use `core-v1` with 480 outputs for broader
coverage, including protected references. Use `full-v1` with 12,724 outputs for larger datasets;
protection tasks currently require `core-v1` separately. These are preview suite protocols, distinct
from paper reproduction. See [suite coverage](core_suite.md).

## What if I do not have a recording of the same text?

Use the metrics supported by your inputs. Speaker similarity only needs a recording of the intended
speaker, WER needs expected text, and SpeechMOS needs only generated audio. MCD and STOI require a
same-text recording; do not compare unrelated sentences. See [audio input requirements](api.md#score).

## Can I score on CPU and resume an interrupted evaluation?

Yes. CPU is the default device, and you can select CUDA for GPU scoring. Reuse an `Evaluator` for
batch API scoring; for dataset scoring, repeat the same command with `--resume` to reuse matching
successful scores and retry failed or changed samples. See [installation](installation.md) and
[resume behavior](adding_a_model.md#resume-a-stopped-evaluation).

## How can I compare models fairly?

Use the same suite, inputs and scoring environment for every model. RVCBench checks per-task scoring
fingerprints and reports coverage and failures. `rvcbench compare` refuses incompatible results by
default. Its explicit incompatibility override produces an unranked inspection. See the
[comparison workflow](adding_a_model.md#several-models-at-once).

## Does comprehensive evaluation mean every possible metric is included?

It means the package covers the five aspects above through one API, with datasets spanning multiple
languages, speakers and recording conditions. The paper additionally studies deepfake detectability
and an audio-LLM expression judge; those two components are not yet in the public API or packaged
suites. See [coverage](core_suite.md) before selecting a protocol for your model.
