---
title: "Voice cloning evaluation with RVCBench"
description: "Evaluate voice cloning with seven automatic speech metrics and ready-to-use benchmark datasets. Use your own audio or export prompts, generate and score."
---

# Using RVCBench

[Read the documentation website](https://nanboy-ronan.github.io/RVCBench/docs/) ·
[Project homepage](https://nanboy-ronan.github.io/RVCBench/) · [PyPI](https://pypi.org/project/rvcbench/)

RVCBench provides automatic speech metrics and datasets for comprehensive voice cloning evaluation.
Both workflows are available through `pip install "rvcbench[eval]"`; a repository checkout is optional.

## Two ways to evaluate voice cloning

**Automatic metrics for your audio.** Score speaker identity, content accuracy, predicted naturalness,
acoustic fidelity, intelligibility and emotion with seven metrics in one Python API. Reuse the same
evaluator across files. [Start with the metrics API](metrics.md).

**Benchmark data for your model.** Export reference audio and text, generate speech using any model
or API, then get per-task scores, coverage and comparison reports. Start with 52 onboarding samples
and expand to the core or full suite. [Evaluate your model](adding_a_model.md).

Your model can run in a separate environment. No built-in adapter is required to score its outputs.
See the [evaluation FAQ](faq.md) for metric inputs, coverage and choosing a suite.

## Start here

1. **[Getting started](quickstart.md)**: install, score your first audio,
   evaluate a model with our data, and read the results.
2. **[Installation](installation.md)**: CPU/GPU setup, model downloads, offline use, upgrades and troubleshooting.
3. Choose the workflow you need:

| Your goal | Guide |
| --- | --- |
| Score files you already have | [Metrics API](metrics.md): seven metrics, required inputs and batch scoring |
| Evaluate your model using benchmark data | [Model evaluation](adding_a_model.md): prompts, inference and output formats |
| Choose evaluation coverage | [Suites](core_suite.md): onboarding, core and full datasets |
| Understand and reuse the data | [Datasets](datasets.md): source folders and metadata |
| Look up an argument or command | [Python API](api.md) and [command-line reference](cli.md) |
| Run a supported model inside RVCBench | [Built-in models](models.md) and [model environments](model_environments.md) |
| Inspect, resume or audit an adapter run | [Run guide](run_protocol.md) |
| Reproduce the published paper | [Codebase versions](versions.md): use the frozen `v1` branch |

The current public API has seven metrics. Packaged suites are previews and have their own fixed selections;
they should not be presented as reproductions of the paper's results. See [suite coverage](core_suite.md).

## Contribute

See [CONTRIBUTING](../CONTRIBUTING.md) for development checks and release validation,
[model integration](adding_a_model.md#route-2-built-in-integration) for adding an adapter, and
[website maintenance](site-src/README.md) for rebuilding the homepage.
