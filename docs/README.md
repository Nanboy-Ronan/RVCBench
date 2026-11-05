---
title: "Voice cloning evaluation with RVCBench"
description: "Evaluate voice cloning with seven automatic speech metrics and ready-to-use benchmark datasets. Use your own audio or export prompts, generate and score."
---

# Using RVCBench

[Read the documentation website](https://nanboy-ronan.github.io/RVCBench/docs/) ·
[Project homepage](https://nanboy-ronan.github.io/RVCBench/) · [PyPI](https://pypi.org/project/rvcbench/)

Score generated speech, or evaluate a model on RVCBench data.

| I want to… | Open |
| --- | --- |
| Install and run my first evaluation | [Quickstart](getting-started/quickstart.md) |
| Look up a command argument, its values or its default | [CLI reference](reference/cli.md) |
| Choose a paper scenario | [Scenarios and task IDs](evaluation/core_suite.md) |
| Score audio from my own dataset | [Python API arguments](reference/api.md) and [metric examples](evaluation/metrics.md) |
| Connect my model or use a batch inference script | [Model evaluation](evaluation/adding_a_model.md) |
| Set up GPU scoring or fix installation | [Installation](getting-started/installation.md) |
| Find dataset names and file formats | [Datasets](evaluation/datasets.md) |
| Run a built-in model | [Models](integrations/models.md) and [environments](integrations/model_environments.md) |
| Inspect or resume a run | [Run guide](reference/run_protocol.md) |
| Reproduce the paper's results | [Frozen v1 codebase](reproduction/versions.md) |

## Run one scenario

After [installation](getting-started/installation.md):

```bash
rvcbench tasks --suite core-v1
rvcbench prompts --suite core-v1 --tasks chinese --output prompts/
# Generate every prompt with your model in outputs/my-model/.
rvcbench score --suite core-v1 --tasks chinese \
  --generated outputs/my-model --output results/my-model --device cpu
```

Read `results/my-model/submission.json` for per-task scores.
Replace `chinese` with [task IDs](reference/cli.md#task-values), separated by spaces.

## Contribute

See [CONTRIBUTING](../CONTRIBUTING.md) for development checks and release validation,
[model integration](evaluation/adding_a_model.md#route-2-built-in-integration) for adding an adapter, and
[website maintenance](../website/README.md) for rebuilding the homepage.
