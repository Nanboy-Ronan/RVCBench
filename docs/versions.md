---
title: "RVCBench package versions and paper reproduction"
description: "Choose the v2 pip package to evaluate new models or the frozen v1 codebase to reproduce the paper. Review changes in execution, datasets and reporting."
---

# Codebase versions

v1 and v2 are two versions of the code for the same benchmark: the datasets, metrics and paper results are
the same. **To reproduce the paper, use v1.** To evaluate a new model, use v2.

```text
tag v1.0 (2026-09-30): the code released with the paper
├── branch v1    v1.0 plus README changes only; frozen
└── branch main  v1.0 plus the v2 refactor (2026-09-30 to 2026-10-02); under development
```

| | v1 (branch [`v1`](https://github.com/Nanboy-Ronan/RVCBench/tree/v1), tag [`v1.0`](https://github.com/Nanboy-Ronan/RVCBench/tree/v1.0)) | v2 (branch [`main`](https://github.com/Nanboy-Ronan/RVCBench/tree/main)) |
| --- | --- | --- |
| Use it to | **Reproduce the paper** | Evaluate a new model |
| Status | Frozen at the paper release | Under active development |
| Install | Clone, then `pip install` a list of packages | `pip install` the `rvcbench` package from GitHub |
| Run a built-in model | `python run_vc.py --config-name ...` | `rvcbench run --config-name ...` (same config names) |
| Evaluate your own model | Add an adapter to the codebase | Score audio generated anywhere (`rvcbench prompts`, `rvcbench score`), or a one-file adapter |
| Evaluation data | Full datasets | Full datasets, plus the `core-v1` suite: 480 pinned utterances covering 16 of the paper's 18 evaluations |
| Run output | `metrics.json` per run | Per-sample `run_manifest.json` with input and output hashes, failures and metric coverage |

```bash
git clone --branch v1 https://github.com/Nanboy-Ronan/RVCBench.git RVCBench-v1   # reproduce the paper
git clone https://github.com/Nanboy-Ronan/RVCBench.git RVCBench                  # v2
```

v1 and v2 name versions of this code. They are unrelated to the paper's arXiv versions and to suite names such
as `core-v1`.

## What changed in v2

| Area | Main code | Change |
| --- | --- | --- |
| Package and commands | [`src/rvcbench/`](../src/rvcbench/), [`benchmark/cli.py`](../src/rvcbench/benchmark/cli.py) | Installable `rvcbench` package with configs inside it; `rvcbench run`, `prompts`, `score`, `setup-scorers` and the run-record commands. |
| Suites | [`benchmark/submission.py`](../src/rvcbench/benchmark/submission.py), [`suites/`](../src/rvcbench/suites/) | Versioned suites with pinned inputs; score audio generated anywhere. |
| Execution and reporting | [`benchmark/`](../src/rvcbench/benchmark/), [`workflows/vc.py`](../src/rvcbench/workflows/vc.py) | Per-sample records of inputs, outputs, failures, metric coverage and run state; generation and scoring can run in separate environments. Completion and comparability are checked separately. |
| Model adapters | [`adversary/`](../src/rvcbench/adversary/), [`models/`](../src/rvcbench/models/) | Explicit per-sample requests for 13 models, seeds from the original sample index, invalid conditioning or audio rejected, owned resources released. Other models still use the compatibility path. |
| Model workers | [`models/worker_protocol.py`](../src/rvcbench/models/worker_protocol.py), [`scripts/`](../scripts/) | Bounded waits, request/response matching and process cleanup. |
| Checkpoints and dependencies | [`benchmark/model_assets.py`](../src/rvcbench/benchmark/model_assets.py), [Hub revisions](hub_revisions.md) | Explicit paths or pinned revisions, recorded hashes, strict state loading for selected models. |
| Data and reproduction | [`datasets/`](../src/rvcbench/datasets/), [`reproduction/`](../reproduction/) | Annotation variants kept in sample identity, validated source indices, frozen subsets with input hashes. |
| Protection, scoring and timing | [`benchmark/`](../src/rvcbench/benchmark/), [run guide](run_protocol.md) | Traced reference stages and scorer provenance; timing scopes declared so incompatible runs are not compared. |

**Behaviour changes.** Malformed inputs or incomplete checkpoints that v1 skipped or patched now fail.
Model versions, seed policies and retry or conditioning variants must be recorded when comparing v1 and v2
results. v2 runs do not replace the paper's tables. To reproduce those, use v1 together with its environment
files (`envs/`) and the checkpoint bundle linked from its README; the code alone does not pin model weights.

## Validation status

Real generation and core scoring on a fixed subset have been recorded for 27 of the 32 integrations,
including the paper's 18 models. These runs span different stages of the refactor and do not establish
reproduction on the current combined source or of the paper's tables. The other five integrations lack
assets or a verified cloning protocol. See [validation coverage](validation.md) and the
[reproduction plan](../reproduction/plan.json).
