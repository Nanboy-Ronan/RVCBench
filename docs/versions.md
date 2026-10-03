# Codebase versions

**v1** is the codebase before the architecture refactor. **v2** is the refactor on branch `v2`, still
under development. These names describe repository versions, not the
paper's arXiv versions.

| Version | Source | Use it to |
| --- | --- | --- |
| v1 | Branch [`main`](https://github.com/Nanboy-Ronan/RVCBench/tree/main), tag [`v1.0`](https://github.com/Nanboy-Ronan/RVCBench/tree/v1.0) | Inspect or run the original architecture and configurations |
| v2 | Branch [`v2`](https://github.com/Nanboy-Ronan/RVCBench/tree/v2) | Use the installable package, suites, run records and validation |

```bash
git clone https://github.com/Nanboy-Ronan/RVCBench.git RVCBench-v1
git clone --branch v2 https://github.com/Nanboy-Ronan/RVCBench.git RVCBench-v2
```

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
results. The paper's result tables remain historical results and are not replaced by v2 subset runs; the
v1 code alone does not reconstruct historical weights or environments.

## Validation status

Real generation and core scoring on a fixed subset have been recorded for 27 of the 32 integrations,
including the paper's 18 models. These runs span different stages of the refactor and do not establish
reproduction on the current combined source or of the paper's tables. The other five integrations lack
assets or a verified cloning protocol. See [validation coverage](validation.md) and the
[reproduction plan](../reproduction/plan.json).
