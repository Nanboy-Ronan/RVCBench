# Command-line reference

Run `rvcbench --help` to list commands and `rvcbench <command> --help` for the current options.
The main pip workflows use the commands below. Examples assume [installation](installation.md)
and scorer setup are complete.

## Check the environment

```bash
rvcbench --version
rvcbench doctor --eval --imports
rvcbench setup-scorers
rvcbench setup-scorers --metrics sim wer speechmos
rvcbench setup-scorers --check-only
```

`doctor` checks dependencies; `--imports` also imports them in a subprocess. `setup-scorers` downloads
and verifies scorer models. `--check-only` verifies and loads cached assets without downloading.

## Export evaluation inputs

```bash
rvcbench prompts --suite onboarding-v1 --output prompts/
```

| Option | Meaning |
| --- | --- |
| `--suite` | Suite name; default `onboarding-v1`. Other packaged options: `core-v1`, `full-v1`. |
| `--output` | Required destination for prompt lists and reference audio. |
| `--data-root` | Optional existing local dataset copy, laid out like the Hub dataset. |

See [prompt formats and inference](adding_a_model.md#route-0-score-audio-you-generated-anywhere).

## Score generated outputs

```bash
rvcbench score --suite onboarding-v1 \
  --generated outputs/my-model --output results/my-model --device cpu
```

| Option | Meaning |
| --- | --- |
| `--suite` | Must match the suite used to export prompts. Default `onboarding-v1`. |
| `--generated` | One or more directories of generated WAV files. |
| `--model` | Optional model name per generated directory; defaults to each directory name. |
| `--output` | Results directory. With several models, one subdirectory is created per model. |
| `--device` | Scoring device; default `cpu`. GPU examples: `cuda`, `cuda:0`. |
| `--resume` | Continue an existing compatible evaluation and reuse matching metric caches. |
| `--data-root` | Optional local dataset copy. Use the same data as prompt export. |

Use a new output directory for a new evaluation. To continue, repeat the original arguments with
`--resume`. See [result interpretation](quickstart.md#score-and-inspect-results) and
[resume behavior](adding_a_model.md#resume-a-stopped-evaluation).

## Compare models

```bash
rvcbench compare results/model-a results/model-b --output results/compare
```

Inputs can be result directories or `submission.json` files. `--output` writes `comparison.md`,
`comparison.csv` and `comparison.json`. Comparison requires matching suites and scoring fingerprints.
`--allow-incompatible` permits inspection with differences listed and disables ranking.

## Inspect a task run

```bash
rvcbench status results/my-model/english-libritts
rvcbench report results/my-model/english-libritts --output task-report.json
```

Pass a task directory containing `run_manifest.json`, not the top-level suite directory.
`report` requires complete outputs and required metrics. See [runs and reports](run_protocol.md).

## Built-in models and research workflows

`run`, `run-protected`, `protect` and `denoise` accept Hydra configurations. They require the relevant
model environments; see [built-in models](models.md). Specialized source-audit, noise replay,
protection and timing commands are listed by `rvcbench --help` and documented in the
[run guide](run_protocol.md).
