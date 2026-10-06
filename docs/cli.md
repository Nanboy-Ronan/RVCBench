---
title: "RVCBench command-line reference"
description: "Commands, argument values, defaults and examples for RVCBench. Look up every suite, task ID, metric and output option."
---

# Command-line reference

```bash
rvcbench --version
rvcbench --help
rvcbench score --help
```

Flags such as `--resume` take no value. Lists use spaces: `--tasks chinese french`, not commas.

## List scenarios

```bash
rvcbench tasks --suite core-v1
rvcbench tasks --suite core-v1 --json
```

| Argument | Accepted values | Default | Result |
| --- | --- | --- | --- |
| `--suite` | `onboarding-v1`, `core-v1`, `full-v1`, or a custom suite `.json` path | `core-v1` | Tasks to list |
| `--json` | Flag; no value | Off | Print JSON instead of text |

## Export evaluation inputs

```bash
rvcbench prompts --suite core-v1 --tasks chinese --output prompts/
```

| Argument | Accepted values | Default | Result |
| --- | --- | --- | --- |
| `--suite` | `onboarding-v1`, `core-v1`, `full-v1`, or a custom suite `.json` path | `onboarding-v1` | Evaluation suite |
| `--tasks` | One or more IDs from the [task list below](#task-values) | All tasks in the suite | Export only selected tasks and their dependencies |
| `--output` | Directory path, e.g. `prompts/`; must be new or empty | **Required** | Prompt lists and reference WAV files |
| `--data-root` | Dataset directory, e.g. `/data/RVCBench` containing `Libritts/`, `VCTK/`, etc. | Download from Hugging Face | Read local data |

Output: `prompts.jsonl`, `prompts.tsv`, `prompts.lst`, reference audio, `suite.json` and `README.md`.
Generate one WAV per prompt with your model. Save it as `<id>.wav` or `<task>/<pair_id>.wav`.
See [inference examples](adding_a_model.md#route-0-score-audio-you-generated-anywhere).

## Score generated outputs

```bash
rvcbench score --suite core-v1 --tasks chinese \
  --generated outputs/my-model --output results/my-model --device cuda
```

| Argument | Accepted values | Default | Result |
| --- | --- | --- | --- |
| `--suite` | `onboarding-v1`, `core-v1`, `full-v1`, or a custom suite `.json` path | `onboarding-v1` | Must match prompt export |
| `--tasks` | One or more IDs from the [task list below](#task-values) | All tasks in the suite | Must match prompt export |
| `--generated` | One or more directories, e.g. `outputs/model-a outputs/model-b` | **Required** | Generated WAV files to score |
| `--model` | One name per generated directory, e.g. `model-a model-b` | Directory names | Labels in reports; batch names must start with a letter/digit, contain only letters, digits, `_`, `-`, `.`, and exclude `__` |
| `--output` | Directory path, e.g. `results/my-model` | **Required** | Write results here; use a new directory unless resuming |
| `--device` | `cpu`, `cuda`, or `cuda:N` where N is a GPU index, e.g. `cuda:1` | `cpu` | Scoring device |
| `--resume` | Flag; no value | Off | Reuse matching scores in an existing output directory |
| `--data-root` | Dataset directory, same as prompt export | Download from Hugging Face | Read local data |

One model: read `OUTPUT/submission.json`. Several models: read `OUTPUT/comparison.md`, `.csv` or `.json`;
each model also gets `OUTPUT/MODEL/submission.json`. Scores are reported per task.

Resume: repeat the same command with `--resume`. Keep the suite, tasks, model names and directories the same.

## Suite values

| `--suite` | Audio files to generate per model | Use |
| --- | ---: | --- |
| `onboarding-v1` | 52 | Check your integration |
| `core-v1` | 480 | Small evaluation across paper scenarios |
| `full-v1` | 12,724 | Larger evaluation; excludes AdvNoise and AntiProtect |
| `/path/to/suite.json` | Defined in the file | Custom evaluation; manifest paths are relative to that file |

These counts are for whole suites. `--tasks` reduces the work to your selection plus its dependencies.

## Task values

| Suite | Scenario | Accepted `--tasks` values |
| --- | --- | --- |
| onboarding-v1 | Onboarding | `libritts`, `vctk`, `robotcall` |
| core-v1, full-v1 | AudioShift | `audioshift` |
| core-v1, full-v1 | TextShift / Expression | `textshift-standard`, `textshift-hallucination`, `textshift-scam`, `textshift-scam-standard` |
| core-v1, full-v1 | Multilingual | `english-libritts`, `chinese`, `french`, `crosslingual` |
| core-v1, full-v1 | LongContext | `longtext`, `longaudio` |
| core-v1, full-v1 | PassiveNoise | `background`, `background-clean`, `multispeaker`, `multispeaker-clean` |
| core-v1 | AdvNoise / AntiProtect | `adv-clean`, `adv-gaussian`, `adv-spec`, `adv-safespeech`, `adv-pop`, `adv-enkidu`, `antiprotect-spec` |
| core-v1, full-v1 | Compression | `compression-mp3-64k`, `compression-aac-64k`, `compression-opus-24k`, `compression-mp3-32k`, `compression-aac-32k`, `compression-opus-16k`, `compression-narrowband` |

```bash
# Multiple tasks: separate IDs with spaces.
rvcbench prompts --suite core-v1 --tasks chinese crosslingual background --output prompts/
```

Omit `--tasks` to select all tasks. `--tasks all` is not supported.
Clean controls and source tasks are added automatically: `background` adds `background-clean`;
`compression-mp3-64k` adds `audioshift`. Generate every exported prompt.
See [task meanings, sample counts and dependencies](core_suite.md#tasks).

## Check the environment

```bash
rvcbench doctor --eval --imports
rvcbench setup-scorers --metrics sim wer speechmos
```

### doctor

| Argument | Accepted values | Default | Result |
| --- | --- | --- | --- |
| `--model` | `qwen3`, `qwen3_omni`, `sparktts` | No model check | Include that model's dependencies |
| `--eval` | Flag; no value | Off | Include scoring dependencies |
| `--imports` | Flag; no value | Off | Import dependencies to detect broken installations |

### setup-scorers

| Argument | Accepted values | Default | Result |
| --- | --- | --- | --- |
| `--metrics` | One or more of `sim`, `sva`, `wer`, `speechmos`, `mcd`, `stoi`, `emotion` | `sim speechmos wer mcd emotion stoi` | Download and verify these scorers |
| `--check-only` | Flag; no value | Off | Check cached files and load models; download nothing |

`sim` and `sva` share a model. `mcd` and `stoi` need no model files. Omit `--metrics` to set up all
public scorers; `--metrics all` is not supported. This command prepares assets; it does not change
the metrics used by `rvcbench score`, which are fixed by the suite.

## Compare models

```bash
rvcbench compare results/model-a results/model-b --output results/compare
```

| Argument | Accepted values | Default | Result |
| --- | --- | --- | --- |
| `submissions` | One or more result directories or `submission.json` paths | **Required** | Reports to compare |
| `--output` | Directory path, e.g. `results/compare` | Print only | Write `comparison.md`, `.csv` and `.json` |
| `--allow-incompatible` | Flag; no value | Off | Show differing scorer protocols without ranking |

All reports must use the same suite and task selection, even with `--allow-incompatible`.

## Inspect a task run

```bash
rvcbench status results/my-model/chinese
rvcbench report results/my-model/chinese --output chinese-report.json
```

| Command | Argument | Accepted values | Default |
| --- | --- | --- | --- |
| `status` | `run_dir` | Task directory containing `run_manifest.json` | **Required** |
| `report` | `run_dir` | Task directory containing `run_manifest.json`; run must be complete | **Required** |
| `report` | `--output` | JSON file path, e.g. `chinese-report.json` | **Required** |
| `smoke` | `--output` | New directory path for a synthetic CPU check | `results/smoke` |

## Built-in models and research workflows

`run`, `run-protected`, `protect` and `denoise` use Hydra `key=value` arguments:

```bash
rvcbench run --config-name ots_vc/clean/libritts/qwen3_tts_ots --cfg job
```

| Argument | Accepted values | Result |
| --- | --- | --- |
| `--config-name` | Config path without `.yaml`, e.g. `ots_vc/clean/libritts/qwen3_tts_ots` | Choose a run configuration |
| `--cfg` | `job`, `hydra`, `all` | Print the selected configuration without running |
| `key=value` | A key and value from that configuration, e.g. `device=cuda:0` | Override a setting |
| `+key=value` | A new configuration key, e.g. `+vc.generate_only=true` | Add a setting |

Config names and model-specific settings are listed in [built-in models](models.md) and
[model setup](quickstart_model_setup.md). `rvcbench run --help` lists available config groups.

### Audit and compare run records

| Command | Argument | Accepted values | Default |
| --- | --- | --- | --- |
| `audit-source` | `run_dir` | Task run directory | **Required** |
| `audit-source` | `--root` | Source checkout or installed package root | Current installation |
| `audit-source` | `--output` | JSON file path | Print only |
| `compare-check`, `compare-timing` | `left`, `right` | Two task run directories | **Required** |
| `compare-check` | `--metrics` | Space-separated metric IDs: `sim`, `sva`, `wer`, `speechmos`, `mcd`, `stoi`, `emotion`; legacy `dnsmos` | `mcd wer sim` |
| `compare-check`, `compare-timing` | `--output` | JSON file path | Print only |

### replay-gr

Replay a recorded Gaussian-noise stage. [Input file formats](gr_noise_replay.md).

| Argument | Accepted values | Default |
| --- | --- | --- |
| `--dataset-root` | Local dataset directory | **Required** |
| `--subset-manifest` | Frozen selection JSON file | **Required** |
| `--noise-archive` | Recorded noise archive path | **Required** |
| `--historical-directory` | Directory of historical WAV files to verify against | **Required** |
| `--output` | Output directory | **Required** |
| `--batch-size` | Integer | `8` |
| `--sample-rate` | Integer, Hz | `24000` |
| `--hop-length` | Integer, samples | `512` |
| `--regenerate-rng` | Flag; no value | Off |
| `--seed` | Integer; used with `--regenerate-rng` | `42` |
| `--epsilon` | Float; perturbation magnitude | `0.03137255` |
| `--device` | `cpu`, `cuda`, `cuda:N` | `cpu` |

### denoise-dns64

Denoise protected references. [Input file formats](dns64_stage.md).

| Argument | Accepted values | Default |
| --- | --- | --- |
| `--dataset-root` | Local dataset directory | **Required** |
| `--subset-manifest` | Frozen selection JSON file | **Required** |
| `--reference-directory` | Protected reference WAV directory | **Required** |
| `--weights` | Local DNS64 weights file | **Required** |
| `--output` | Output directory | **Required** |
| `--device` | `cpu`, `cuda`, `cuda:N` | `cpu` |
| `--dry` | Float from `0` to `1`; fraction of original audio to mix in | `0.0` |
| `--dataset-rate` | Positive integer, Hz | `16000` |
| `--runtime-python` | Python executable path | Current interpreter |
| `--timeout-seconds` | Number of seconds for the external runtime | `600` |

### protect-enkidu

Generate Enkidu-protected references. [Input file formats](enkidu_stage.md).

| Argument | Accepted values | Default |
| --- | --- | --- |
| `--dataset-root` | Local dataset directory | **Required** |
| `--subset-manifest` | Frozen selection JSON file | **Required** |
| `--model-directory` | Local Enkidu model directory | **Required** |
| `--output` | Output directory | **Required** |
| `--device` | `cpu`, `cuda`, `cuda:N` | `cpu` |
| `--seed` | Integer | `42` |
| `--epochs` | Positive integer | `10` |
