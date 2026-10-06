---
title: "RVCBench scenarios and task values"
description: "Choose a suite and task IDs. Copy a command for one scenario, check sample counts, and see which controls are included automatically."
---

# Scenarios and tasks

## Run specific scenarios

```bash
# List available task IDs.
rvcbench tasks --suite core-v1

# Export Chinese voice-cloning inputs.
rvcbench prompts --suite core-v1 --tasks chinese --output prompts-chinese/

# Generate one <id>.wav per prompt with your model in outputs/chinese/.

# Score the generated audio.
rvcbench score --suite core-v1 --tasks chinese \
  --generated outputs/chinese --output results/chinese --device cuda
```

Read `results/chinese/submission.json` for per-task scores.

| Argument | What to pass |
| --- | --- |
| `--suite` | `onboarding-v1`, `core-v1` or `full-v1` |
| `--tasks` | IDs from the table below, separated by spaces; e.g. `chinese french background` |
| `--output` | A new output directory |
| `--generated` | Directory containing your model's WAV files |
| `--device` | `cpu`, `cuda` or `cuda:N`, e.g. `cuda:1`; default `cpu` |

Use the same `--suite` and `--tasks` for export and scoring. Omit `--tasks` to run the whole suite.
For all arguments, see the [CLI reference](cli.md).

## Choose a suite

| `--suite` | Audio files to generate | Use |
| --- | ---: | --- |
| `onboarding-v1` | 52 | Quick integration check: `libritts` (16), `vctk` (16), `robotcall` (20) |
| `core-v1` | 480 | Small evaluation across the scenarios below |
| `full-v1` | 12,724 | Larger evaluation; excludes AdvNoise and AntiProtect |

Counts above are for a whole suite. Selecting tasks reduces the workload.
These suites are evaluation previews. To reproduce the paper's reported results, use [v1](versions.md).

## Tasks

Each row is one accepted `--tasks` value for `core-v1`. The Full column shows availability in `full-v1`.
Counts are WAV files **you generate**, excluding automatically added tasks. `Auto` means RVCBench
creates that task's audio during scoring; ffmpeg is required.

| `--tasks` value | Paper scenario | Core | Full | Automatically added |
| --- | --- | ---: | ---: | --- |
| `audioshift` | AudioShift: accent, gender and age | 24 | 2000 | — |
| `textshift-standard` | TextShift: standard-text control | 24 | 200 | — |
| `textshift-hallucination` | TextShift: hallucination prompts | 24 | 200 | `textshift-standard` |
| `textshift-scam` | TextShift / Expression: scam scripts | 20 | 200 | `textshift-scam-standard` |
| `textshift-scam-standard` | Expression: normal-text control | 20 | 100 | — |
| `english-libritts` | Multilingual: English | 24 | 2000 | — |
| `chinese` | Multilingual: Chinese | 24 | 1998 | — |
| `crosslingual` | Multilingual: English ↔ Chinese | 24 | 650 | — |
| `french` | Multilingual: French | 24 | 2000 | — |
| `longtext` | LongContext: long target text | 20 | 20 | — |
| `longaudio` | LongContext: long reference audio | 24 | 156 | — |
| `background-clean` | PassiveNoise: clean background control | 20 | 800 | — |
| `background` | PassiveNoise: background noise | 20 | 800 | `background-clean` |
| `multispeaker-clean` | PassiveNoise: single-speaker control | 24 | 800 | — |
| `multispeaker` | PassiveNoise: competing speakers | 24 | 800 | `multispeaker-clean` |
| `adv-clean` | AdvNoise / AntiProtect: clean control | 20 | Unavailable | — |
| `adv-gaussian` | AdvNoise: Gaussian noise | 20 | Unavailable | `adv-clean` |
| `adv-spec` | AdvNoise: SPEC | 20 | Unavailable | `adv-clean` |
| `adv-safespeech` | AdvNoise: SafeSpeech | 20 | Unavailable | `adv-clean` |
| `adv-pop` | AdvNoise: POP | 20 | Unavailable | `adv-clean` |
| `adv-enkidu` | AdvNoise: Enkidu | 20 | Unavailable | `adv-clean` |
| `antiprotect-spec` | AntiProtect: DEMUCS on SPEC | 20 | Unavailable | `adv-clean` |
| `compression-mp3-64k` | Compression: MP3 64 kbps | Auto | Auto | `audioshift` |
| `compression-aac-64k` | Compression: AAC 64 kbps | Auto | Auto | `audioshift` |
| `compression-opus-24k` | Compression: OPUS 24 kbps | Auto | Auto | `audioshift` |
| `compression-mp3-32k` | Compression: MP3 32 kbps | Auto | Auto | `audioshift` |
| `compression-aac-32k` | Compression: AAC 32 kbps | Auto | Auto | `audioshift` |
| `compression-opus-16k` | Compression: OPUS 16 kbps | Auto | Auto | `audioshift` |
| `compression-narrowband` | Compression: telephone band | Auto | Auto | `audioshift` |

Examples:

```bash
# Chinese and French: 24 + 24 = 48 generations.
rvcbench prompts --suite core-v1 --tasks chinese french --output prompts-languages/

# Background noise: 20 noisy + 20 clean = 40 generations.
rvcbench prompts --suite core-v1 --tasks background --output prompts-noise/

# MP3 compression: generate 24 audioshift prompts; scoring creates the MP3 versions.
rvcbench prompts --suite core-v1 --tasks compression-mp3-64k --output prompts-mp3/
```

Generate every exported prompt, including automatically added controls. Selecting `audioshift` alone
does not run compression tasks. `onboarding-v1` accepts only `libritts`, `vctk`, and `robotcall`.

## Metrics and results

| Tasks | Reported metrics |
| --- | --- |
| Most tasks | `sim`, `speechmos`, `wer`, `mcd` |
| `textshift-scam`, `textshift-scam-standard` | `sim`, `speechmos`, `wer`, `sva`, `emotion` |
| `compression-*` | `stoi`, `mcd`, `sim`, `wer` |

Metric definitions: [SIM, SVA, WER, MOS, MCD, STOI and emotion](metrics.md#metric-definitions).
The suite chooses the metrics; `score` has no `--metrics` argument.

| Result | Where to find it |
| --- | --- |
| Per-task means and completion status | `submission.json` → `tasks` → task ID |
| Change from the clean control | Task's `relative_change_percent` |
| Accent, gender and age breakdown | `audioshift` → `group_means` |
| Other group breakdowns | `english-libritts`: gender; `crosslingual`: direction; `longaudio`: reference duration; `background`: noise; `multispeaker`: SNR/interferer; `textshift-scam`: scam type |
| Per-sample scores, errors and coverage | `<task>/run_manifest.json` |
| 95% bootstrap intervals | Metric reports in the task directory |

`complete` means every selected task and automatically added dependency succeeded. Incomplete tasks
have no `means`. Resume with the original command plus `--resume`. Compare models using the same
suite, task selection and scoring setup.

## Coverage limits

- Deepfake detection and the audio-LLM emotion-alignment judge are not included.
- `core-v1` includes AdvNoise and AntiProtect; `full-v1` does not.
- For shared tasks, core pairs are included in full with the same IDs and inputs.

## Full suite

```bash
rvcbench prompts --suite full-v1 --output prompts-full/
# Generate all exported prompts in outputs/my-model/.
rvcbench score --suite full-v1 --generated outputs/my-model \
  --output results-full/my-model --device cuda
```

The full suite needs 12,724 generated WAVs and scores 26,724 files including compression variants.
You can still select a smaller set with `--tasks`.

To use a local dataset copy:

```bash
hf download Nanboy/RVCBench --repo-type dataset --revision a932bd08d6858f14bdda52356dde5a7b771f0245 \
  --local-dir rvcbench-data
rvcbench prompts --suite full-v1 --tasks chinese --output prompts-local/ --data-root rvcbench-data
```

Pass the same `--data-root rvcbench-data` to `score`. Downloads require about 12.6 GB for the whole
snapshot; use `hf auth login` if the Hub rate-limits requests.

## How it is built

<details>
<summary>Sampling, input verification and comparison details</summary>

- Core selections are fixed before looking at model outputs, using seeded SHA-256 ranking
  (seed 20261002). Audio and metadata are checked against the suite's recorded hashes.
- Noisy and protected tasks use matched clean controls. Compression tasks compare processed
  generated audio with its unprocessed version.
- Selected runs record `selected_tasks`, `parent_suite_sha256` and `suite_sha256`. Different task
  selections cannot share a resumed run or a ranked comparison. Selecting all tasks keeps the
  whole-suite identity.
- Protected references come from `Protected_LibriTTS/`. POP is called `em` in the protection code.

</details>

## Things to know when reading results

<details>
<summary>Task-specific interpretation</summary>

- Enkidu and DEMUCS references are 16 kHz; clean LibriTTS references are 24 kHz.
  Their comparison includes the bandwidth difference.
- Scam scripts have no same-text target recording. SIM and emotion compare with the speaker
  reference; MCD is omitted. WER checks whether the generated speech follows the script.
- In full English-LibriTTS, 19 speakers have no gender annotation and appear as `unknown`.
- Multi-line text is preserved in `prompts.jsonl`; TSV/LST exports replace line breaks with spaces.
  WER treats these line breaks as spaces.

</details>
