---
title: "Voice cloning evaluation quickstart"
description: "Install RVCBench, choose your tasks, generate audio and read per-task scores. Copy commands for scoring your own audio or benchmark data."
---

# Quickstart

| You have | Use |
| --- | --- |
| Generated audio and your own reference data | [Score your own audio](#2a-score-your-own-audio) |
| A model to evaluate on RVCBench data | [Export, generate, score](#2b-evaluate-a-model-with-our-data) |
| A parameter to look up | [CLI values and defaults](cli.md) or [Python API arguments](api.md) |

## 1. Install and prepare the scorers

Linux, Python 3.10–3.13, FFmpeg. CPU installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "rvcbench[eval]==2.2.0"
rvcbench setup-scorers
```

Install FFmpeg with your system package manager, e.g. `sudo apt-get install ffmpeg` on Ubuntu.
Scorer downloads need several GB of cache space. [GPU installation](installation.md).

## 2A. Score your own audio

Replace the file paths and text, then run this Python code:

```python
from rvcbench import metrics

with metrics.Evaluator(["sim", "wer", "speechmos"], device="cpu") as evaluator:
    scores = evaluator.score(
        "generated.wav",
        reference="speaker_reference.wav",
        text="Hello there.",
        language="en",
    )
print(scores)
```

| Argument | Values |
| --- | --- |
| `metrics` | `sim`, `sva`, `wer`, `speechmos`, `mcd`, `stoi`, `emotion`; pass one name, a list, or `"all"` |
| `device` | `cpu`, `cuda`, `cuda:N` |
| `language` | `en` (English), `zh` (Chinese), `fr` (French), or `None` for detection; other Whisper languages also work |

`sim`: higher speaker similarity is better. `wer`: lower error rate is better. `speechmos`: higher
predicted naturalness is better. MCD/STOI also need a same-text recording via `target=`.
[All inputs and defaults](api.md).

## 2B. Evaluate a model with our data

### Export prompts

```bash
rvcbench tasks --suite core-v1
rvcbench prompts --suite core-v1 --tasks chinese --output prompts/
```

| Argument | Values |
| --- | --- |
| `--suite` | `onboarding-v1` (52 prompts), `core-v1` (480), `full-v1` (12,724); counts are for whole suites |
| `--tasks` | Space-separated task IDs, e.g. `chinese french background`; omit for all tasks. [Complete list](cli.md#task-values) |
| `--output` | New or empty directory for prompt lists and reference audio |

The example exports 24 Chinese prompts. Some tasks automatically add clean controls; generate all
exported prompts. [Task counts and paper scenarios](core_suite.md#tasks).

### Generate with your model

Read `prompts/prompts.jsonl`, synthesize each entry, and save `outputs/my-model/<id>.wav`.
Replace `model.synthesize` below with your model's API:

```python
import json
from pathlib import Path
import soundfile as sf

# Load your voice cloning model here as `model`.
# The synthesize call below is an integration example: replace it with your model's API.
prompts = Path("prompts")
outputs = Path("outputs/my-model")
outputs.mkdir(parents=True, exist_ok=True)

for line in (prompts / "prompts.jsonl").read_text().splitlines():
    item = json.loads(line)
    waveform, sample_rate = model.synthesize(
        text=item["text"],
        reference_audio=str(prompts / item["reference_audio"]),
        reference_text=item["reference_text"],
        language=item["language"],
    )
    sf.write(outputs / f"{item['id']}.wav", waveform, sample_rate)
```

Use the exported `id` unchanged. Write mono audio at the model's actual sample rate.
Native batch scripts can use `prompts.tsv` or `prompts.lst`; see [batch formats](adding_a_model.md#batch-inference-scripts).

### Score and inspect results

```bash
rvcbench score --suite core-v1 --tasks chinese \
  --generated outputs/my-model --output results/my-model --device cpu
```

Use the same `--suite` and `--tasks` as export. Set `--device cuda` or `--device cuda:1` to score on a GPU.

| Read | Contains |
| --- | --- |
| `results/my-model/submission.json` | Per-task metrics, completion status and coverage |
| `results/my-model/chinese/run_manifest.json` | Per-sample scores and errors |

`complete`: all selected samples and metrics succeeded. `partial`: some failed; incomplete tasks have
no mean score. Fix the failed outputs, then repeat the score command with `--resume`.

### Compare models and expand coverage

Score several models with the same suite and tasks:

```bash
rvcbench score --suite core-v1 --tasks chinese \
  --generated outputs/model-a outputs/model-b --output results/comparison --device cpu
```

Read `results/comparison/comparison.md` (also `.csv` and `.json`).
To compare existing results:

```bash
rvcbench compare results/model-a results/model-b --output results/compare
```

To change tasks or suites, export new prompts and use new output directories.

## Next steps

- [CLI reference](cli.md): every argument, accepted value and default.
- [Scenarios](core_suite.md): task IDs, counts, controls and metrics.
- [Metrics API](api.md): Python arguments and input requirements.
