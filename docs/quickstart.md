# Getting started with RVCBench

[中文](quickstart_zh.md) · [All documentation](README.md)

RVCBench has two entry points. **Use the Python metrics API** if you already have generated audio and
your own data. **Use a benchmark suite** if you want us to supply the evaluation inputs and produce
task-level reports. Both work with audio from any voice cloning model, including hosted APIs.

## 1. Install and prepare the scorers

Use Linux and Python 3.10–3.12. These commands create a CPU scoring environment; no GPU or repository
checkout is needed. Install FFmpeg with your system package manager first (on Ubuntu/Debian,
`sudo apt-get install ffmpeg`).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "rvcbench[eval]==2.1.0"
rvcbench doctor --eval --imports
rvcbench setup-scorers
```

The last command downloads and verifies the scorer models. Allow several GB of cache space and time
for the initial downloads; subsequent runs reuse them. All examples below run from your working
directory, with the environment activated. For GPU setup or installation errors, see
[installation](installation.md).

## 2A. Score your own audio

Prepare three inputs:

| Input | What it is |
| --- | --- |
| `generated.wav` | Speech produced by your model |
| `speaker_reference.wav` | A recording of the speaker you want to clone; it may say different words |
| Expected text | The words you asked your model to generate |

Use real audio files, preferably mono WAV. Save the following as `score_audio.py`, replacing the
filenames and expected text with yours:

```python
import json
from rvcbench import metrics

with metrics.Evaluator(["sim", "wer", "speechmos"], device="cpu") as evaluator:
    scores = evaluator.score(
        "generated.wav",
        reference="speaker_reference.wav",
        text="Hello there.",
        language="en",
    )

print(json.dumps(scores, indent=2))
```

```bash
python score_audio.py
```

The result is a dictionary of scalar scores, for example this **illustrative, non-benchmark** output:

```json
{"sim": 0.72, "wer": 0.1, "speechmos": 3.9}
```

Higher `sim` means more similar speaker embeddings. `wer` is an error rate (`0.1` means 10%, and it can
exceed 1); lower is better. `speechmos` estimates speech naturalness; higher is better. Compare models
on the same inputs with the same scoring environment. These metrics do not combine into a universal
single quality score.

For **all seven metrics**, use the same import and change the evaluation block to:

```python
with metrics.Evaluator("all", device="cpu") as evaluator:
    scores = evaluator.score(
        "generated.wav",
        reference="speaker_reference.wav",
        target="same_text_recording.wav",
        text="Hello there.",
        language="en",
    )
```

The extra `target` recording must contain the **same words** as `generated.wav`; it is used for MCD
and STOI. Emotion compares against `reference`, so choose one with the intended expression. If you do
not have a same-text recording, select the metrics your inputs support. Reuse one `Evaluator` in a
loop to score many files without reloading models. See [metric definitions and batch scoring](metrics.md).

## 2B. Evaluate a model with our data

### Export prompts

Start with the 52-utterance onboarding suite:

```bash
rvcbench prompts --suite onboarding-v1 --output prompts/
```

This downloads the selected data and creates `prompts.jsonl`, `prompts.tsv`, `prompts.lst`, and
reference WAV files under `prompts/`. Your model needs only the exported reference audio, reference
transcript and requested text. Evaluation target recordings are kept out of the prompt export.

### Generate with your model

Run inference in your model's environment. If its script accepts ZipVoice or Seed-TTS-style batch
lists, use `prompts/prompts.tsv` or `prompts/prompts.lst`; see [batch inference](adding_a_model.md#batch-inference-scripts).
Otherwise adapt this loop to your own model API:

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

Return a mono waveform and its actual sample rate. Write one `<id>.wav` per prompt, using the exported
`id` unchanged. Do not rename outputs sequentially or copy reference/target audio as model predictions.
RVCBench handles data preparation and scoring; you supply this inference step.

### Score and inspect results

Return to the RVCBench environment and use the **same suite** as the export:

```bash
rvcbench score --suite onboarding-v1 \
  --generated outputs/my-model --output results/my-model --device cpu
```

Open `results/my-model/submission.json`, or print a compact summary:

```python
import json
from pathlib import Path

report = json.loads(Path("results/my-model/submission.json").read_text())
print(report["model"], report["status"])
for task, result in report["tasks"].items():
    print(task, result["status"], result.get("means", {}))
    if result["status"] != "complete":
        print("Coverage:", result["coverage"])
        print("Failures:", result["failures"])
```

`complete` means all required samples and metrics succeeded. `partial` means some inputs or scores
failed; inspect each task's `run_manifest.json` for per-sample errors, including scorer errors.
Completed tasks have `means`; incomplete tasks do not get a misleading successful-only mean.
The task folders also contain detailed metric reports and uncertainty estimates. There is no single
overall score that substitutes for the individual dimensions.

Fix missing/invalid outputs, then continue with:

```bash
rvcbench score --suite onboarding-v1 \
  --generated outputs/my-model --output results/my-model --device cpu --resume
```

Keep the same suite, model name, input directory and output directory. Matching successful scores are
reused; missing, changed or failed samples are checked again.

### Compare models and expand coverage

Generate the same prompts with a second model, then score both into a fresh directory:

```bash
rvcbench score --suite onboarding-v1 \
  --generated outputs/model-a outputs/model-b --output results/comparison --device cpu
```

Read `results/comparison/comparison.md` for the side-by-side table, or use `.csv` and `.json` for analysis.
Each model also has its own `submission.json` under `results/comparison/<model>/`. To compare existing
reports from the same suite and scoring environment:

```bash
rvcbench compare results/model-a/submission.json results/model-b/submission.json --output results/compare
```

After onboarding, choose a larger suite and **export its prompts again**:

| Suite | Outputs to generate per model | Purpose |
| --- | ---: | --- |
| `onboarding-v1` | 52 | Verify your integration |
| `core-v1` | 480 | Broader languages, speakers and recording conditions, including protected references |
| `full-v1` | 12,724 | Larger datasets; use `core-v1` separately for protection tasks |

Both export and scoring must use that suite name. Larger suites require more data, inference and scoring
time. See [suite coverage and costs](core_suite.md). The suites are currently previews; paper reproduction
uses the separate [v1 codebase](versions.md).

## Next steps

- [Metrics API](metrics.md): metric meanings, required inputs and batch scoring.
- [Model evaluation](adding_a_model.md): output layouts, external APIs and adapters.
- [Installation](installation.md): GPU, cache locations, offline use and upgrading old results.
