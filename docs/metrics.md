# Automatic speech metrics

Use `rvcbench.metrics` to score your own voice-cloning outputs, with your own data. No benchmark suite
or model adapter is required. Install `rvcbench[eval]` and prepare the scorers:

```bash
python -m pip install "rvcbench[eval]"
rvcbench setup-scorers
```

See [installation](installation.md) for FFmpeg, CPU/GPU environments and downloads.

## Score one file

```python
from rvcbench import metrics

with metrics.Evaluator(["sim", "wer", "speechmos"], device="cpu") as evaluator:
    scores = evaluator.score(
        "generated.wav",
        reference="speaker_reference.wav",
        text="Hello there.",
        language="en",
    )
    print(scores)  # {'sim': ..., 'wer': ..., 'speechmos': ...}
```

Choose `device="cuda"` to use a GPU. `metrics.available()` lists the seven supported names.

## Score every metric

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

`reference` is a recording of the intended speaker. `target` is a recording of the **same text** as the
generated audio; it is used for MCD and STOI. These can be different recordings. When `target` is omitted,
MCD/STOI use `reference` for backwards compatibility, so only omit it when that recording has the same text.
If no same-text recording exists, select the other metrics rather than calculating MCD/STOI on unrelated words.
Emotion agreement uses `reference`; choose a reference with the intended expression.

## Score a dataset

Reuse one evaluator to avoid reloading models for each file. For example, create `audio.jsonl`:

```json
{"id": "sample-1", "generated": "outputs/1.wav", "reference": "refs/1.wav", "text": "Hello there.", "language": "en"}
{"id": "sample-2", "generated": "outputs/2.wav", "reference": "refs/2.wav", "text": "Good morning.", "language": "en"}
```

Then run:

```python
import json
from rvcbench import metrics

with metrics.Evaluator(["sim", "wer", "speechmos"], device="cpu") as evaluator:
    with open("audio.jsonl") as source, open("scores.jsonl", "w") as output:
        for line in source:
            item = json.loads(line)
            sample_id = item.pop("id")
            scores = evaluator.score(**item)
            output.write(json.dumps({"id": sample_id, **scores}) + "\n")
```

Paths in this example are relative to the directory where the script runs. Each scorer loads on first use
and stays loaded until the context closes. Errors are raised with their cause; invalid scores are not
silently included in an average. Scoring restores the caller's Python, NumPy, CPU and selected CUDA RNG
states and cuDNN flags, including after an error. Run concurrent training/scoring in separate processes,
as RNG state is temporarily changed during a call.

## One-line functions

```python
metrics.speaker_similarity("generated.wav", "speaker_reference.wav")
metrics.word_error_rate("generated.wav", "Hello there.", language="en")
metrics.mos("generated.wav")
metrics.mel_cepstral_distortion("generated.wav", "same_text_recording.wav")
metrics.stoi("generated.wav", "same_text_recording.wav")
```

These functions load and release the required model per call. Use an `Evaluator` for repeated scoring.

## Metric definitions

| Name | Definition | Compared with |
| --- | --- | --- |
| `sim` | Cosine similarity of ECAPA-TDNN speaker embeddings (SpeechBrain/VoxCeleb). Higher is better. | `reference` |
| `sva` | Boolean speaker verification decision from the same model. | `reference` |
| `wer` | Whisper medium transcription error rate, lowercased with ASCII punctuation removed; Chinese is segmented with jieba. Lower is better. | `text` |
| `speechmos` | UTMOS22 strong predicted naturalness, nominally 1–5. Higher is better. | No reference |
| `mcd` | DTW-aligned mel-cepstral distortion (pymcd). Lower is better. | `target`, or same-text `reference` |
| `stoi` | Short-time objective intelligibility, resampled to 16 kHz and trimmed to the shorter recording. Higher is better. | `target`, or same-text `reference` |
| `emotion` | Boolean equality of emotion labels predicted by SpeechBrain wav2vec2/IEMOCAP. | `reference` |

`language` is passed to Whisper as a hint (`"en"`, `"zh"`, `"fr"`, etc.); without it Whisper detects the language.
Automatic MOS is a model prediction, and emotion is agreement within the recognizer's four labels.
These are the seven public API metrics; the research workflows contain additional specialized measurements.

## Relationship to the benchmark

The API and `rvcbench score` use the same scorer implementations and pinned models. A suite supplies
its target recordings and expected text automatically. Standalone calls let you choose your own references.
Matching files, normalization and seeds give matching metric definitions; a suite seeds Whisper per sample,
so fallback sampling can differ from the standalone API's default seed of 42. You can set `seed=` on an
`Evaluator`. Floating-point differences between hardware and library versions can also occur.

Suite reports additionally record scoring fingerprints and coverage. Use the
[dataset workflow](adding_a_model.md) for comparable task reports, interruption recovery and multi-model tables.
