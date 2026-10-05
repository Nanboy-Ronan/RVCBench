---
title: "RVCBench Python metrics API reference"
description: "Look up Evaluator arguments, score inputs, return values and convenience functions for speaker similarity, WER, speech quality, MCD, STOI and emotion."
---

# Python API reference

For a worked example, start with [Score your own audio](metrics.md). This page describes the public
metrics API in RVCBench 2.1.x. Import it with:

```python
from rvcbench import metrics
```

## Available metrics

```python
metrics.available()
# ['sim', 'sva', 'wer', 'speechmos', 'mcd', 'stoi', 'emotion']
```

See [metric definitions](metrics.md#metric-definitions) for scoring directions and required inputs.

## Evaluator

```text
metrics.Evaluator(
    metrics=("sim", "wer", "speechmos"),
    *,
    device="cpu",
    seed=42,
    logger=None,
)
```

| Argument | Meaning |
| --- | --- |
| `metrics` | A list/tuple of metric names, a single name, or `"all"`. Unknown names and empty selections raise `ValueError`. |
| `device` | PyTorch device, such as `"cpu"`, `"cuda"` or `"cuda:0"`. |
| `seed` | Seed applied during each scoring call; default `42`. |
| `logger` | Optional Python logger for scoring messages. |

Scorer models load on first use and are reused. Use the evaluator as a context manager so its resources
are released, including when a call raises an error:

```python
with metrics.Evaluator("speechmos", device="cpu") as evaluator:
    scores = evaluator.score("generated.wav")
```

### score

```text
evaluator.score(generated, *, reference=None, target=None, text=None, language=None)
```

| Argument | Type / meaning |
| --- | --- |
| `generated` | Audio file path (`str` or `pathlib.Path`). |
| `reference` | Recording of the intended speaker; also the comparison recording for emotion. |
| `target` | Same-text recording for MCD/STOI. If omitted, these metrics use `reference`. |
| `text` | Expected transcript, required by `wer`. |
| `language` | Optional Whisper language hint, such as `"en"` or `"zh"`. |

Returns a dictionary with exactly the selected metric names. Values are Python floats; `sva` and
`emotion` are booleans. Missing required arguments raise `ValueError`; missing files raise
`FileNotFoundError`. Scorer/dependency errors propagate to the caller, and invalid metric values raise
an error rather than silently entering an average.

### close

`evaluator.close()` releases the loaded scorers. The context manager calls it automatically.
Python, NumPy and relevant PyTorch RNG states are restored after scoring; run concurrent training and
scoring in separate processes because global RNG state is temporarily changed during a call.

## Convenience functions

Each function scores one file and then releases its scorer. For repeated calls, prefer `Evaluator`.

```text
metrics.speaker_similarity(generated, reference, *, device="cpu")
metrics.word_error_rate(generated, text, *, language=None, device="cpu")
metrics.mos(generated, *, device="cpu")
metrics.mel_cepstral_distortion(generated, reference)
metrics.stoi(generated, reference)
```

These are signature summaries: `*` marks keyword-only parameters. For MCD/STOI, the convenience
function's `reference` must be a recording of the same text. Use `Evaluator` for SVA and emotion.

## Model adapters

`from rvcbench import VoiceCloningAdapter` exposes the external adapter base class. Implement
`clone(self, *, text, reference_audio, reference_text, language)` and return a mono waveform and sample
rate. Optional `load()` and `unload()` methods control model lifetime. See the
[adapter contract and full example](adding_a_model.md#route-1-external-adapter).
