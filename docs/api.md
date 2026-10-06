---
title: "RVCBench Python metrics API reference"
description: "Look up Evaluator arguments, score inputs, return values and convenience functions for speaker similarity, WER, speech quality, MCD, STOI and emotion."
---

# Python API reference

Import the metrics API:

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

| Argument | Accepted values | Default |
| --- | --- | --- |
| `metrics` | `"sim"`, `"sva"`, `"wer"`, `"speechmos"`, `"mcd"`, `"stoi"`, `"emotion"`; a list/tuple of these; or `"all"` | `("sim", "wer", "speechmos")` |
| `device` | `"cpu"`, `"cuda"`, `"cuda:N"` (e.g. `"cuda:1"`), or a compatible `torch.device` | `"cpu"` |
| `seed` | Integer | `42` |
| `logger` | `logging.Logger` or `None` | `None` |

Example: `metrics.Evaluator(["sim", "wer"], device="cuda")`.
Unknown metric names and empty lists raise `ValueError`.

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

| Argument | Accepted values | Required / default |
| --- | --- | --- |
| `generated` | Audio file path: `str` or `pathlib.Path`, e.g. `"generated.wav"` | **Required** |
| `reference` | Speaker reference audio path: `str`, `Path` or `None` | Required for `sim`, `sva`, `emotion`; default `None` |
| `target` | Same-text target audio path: `str`, `Path` or `None` | Used by `mcd` and `stoi`; falls back to `reference` |
| `text` | Nonempty transcript string, e.g. `"Hello there."`, or `None` | Required for `wer`; default `None` |
| `language` | Whisper language code/name; benchmark languages are `"en"`, `"zh"`, `"fr"`; `None` or `"auto"` for detection | `None` |

`target` must contain the same words as `generated`. `reference` may contain different words.
For MCD/STOI, provide either `target` or a same-text `reference`.

```python
with metrics.Evaluator(["sim", "wer"], device="cpu") as evaluator:
    scores = evaluator.score(
        "generated.wav", reference="speaker.wav", text="Hello there.", language="en"
    )
```

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
