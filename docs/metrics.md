# Metrics API

`rvcbench.metrics` computes the speech metrics of the benchmark in your own code, with the same models
and definitions as `rvcbench score`. Use it when you want RVCBench-comparable numbers inside an existing
evaluation script, without running a suite.

```bash
pip install "rvcbench[eval] @ git+https://github.com/Nanboy-Ronan/RVCBench@main"
rvcbench setup-scorers        # once: download the metric models
```

## One file

```python
from rvcbench import metrics

metrics.speaker_similarity("generated.wav", "reference.wav")                       # 0.61
metrics.word_error_rate("generated.wav", "The text it should say.", language="en")  # 0.05
metrics.mos("generated.wav")                                                       # 4.12
metrics.mel_cepstral_distortion("generated.wav", "same_text_recording.wav")         # 6.02
metrics.stoi("generated.wav", "same_text_recording.wav")                            # 0.93
```

Each call loads its model and releases it afterwards. For more than a few files, use an `Evaluator`.

## Many files

```python
from rvcbench import metrics

with metrics.Evaluator(["sim", "wer", "speechmos"], device="cuda") as evaluator:
    for item in items:
        scores = evaluator.score(item.generated, reference=item.reference, text=item.text, language="en")
        # {'sim': 0.61, 'wer': 0.05, 'speechmos': 4.12}
```

The evaluator loads each metric model once, on first use, and keeps it until `close()` or the end of the
`with` block.

## Metrics

| Name | Measures | Compares the generated audio with |
| --- | --- | --- |
| `sim` | Speaker similarity: cosine of ECAPA-TDNN speaker embeddings (SpeechBrain, VoxCeleb). Higher is better. | `reference`: a recording of the target speaker |
| `sva` | Speaker verification: `True` when the same model accepts both recordings as one speaker. | `reference` |
| `wer` | Word error rate of the Whisper medium transcript, after lower-casing and removing punctuation; Chinese is segmented with jieba. Lower is better. | `text` |
| `speechmos` | Predicted naturalness, 1 to 5 (UTMOS22 strong via SpeechMOS). Higher is better. | nothing |
| `mcd` | Mel-cepstral distortion, DTW-aligned (pymcd). Lower is better. | `reference`: a recording of the same text |
| `stoi` | Short-time objective intelligibility. Higher is better. | `reference`: a recording of the same text |
| `emotion` | `True` when the emotion recognized in both recordings matches (wav2vec2 on IEMOCAP, SpeechBrain). | `reference` |

`language` (for example `"en"`, `"zh"` or `"fr"`) is passed to Whisper as a hint; without it Whisper
detects the language. `device` is a torch device such as `"cpu"` or `"cuda"`.

## Agreement with `rvcbench score`

The scorers are the ones the suites use, so the values match `rvcbench score` for the same files, up to
floating-point differences between GPUs and library versions. The one exception is the seed: a suite seeds
Whisper per sample, which only matters when its decoding falls back to sampling. Scores are tested to agree
across torch 2.6 and 2.9 and numpy 1.26 and 2.2.
