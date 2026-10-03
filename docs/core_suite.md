# RVCBench-Core v1

`core-v1` is a small, fixed version of the robustness evaluations in the
[RVCBench paper](https://arxiv.org/abs/2602.00443). It has 480 utterances to generate (20 of them
long-form) and reports the paper's metrics per evaluation. It ships with the `rvcbench` package; the audio comes from the
[Hugging Face dataset](https://huggingface.co/datasets/Nanboy/RVCBench) at a pinned revision.

```bash
rvcbench prompts --suite core-v1 --output prompts/
# synthesize every line of prompts/prompts.jsonl with your model
rvcbench score --suite core-v1 --generated my_outputs/ --model my-model --output results/my-model/ --device cuda
```

See [Evaluate your own model](adding_a_model.md#route-0-score-audio-you-generated-anywhere) for the file
format. `core-v1` is not yet a leaderboard suite: its scores become comparable leaderboard results once
baseline results for the paper's models are published.

## Tasks

| Task | Dimension | Paper evaluation | Data | Utterances | Metrics | Clean anchor | Reported by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `audioshift` | input | RVC-AudioShift/Demography | VCTK | 24 | SIM, MOS, WER, MCD |  | accent, gender, age_group |
| `textshift-standard` | input | RVC-TextShift/Standard prompts (reference for Hallucination) | VCTK | 24 | SIM, MOS, WER, MCD |  |  |
| `textshift-hallucination` | input | RVC-TextShift/Hallucination | VCTK (hallucination prompts) | 24 | SIM, MOS, WER, MCD | `textshift-standard` |  |
| `textshift-scam` | input | RVC-TextShift/Scam; RVC-Expression/Persuasion | Robocall scripts, VCTK voices | 20 | SIM, MOS, WER, SVA, EMC | `textshift-scam-standard` | spam_type |
| `textshift-scam-standard` | input | RVC-Expression/Normal VCTK context (reference for Scam) | Robocall scripts, VCTK voices | 20 | SIM, MOS, WER, SVA, EMC |  |  |
| `english-libritts` | generation | RVC-Multilingual/English-VC | LibriTTS | 24 | SIM, MOS, WER, MCD |  | gender |
| `chinese` | generation | RVC-Multilingual/Chinese-VC | AISHELL-1 | 24 | SIM, MOS, WER, MCD |  |  |
| `crosslingual` | generation | RVC-Multilingual/CrossLingual | EMIME bilingual | 24 | SIM, MOS, WER, MCD |  | direction |
| `french` | generation | RVC-Multilingual/French (Appendix Table 38) | Common Voice FR | 24 | SIM, MOS, WER, MCD |  |  |
| `longtext` | generation | RVC-LongContext/LongText | LibriSpeech-Long | 20 | SIM, MOS, WER, MCD |  |  |
| `longaudio` | generation | RVC-LongContext/LongAudio | LibriTTS | 24 | SIM, MOS, WER, MCD |  | reference_duration_bin |
| `background-clean` | perturbation | RVC-PassiveNoise/Background (clean references) | VoiceBank+DEMAND | 20 | SIM, MOS, WER, MCD |  |  |
| `background` | perturbation | RVC-PassiveNoise/Background | VoiceBank+DEMAND | 20 | SIM, MOS, WER, MCD | `background-clean` | noise |
| `multispeaker-clean` | perturbation | RVC-PassiveNoise/MultiSpeaker (clean references) | Multispeaker Libri | 24 | SIM, MOS, WER, MCD |  |  |
| `multispeaker` | perturbation | RVC-PassiveNoise/MultiSpeaker | Multispeaker Libri | 24 | SIM, MOS, WER, MCD | `multispeaker-clean` | snr, interferer |
| `adv-clean` | perturbation | RVC-AdvNoise (clean references) | LibriTTS + protected references | 20 | SIM, MOS, WER, MCD |  |  |
| `adv-gaussian` | perturbation | RVC-AdvNoise/Gaussian | LibriTTS + protected references | 20 | SIM, MOS, WER, MCD | `adv-clean` |  |
| `adv-spec` | perturbation | RVC-AdvNoise/Adversary (SPEC) | LibriTTS + protected references | 20 | SIM, MOS, WER, MCD | `adv-clean` |  |
| `adv-safespeech` | perturbation | RVC-AdvNoise/Adversary (SafeSpeech) | LibriTTS + protected references | 20 | SIM, MOS, WER, MCD | `adv-clean` |  |
| `adv-pop` | perturbation | RVC-AdvNoise/Adversary (POP) | LibriTTS + protected references | 20 | SIM, MOS, WER, MCD | `adv-clean` |  |
| `adv-enkidu` | perturbation | RVC-AdvNoise/Adversary (Enkidu) | LibriTTS + protected references | 20 | SIM, MOS, WER, MCD | `adv-clean` |  |
| `antiprotect-spec` | perturbation | RVC-AntiProtect/AntiProtection (DEMUCS on SPEC) | LibriTTS + protected references | 20 | SIM, MOS, WER, MCD | `adv-clean` |  |
| `compression-mp3-64k` | output | RVC-Compression/CodecCompression | `audioshift` outputs, processed | 24 | STOI, MCD, SIM, WER |  |  |
| `compression-aac-64k` | output | RVC-Compression/CodecCompression | `audioshift` outputs, processed | 24 | STOI, MCD, SIM, WER |  |  |
| `compression-opus-24k` | output | RVC-Compression/CodecCompression | `audioshift` outputs, processed | 24 | STOI, MCD, SIM, WER |  |  |
| `compression-mp3-32k` | output | RVC-Compression/CodecCompression | `audioshift` outputs, processed | 24 | STOI, MCD, SIM, WER |  |  |
| `compression-aac-32k` | output | RVC-Compression/CodecCompression | `audioshift` outputs, processed | 24 | STOI, MCD, SIM, WER |  |  |
| `compression-opus-16k` | output | RVC-Compression/CodecCompression | `audioshift` outputs, processed | 24 | STOI, MCD, SIM, WER |  |  |
| `compression-narrowband` | output | RVC-Compression/NarrowBand | `audioshift` outputs, processed | 24 | STOI, MCD, SIM, WER |  |  |

Metrics: SIM and SVA (ECAPA speaker similarity and verification), MOS (UTMOS via SpeechMOS), WER (Whisper
medium), MCD (DTW mel-cepstral distortion), EMC (emotion label agreement) and STOI. `submission.json` reports the
mean of each metric per task; each task's run directory also records 95% bootstrap intervals.

## How it is built

- **Selection.** `scripts/build_core_suite.py` selects pairs from the dataset metadata with seeded SHA-256
  ranking (seed 20261002), stratified as listed under "Reported by". It never looks at model outputs.
- **Pinned inputs.** The suite pins the dataset revision and the SHA-256 of every reference and target;
  `rvcbench prompts` and `rvcbench score` refuse to run when any input differs.
- **Paired anchors.** Each perturbed task uses the same targets as its clean anchor, and
  `submission.json` reports the percentage change of every shared metric against the anchor, as in the
  paper's Figures 4 and 7.
- **Post-processing.** The seven compression tasks re-encode the submitted `audioshift` outputs (MP3 and
  AAC at 64 and 32 kbps, Opus at 24 and 16 kbps, all at 24 kHz, and an 8 kHz 300-3400 Hz telephone
  channel) and compare each processed clone with the unprocessed one (paper Tables 43-44).
- **Protected references.** The AdvNoise and AntiProtect tasks use the protected references from the
  paper's runs, published under `Protected_LibriTTS/` in the dataset. POP is the method implemented as
  the error-minimizing protector (`em`) in the codebase.

## Things to know when reading results

- The Enkidu and DEMUCS references are 16 kHz while the clean LibriTTS references are 24 kHz, so part of
  their change against `adv-clean` comes from the lower bandwidth. This matches the paper's runs.
- In `textshift-scam` the target recording is the reference recording itself: the script text has no
  recording of its own. SIM and EMC there measure agreement with the reference, and MCD is not computed
  (paper Table 25). Returning the reference unchanged scores SIM 1.0 on this task; WER exposes it.
- Not included in v1: RVC-Detectability (deepfake detectors) and the audio-LLM emotion-alignment judge
  of RVC-Expression. Both are planned for `core-v1.1`.
