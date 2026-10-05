---
title: "Voice cloning integration validation coverage"
description: "Review RVCBench model integration checks and fixed-subset validation. Distinguish tested generation and scoring from full benchmark or paper reproduction."
---

# Validation coverage

The repository exposes 32 model integrations. Currently, 27 have completed
real generation and scoring on a fixed subset. This is integration validation;
full paper-table and bitwise generation reproduction remain unestablished.

Paper membership refers to the 18 models of the paper's main results (arXiv v3). Experimental
integrations remain in scope even when assets or compatible protocols are missing.

| Model | Paper | Subset | Generation / scoring |
| --- | --- | --- | --- |
| BertVITS2 | No | libritts16_v1 | Pending |
| Qwen3-TTS | Yes | libritts16_v1 | Complete |
| Qwen3-Omni | No | libritts16_v1 | Pending |
| FireRedTTS-2 | No | libritts_short10_legacy_v1 | Complete |
| VoxCPM | No | libritts16_v1 | Complete |
| F5-TTS | Yes | libritts16_v1 | Complete |
| MaskGCT | Yes | libritts16_v1 | Complete |
| OpenVoice V2 | Yes | libritts16_v1 | Complete |
| Coqui XTTS-v2 | Yes | libritts16_v1 | Complete |
| IndexTTS | Yes | libritts16_v1 | Complete |
| ZipVoice | Yes | libritts16_v1 | Complete |
| FishSpeech | Yes | libritts16_v1 | Complete |
| Fish Audio S2 (in-proc) | No | libritts16_v1 | Complete |
| Fish Audio S2 (server) | No | libritts16_v1 | Complete |
| CosyVoice / 2 | Yes | libritts16_v1 | Complete |
| Higgs Audio | Yes | libritts16_v1 | Complete |
| Higgs TTS 3 | No | libritts16_v1 | Pending |
| SparkTTS | Yes | libritts16_v1 | Complete |
| VALL-E | No | libritts16_v1 | Pending |
| StyleTTS 2 | Yes | libritts16_v1 | Complete |
| GLM-TTS | Yes | libritts16_v1 | Complete |
| GlowTTS | No | libritts16_v1 | Pending |
| Kimi Audio | No | libritts16_v1 | Complete |
| MGM-Omni | Yes | libritts16_v1 | Complete |
| MOSS TTSD | Yes | libritts16_v1 | Complete |
| MOSS-TTS | No | libritts16_v1 | Complete |
| dots.tts | No | libritts16_v1 | Complete |
| ZONOS2 | No | vctk16_v1 | Complete |
| PlayDiffusion | Yes | libritts16_v1 | Complete |
| Bark Voice Clone | No | libritts16_v1 | Complete |
| OZSpeech | Yes | libritts16_v1 | Complete |
| VibeVoice | Yes | libritts16_v1 | Complete |

## Pending integrations

| Integration | Required before real subset validation |
| --- | --- |
| BERT-VITS2 | A trained reference-conditioned checkpoint and compatible native inference. The current closed-set speaker model cannot substantiate zero-shot cloning. |
| Qwen3-Omni | A complete checkpoint and a verified reference-speaker cloning mechanism. |
| Higgs TTS 3 | A compatible speech-generation service with the configured model and reference-audio support. |
| Native VALL-E | A checkpoint compatible with lifeiteng/vall-e. The validated Amphion variant has a separate configuration. |
| GlowTTS | Model assets and a validated wrapper implementing the intended speaker-conditioning protocol. |

GR seeded-noise reconstruction, DNS64 and Enkidu cohort production have been
validated on the fixed LibriTTS subset. Enkidu completed the full 2,000-pair
training cohort and Qwen3 generation/core scoring for 16 selected outputs. Its
audio-only protocol differs from historical protection output and does not
establish historical equivalence.

The Bark, FireRedTTS2, VoxCPM, IndexTTS, MaskGCT, XTTS, ZipVoice, SparkTTS, CosyVoice,
OpenVoice and StyleTTS2 direct-backend migrations have contract checks,
including explicit seed validation and failure propagation. IndexTTS and
MaskGCT also have worker timeout, request matching and cleanup checks. Native generation
of the other migrated frozen subsets remains pending. OpenVoice additionally
completed all 16 current native CPU generation requests with matching frozen
inputs, valid WAV/hash/seed receipts and matching recorded source hashes. Its MCD/WER/SIM scoring is complete for all 16 pairs, and a second fresh CPU
generation matches every WAV byte. Historical comparison fails the scorer/runtime
consistency check; GPU/full-paper equivalence remains unproven. StyleTTS2 also
completed two fresh CPU generations of all 16 frozen samples with byte-identical
WAVs and full MCD/WER/SIM scoring. All 13 constructed native modules load complete
finite checkpoint state, and corrupted learned-weight checks fail explicitly.
Its historical GPU scorer/runtime fingerprints also differ, so historical/full-paper
equivalence remains unproven. FireRedTTS2 completed native CPU generation and
MCD/WER/SIM scoring for all 10 frozen short10/retry20 pairs with row/CSV/aggregate
checks. This explicit variant preserves its historical input and asset hashes;
historical GPU scorer/runtime fingerprints differ, and native CPU repeatability
has not yet been measured. These validations apply to each migration's recorded
source revision; final combined-source native revalidation remains outstanding.
The table above retains the earlier generation/scoring campaigns.

Other outstanding work includes remaining protection-method production, historical auxiliary
metric replay, controlled timing campaigns across all runtimes, and final
website/Hugging Face integration.
These are retained in [the reproduction plan](../reproduction/plan.json).

Run records store sample hashes, assets, runtime provenance and metric coverage.
Local debug runs and detailed audit snapshots are excluded from Git. Use
[the run guide](run_protocol.md) to generate your own reports.

## Check the source version of a retained run

```bash
python -m rvcbench.benchmark.cli audit-source results/my_run/timestamped_directory
```

This read-only command compares each recorded generation source hash with the
current file. Use `--root /path/to/checkout` for another checkout and
`--output results/source_audit.json` to save the report. A changed or missing
file, unavailable provenance or invalid recorded hash fails the check. Absolute
upstream paths are checked at their recorded locations.

A match covers only recorded files; it does not verify complete current import
coverage, dependencies, weights, audio or metrics. A mismatch marks a source
version difference and does not invalidate historical results. The table above
records completed subset campaigns, rather than a guarantee for every later
source revision. Qwen3 and F5 direct campaigns were run before subsequent shared
runner changes; native revalidation of the final combined refactor remains
outstanding.
