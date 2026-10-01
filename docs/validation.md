# Validation coverage

The repository exposes 32 model integrations. Currently, 27 have completed
real generation and scoring on a fixed subset. This is integration validation;
full paper-table and bitwise generation reproduction remain unestablished.

Paper membership refers to the 18-model arXiv v2 evaluation. Experimental
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

Other outstanding work includes remaining protection-method production, historical auxiliary
metric replay, timing comparability, and final website/Hugging Face integration.
These are retained in [the reproduction plan](../reproduction/plan.json).

Run records store sample hashes, assets, runtime provenance and metric coverage.
Local debug runs and detailed audit snapshots are excluded from Git. Use
[the run guide](run_protocol.md) to generate your own reports.
