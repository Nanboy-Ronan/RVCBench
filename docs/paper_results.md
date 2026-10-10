# Paper results


From the [paper](https://arxiv.org/abs/2602.00443) (arXiv v3). SIM, MOS, WER and MCD are on clean LibriTTS
prompts; the last three columns are speaker similarity on other datasets. **Bold** marks the best value
per column, and † marks models reported in the paper's appendix.

| Model | SIM ↑ | MOS ↑ | WER ↓ | MCD ↓ | VCTK SIM ↑ | Chinese SIM ↑ | Cross-lingual SIM ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-TTS | **0.61** | **4.39** | 0.05 | **5.79** | **0.62** | **0.72** | **0.67** |
| IndexTTS | **0.61** | 4.06 | 0.05 | 6.61 | 0.57 | **0.72** | **0.67** |
| dots.tts † | 0.60 | 4.17 | 0.06 | 6.11 | 0.57 | 0.68 | 0.61 |
| CosyVoice 2 | 0.58 | 4.37 | 0.05 | 6.02 | 0.58 | **0.72** | 0.65 |
| ZipVoice | 0.58 | 4.13 | 0.05 | 7.09 | 0.55 | 0.71 | 0.63 |
| MOSS-TTS v1.5 † | 0.57 | 4.32 | 0.06 | 6.66 | 0.51 | 0.68 | 0.65 |
| GLM-TTS | 0.57 | 4.08 | 0.09 | 6.41 | 0.57 | 0.69 | 0.66 |
| MaskGCT | 0.57 | 3.93 | 0.09 | 6.91 | 0.56 | 0.67 | 0.63 |
| Higgs TTS 3 † | 0.56 | 4.23 | 0.05 | 6.32 | 0.45 | 0.63 | 0.30 |
| F5-TTS | 0.56 | 3.99 | 0.12 | 6.96 | 0.54 | 0.70 | 0.65 |
| Higgs Audio | 0.56 | 4.30 | 0.25 | 6.06 | 0.52 | 0.58 | 0.54 |
| Fish Audio S2 † | 0.54 | 4.37 | **0.04** | 6.16 | 0.51 | 0.66 | 0.62 |
| MGM-Omni | 0.54 | 4.28 | 0.09 | 5.82 | 0.45 | 0.71 | 0.63 |
| PlayDiffusion | 0.51 | 4.15 | 0.05 | 8.06 | 0.43 | 0.44 | 0.46 |
| MOSS-TTSD | 0.49 | 4.10 | 0.38 | 7.09 | 0.44 | 0.44 | 0.44 |
| VibeVoice | 0.48 | 3.83 | 0.23 | 6.76 | 0.44 | 0.56 | 0.53 |
| FishSpeech | 0.47 | 4.37 | 0.17 | 6.47 | 0.43 | 0.61 | 0.57 |
| XTTS-v2 | 0.45 | 3.81 | 0.07 | 8.62 | 0.45 | 0.57 | 0.51 |
| Spark-TTS | 0.41 | 4.06 | 0.33 | 5.83 | 0.53 | 0.57 | 0.48 |
| OZSpeech | 0.39 | 3.21 | 0.06 | 6.87 | 0.25 | 0.00 | 0.17 |
| OpenVoice V2 | 0.24 | 4.30 | 0.07 | 7.06 | 0.39 | 0.43 | 0.30 |
| StyleTTS 2 | 0.23 | 4.30 | 0.05 | 6.81 | 0.24 | 0.11 | 0.21 |

<details>
<summary><b>Speaker similarity under anti-cloning protection</b> (LibriTTS)</summary>

<br>

A lower SIM under protection means the protection hides the speaker's voice better.

| Model | Clean | SafeSpeech | SPEC | Enkidu | Gaussian | POP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-TTS | 0.61 | 0.38 | 0.36 | 0.50 | 0.41 | 0.58 |
| IndexTTS | 0.61 | 0.35 | 0.32 | 0.47 | 0.39 | 0.57 |
| dots.tts † | 0.60 | 0.41 | 0.39 | 0.49 | 0.44 | 0.57 |
| CosyVoice 2 | 0.58 | 0.32 | 0.30 | 0.45 | 0.38 | 0.55 |
| ZipVoice | 0.58 | 0.29 | 0.26 | 0.44 | 0.26 | 0.54 |
| MOSS-TTS v1.5 † | 0.57 | 0.33 | 0.31 | 0.43 | 0.33 | 0.53 |
| GLM-TTS | 0.57 | 0.33 | 0.31 | 0.44 | 0.39 | 0.53 |
| MaskGCT | 0.57 | 0.30 | 0.28 | 0.41 | 0.31 | 0.53 |
| Higgs TTS 3 † | 0.56 | 0.48 | 0.48 | 0.49 | 0.34 | 0.53 |
| F5-TTS | 0.56 | 0.21 | 0.18 | 0.43 | 0.14 | 0.52 |
| Higgs Audio | 0.56 | 0.26 | 0.24 | 0.43 | 0.27 | 0.52 |
| Fish Audio S2 † | 0.54 | 0.32 | 0.30 | 0.43 | 0.34 | 0.52 |
| MGM-Omni | 0.54 | 0.18 | 0.17 | 0.32 | 0.23 | 0.49 |
| PlayDiffusion | 0.51 | 0.17 | 0.15 | 0.34 | 0.16 | 0.47 |
| MOSS-TTSD | 0.49 | 0.24 | 0.22 | 0.34 | 0.25 | 0.08 |
| VibeVoice | 0.48 | 0.27 | 0.25 | 0.37 | 0.28 | 0.45 |
| FishSpeech | 0.47 | 0.24 | 0.21 | 0.33 | 0.23 | **0.01** |
| XTTS-v2 | 0.45 | 0.26 | 0.24 | 0.31 | 0.24 | 0.41 |
| Spark-TTS | 0.41 | 0.13 | 0.11 | 0.14 | 0.06 | 0.36 |
| OZSpeech | 0.39 | 0.16 | 0.15 | 0.19 | 0.15 | 0.34 |
| OpenVoice V2 | 0.24 | 0.18 | 0.18 | 0.19 | 0.18 | 0.24 |
| StyleTTS 2 | 0.23 | **0.09** | **0.08** | **0.12** | **0.03** | 0.21 |

</details>

Results for compression, deepfake detectability, long-form and expressive speech are in the paper and on
the [website](https://nanboy-ronan.github.io/RVCBench/).


[Back to the repository](https://github.com/Nanboy-Ronan/RVCBench) · [Paper reproduction](versions.md)
