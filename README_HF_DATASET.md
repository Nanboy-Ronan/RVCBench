---
pretty_name: RVCBench
license: cc0-1.0
language:
  - en
  - zh
  - fr
task_categories:
  - text-to-speech
  - automatic-speech-recognition
tags:
  - voice-cloning
  - text-to-speech
  - speaker-privacy
  - audio-protection
  - adversarial-audio
  - audio-deepfake
  - speaker-verification
  - speech-synthesis
  - robustness
  - benchmark
  - zero-shot-voice-cloning
configs:
  - config_name: AISHELL1_dev
    data_files:
      - split: default
        path: AISHELL1_dev/metadata.parquet
  - config_name: Background_noise
    data_files:
      - split: default
        path: Background_noise/metadata.parquet
  - config_name: Bilingual_uedin
    data_files:
      - split: default
        path: Bilingual_uedin/metadata.parquet
  - config_name: CommonVoiceFR_dev
    data_files:
      - split: default
        path: CommonVoiceFR_dev/metadata.parquet
  - config_name: Libritts
    data_files:
      - split: default
        path: Libritts/metadata.parquet
  - config_name: Long_context
    data_files:
      - split: default
        path: Long_context/metadata.parquet
  - config_name: Multispeaker_libri
    data_files:
      - split: default
        path: Multispeaker_libri/metadata.parquet
  - config_name: VCTK
    data_files:
      - split: default
        path: VCTK/metadata.parquet
  - config_name: robotcall
    data_files:
      - split: default
        path: robotcall/metadata.parquet
  - config_name: vctk_text_robust
    data_files:
      - split: default
        path: vctk_text_robust/metadata.parquet
---

# RVCBench

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026%20Accepted-6842c2.svg)](https://arxiv.org/abs/2602.00443)

**Accepted to NeurIPS 2026.**

RVCBench is a benchmark dataset for studying robustness in voice cloning, text-to-speech, speaker privacy, audio protection, adversarial audio perturbations, and related audio generation pipelines.

Dataset page: https://huggingface.co/datasets/Nanboy/RVCBench
Code repository: https://github.com/Nanboy-Ronan/RVCBench
Paper: https://arxiv.org/abs/2602.00443

RVCBench is designed for evaluating how modern voice cloning (VC), TTS, and audio generation systems behave under clean prompts, protected prompts, and denoised protected prompts. It supports research on audio deepfake robustness, anti-spoofing, speaker verification resilience, privacy-preserving speech generation, and standardized benchmark evaluation.

Each subset is exposed as its own Hugging Face dataset configuration. Most subsets contain:

- `metadata.parquet`
- `audios/`

The canonical metadata stores one row per benchmark pair with columns such as:

- `speaker_id`
- `prompt_file_name`
- `prompt_text`
- `prompt_language`
- `target_file_name`
- `target_text`
- `target_language`
- `pair_id`
- `dataset_name`
- `data_split`

When available, training-oriented annotations are also preserved:

- `prompt_phonemes`, `prompt_tone`, `prompt_word2ph`
- `target_phonemes`, `target_tone`, `target_word2ph`

Some subsets include additional task-specific metadata, for example `spam_type` in `robotcall`.

## Available Configs

- `AISHELL1_dev`
- `Background_noise`
- `Bilingual_uedin`
- `CommonVoiceFR_dev`
- `Libritts`
- `Long_context`
- `Multispeaker_libri`
- `VCTK`
- `robotcall`
- `vctk_text_robust`

## Getting started

Follow the [public quickstart](https://github.com/Nanboy-Ronan/RVCBench#evaluate-your-model).
The Qwen quickstart downloads the selected speaker audio and generates a small
sample without loading the full evaluation stack. Use the
[run protocol](https://github.com/Nanboy-Ronan/RVCBench/blob/main/docs/run_protocol.md)
to evaluate, inspect coverage, or resume a run. Keep the dataset revision or input
hashes with any reported result; do not compare scores computed on different
successful subsets without reporting coverage.

## Intended Use

Use this dataset with the RVCBench codebase to run reproducible voice cloning robustness experiments across source audio, protected audio, denoised audio, and generated audio. Typical tasks include:

- benchmarking zero-shot voice cloning and TTS models;
- comparing audio protection methods such as SafeSpeech, Enkidu, EM, spectral perturbation, and Gaussian noise;
- measuring speaker similarity, intelligibility, perceptual quality, word error rate, and runtime;
- studying speaker privacy and audio deepfake robustness under adversarial perturbations.

## Citation

If you use RVCBench in your research, please cite:

```bibtex
@inproceedings{jin2026rvcbench,
  title   = {RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models},
  author  = {Jin, Ruinan and Liao, Xinting and Yu, Hanlin and Pandya, Deval and Li, Xiaoxiao},
  booktitle = {Advances in Neural Information Processing Systems},
  url     = {https://arxiv.org/abs/2602.00443},
  year    = {2026}
}
```

## Notes

- The `compression` directory is intentionally excluded from this dataset release.
- This repository is organized for direct browsing in the Hugging Face dataset viewer.
