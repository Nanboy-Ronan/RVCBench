# Datasets

The benchmark data is on Hugging Face at
[Nanboy/RVCBench](https://huggingface.co/datasets/Nanboy/RVCBench). Dataset configs download it
automatically (`use_hf_dataset: true`, the default); `rvcbench prompts` and `rvcbench score` download only
the files a suite uses. An offline snapshot is also available on
[Google Drive](https://drive.google.com/file/d/1ZDOMorDGV8i5oVNtA5BaJLbFj2dVo5AU/view?usp=drive_link).
How each source corpus was selected and preprocessed is described in
[dataset preprocessing](dataset_preprocessing.md).

## Folders and the paper's evaluations

| Hub folder | Source | Used for (paper evaluation) | Dataset config |
| --- | --- | --- | --- |
| `Libritts` | LibriTTS test/dev-clean, 40 speakers | English-VC, LongAudio, AdvNoise and AntiProtect references | `libritts` |
| `VCTK` | VCTK, 40 speakers, 12 accents | AudioShift (accent, gender, age), English-VC | `vctk` |
| `vctk_text_robust` | VCTK voices, hallucination-style prompts | TextShift: Hallucination | `vctk_text_robust` |
| `robotcall` | Robocall scam scripts, VCTK voices | TextShift: Scam, Expression | `robotcall` |
| `AISHELL1_dev` | AISHELL-1, 40 speakers | Multilingual: Chinese-VC | `aishell` |
| `Bilingual_uedin` | EMIME English-Mandarin, 13 speakers | Multilingual: CrossLingual | `bilingual_uedin` |
| `CommonVoiceFR_dev` | Common Voice French, 40 speakers | Multilingual: French (appendix) | `french` |
| `Long_context` | LibriSpeech-Long, 10 speakers | LongContext: LongText | `long_librispeech` |
| `Background_noise` | VoiceBank+DEMAND, 10 noise types at 10 dB | PassiveNoise: Background | `background_noise`, `background_clean` |
| `Multispeaker_libri` | LibriTTS mixed with two interferers at four SNRs | PassiveNoise: MultiSpeaker | `multispeaker_libri` |
| `Protected_LibriTTS` | Protected references of the `core-v1` suite | AdvNoise, AntiProtect | used by `core-v1` |

Dataset configs live in [`src/rvcbench/configs/dataset/`](../src/rvcbench/configs/dataset/). To use a local
copy, pass `dataset.use_hf_dataset=false dataset.root_path=/path/to/<folder>`.

## Format

Each folder follows one layout:

```text
<folder>/
├── audios/<speaker_id>/*.wav
├── filelists/          # legacy per-speaker JSON manifests, kept for compatibility
└── metadata.parquet    # canonical manifest read by all loaders
```

`metadata.parquet` has one row per evaluation pair:

| Column | Description |
| --- | --- |
| `speaker_id` | Target speaker |
| `prompt_file_name` | Reference (prompt) audio |
| `prompt_text`, `prompt_language` | Transcript and language of the reference |
| `target_file_name` | Ground-truth target audio |
| `target_text`, `target_language` | Text to synthesize and its language |
| `pair_id`, `dataset_name`, `split` | Provenance |

Phoneme and alignment annotations (`prompt_phonemes`, `prompt_tone`, `prompt_word2ph` and their `target_*`
counterparts) are kept when available, and dataset-specific fields such as `spam_type` in `robotcall` are
extra columns. LibriTTS has two annotation exports (`speaker` and `speaker_text`); they stay distinct, see
the [run guide](run_protocol.md#libritts-manifest-variants).

To rebuild canonical manifests from legacy per-speaker JSON files:

```bash
python src/rvcbench/datasets/build_canonical_manifests.py --force
```
