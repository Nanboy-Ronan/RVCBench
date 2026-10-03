# Dataset preprocessing

The preprocessed datasets are published on Hugging Face at
[Nanboy/RVCBench](https://huggingface.co/datasets/Nanboy/RVCBench); see [Datasets](datasets.md) for the
layout and how to use a local copy. This page describes how each dataset was preprocessed.

## Preprocessing
### VCTK (English multi-speaker, multi-accent)

We use a curated subset of the **CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)** [Yamagishi et al., 2019](https://doi.org/10.7488/ds/2645).  

The original VCTK corpus contains ~44 hours of English read speech from 110 speakers with diverse accents (American, British, Irish, Scottish, Indian, etc.), each reading about 400 phonetically rich sentences recorded with high-quality microphones in a hemi-anechoic chamber.

In our benchmark, we select **40 speakers** to balance **gender** and **accent** while keeping the dataset compact:

- **Speakers:** 40 in total (22 female, 18 male)  
- **Accents:** 12 accent categories  
  - Major accents (American, English, Canadian, Irish, Scottish, South African) have 4–6 speakers each  
  - Minor accents (e.g., Australian, New Zealand, Welsh, Northern Irish, Indian, British RP) have 1–3 speakers each  

All audio is used at **48 kHz, 16-bit PCM, mono**, following the original VCTK speaker IDs (e.g., `p294`, `p303`, `s5`).  

### LibriTTS (English multi-speaker TTS)

We use a small, speaker-balanced subset of the **LibriTTS** corpus.
LibriTTS is a large-scale, multi-speaker English corpus of read audiobooks, derived from LibriSpeech and released at 24 kHz for TTS research: it contains approximately **585 hours** of speech from **2,456 speakers**, with sentence-level segmentation, normalized text, and noisy utterances removed.

Official LibriTTS page: [https://www.openslr.org/60/](https://www.openslr.org/60/)

In our benchmark subset:

- **Speakers:** 40 total, all from LibriTTS `dev` and `test` splits  
- **Gender balance (overall):**  
  - 20 female speakers  
  - 20 male speakers  

- **Split design (speaker-disjoint):**  
  - **dev:** 19 speakers (7 F / 12 M)  
  - **test:** 21 speakers (13 F / 8 M)  

All audio is used at **24 kHz, mono, 16-bit PCM**, following the original LibriTTS speaker IDs  

### AISHELL-1

We use a subset of the **AISHELL-1** Mandarin corpus for development and evaluation.  
AISHELL-1 is an open-source Mandarin speech corpus containing about **178 hours** of transcribed speech from **400 speakers**, recorded in quiet indoor environments and released at **16 kHz**. The texts cover 11 application-oriented domains (finance, science & technology, sports, entertainment, news, etc.).

Official AISHELL-1 page: https://www.openslr.org/33/

In our benchmark subset:

- **Speakers:** 40 total (IDs `S0724`–`S0763`)  
- **Gender balance:** 28 female, 12 male  
- **Per-speaker content:** 50 paired utterances (`gt`, `ori`) per speaker  
- **Sampling rate:** 16 kHz, mono, 16-bit PCM  

### UEDIN_bilingual_96kHz (Mandarin/English bilingual)

We use the **Mandarin talkers** portion of the 96 kHz release of the **EMIME Bilingual English–Mandarin Database**.  
The original EMIME Mandarin/English database contains studio recordings of 14 native Mandarin speakers (7 female, 7 male) who read parallel material in both Mandarin and English for research on personalized speech-to-speech translation and cross-lingual voice conversion.
Official EMIME bilingual database page: https://www.emime.org/participate/emime-bilingual-database.html  

In our benchmark subset:

- **Speakers:** 13 total from `Mandarin_talkers/`  
  - Female: 6 (`MF1`, `MF3`, `MF4`, `MF5`, `MF6`, `MF7`)  
  - Male: 7 (`MM1`–`MM7`)  
  - We exclude **MF2** because some of her Mandarin recordings exhibit abnormal behaviour.
- **Gender balance:** 6 female, 7 male  
- **Bilingual content per speaker:**  
  - 25 **English → Mandarin** sentence pairs  
  - 25 **Mandarin → English** sentence pairs  
  → 50 bilingual pairs per speaker

- **Sampling rate:** 96 kHz, mono, 16-bit PCM  

### CommonVoiceFR_dev (French crowd-sourced speech)

We use a subset of **Mozilla Common Voice French v23.0**, taking clips from the `validated.tsv` split.  

Official Common Voice site: [https://commonvoice.mozilla.org/](https://datacollective.mozillafoundation.org/datasets/cmflnuzw5ahjms0zbrcl0vg4e)

In our benchmark subset:

- **Source split:** `validated.tsv` from Common Voice French v23.0  
- **Speakers:** 40 unique French speakers 
- **Per-speaker content:** 50 utterance pairs per speaker.
- **Audio format:** mono, 16-bit PCM, 48 kHz (downsampled in code if needed)

### Long_context (LibriSpeech-Long subset)

We use a long-form English subset derived from **LibriSpeech-Long**, a benchmark dataset for long-form speech processing released with *“Long-Form Speech Generation with Spoken Language Models”* (Park et al., 2024).  

- Original LibriSpeech-Long card: https://huggingface.co/datasets/ilyakam/librispeech-long  

In our benchmark subset (“Long_context”):

- **Language:** English  
- **Speakers:** 10  
  - Speaker IDs: `1272, 1673, 1919, 1993, 2078, 3576, 3853, 422, 6241, 8842`
- **Pairs per speaker:** 2 long-form utterance pairs (e.g., original vs processed)  
- **Total pairs:** 20  
- **Audio duration:** long-form segments, typically **0.4–4 minutes** per file  
- **Sampling rate:** 16 kHz, mono, 16-bit PCM (inherited from LibriSpeech)  


### VCTK natural_noise (VoiceBank+DEMAND subset)

We use a noisy VCTK subset derived from the **“Noisy speech database for training speech enhancement algorithms and TTS models”** (also known as the VoiceBank+DEMAND dataset).  
The original database provides parallel **clean/noisy speech at 48 kHz** based on the VCTK multi-speaker corpus, where clean VCTK utterances are mixed with environmental noises (mainly from the DEMAND corpus) and additional speech-shaped / babble noise, and is widely used for speech enhancement and noise-robust TTS research.   

Official dataset page: https://datashare.ed.ac.uk/handle/10283/2791  

In our benchmark subse:

- **Language:** English (multi-accent, inherited from VCTK)  
- **Speakers:** 20 VCTK speakers (`p226`–`p273`)  
- **Background noise:** 10 natural noise environments, all mixed at **10 dB SNR**  
  - `babble`, `cafeteria`, `car`, `kitchen`, `meeting`, `metro`, `restaurant`,  
    `ssn` (speech-shaped noise), `station`, `traffic`  
- **Per-speaker structure:**  
  - 20 **noisy–clean pairs** (each noisy utterance has a clean VCTK reference)  
  - 10 additional **clean-only** utterances  
  - → 30 sentence-level items per speaker  
- **Overall size:** 20 speakers × 30 items ≈ **600 sentence items** (clean/noisy together give about **1,200 audio files**)  
- **Audio format:** 48 kHz, mono, 16-bit PCM  

### Multispeaker_libri (English multi-speaker interference)

We use a synthetic multi-speaker interference set built from our **LibriTTS** English audiobook subset.  
Clean target utterances from 10 LibriTTS speakers are mixed with interfering speech from 2 additional speakers at controlled SNR levels, producing parallel **clean / mixture** pairs for studying multi-speaker robustness.

In our benchmark subset:

- **Target speakers:** 10 (5 male, 5 female)  
  - Male: `61, 908, 2300, 2830, 7729`  
  - Female: `237, 1221, 1284, 4970, 6829`  
- **Interferer speakers:** 2  
  - Female interferer: `121`  
  - Male interferer: `672`  
- **Clean groundtruth segments:** 100 (10 per target speaker)  
- **Mixtures per groundtruth:** 8  
  - 4 SNR levels: `-5 dB`, `0 dB`, `+5 dB`, `+10 dB`  
  - 2 interferers: `121`, `672`  
- **Total entries in `manifest.json`:** 800 mixture entries (each with a linked clean groundtruth)  
- **Sampling rate:** 16 kHz, mono, 16-bit PCM  

### iemocap (TO BE EXPLORED — emotional English speech)

> Status: **to be explored** — this dataset is prepared but not yet integrated into the main benchmark pipeline.

We construct an emotional subset from the **IEMOCAP (Interactive Emotional Dyadic Motion Capture)** corpus.  

In our prepared subset:

- **Language:** English  
- **Speakers:** 10 total (5 male, 5 female)  
  - Female: `Ses01_F, Ses02_F, Ses03_F, Ses04_F, Ses05_F`  
  - Male:   `Ses01_M, Ses02_M, Ses03_M, Ses04_M, Ses05_M`  
- **Total pairs:** 184 (`ori`, `gt`) audio pairs  
- **Emotion categories (6):** `ang`, `exc`, `fru`, `hap`, `neu`, `sad`  

**Pair construction rule**

For each speaker, we select utterances from the six emotion types and build pairs such that:

- `ori` and `gt` always come from the **same speaker**, but  
- **their emotions are deliberately different** (e.g., `ori = ang`, `gt = neu`).

### robocall_ftc (TO BE DONE — real-world scam calls)

> Status: **to be done** — planned for trustworthy deepfake / telecom-fraud experiments.

We plan to integrate the **Robocall Audio Dataset** released by Prasad & Reaves (NCSU) based on the FTC’s *Project Point of No Entry* (PPoNE).  
This dataset contains real-world audio recordings of automated or semi-automated phone calls (“robocalls”), most of which are suspected to be **illegal scam or spam calls**.   

Official repo: https://github.com/wspr-ncsu/robocall-audio-dataset  

Planned usage in our benchmark:


