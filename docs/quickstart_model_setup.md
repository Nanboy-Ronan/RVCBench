# Quickstart Model Setup

This repository publishes the benchmark dataset on Hugging Face, but the model
artifacts used by the quickstart notebooks are separate. Some are gated, and a
few require extra runtime packages or a local repo checkout.

The commands below make those dependencies explicit.

## 0. Environment Choice

The quickstarts and full benchmark runs should be launched from a
model-specific environment. The public release includes the same environment
base specifications under [`../envs/`](../envs/). The Qwen template installs
only core and Qwen dependencies. Other templates require the corresponding
upstream runtime and are not environment locks.

For example:

```bash
cd envs
conda env create -f qwen3-tts.yml
conda activate qwen3
cd ..
```

See [`model_environments.md`](model_environments.md) for the full
model-to-environment map and notes on lock files or containers.

## 1. Common Login

Accept the gated model terms first, then log in once:

- Qwen3-TTS: `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- FishSpeech S1-mini: `https://huggingface.co/fishaudio/s1-mini`

```bash
huggingface-cli login
```

## 2. Qwen3-TTS Quickstart

Install the runtime package:

```bash
python -m pip install -e '.[qwen3]'
```

Optional but recommended: pre-download the gated checkpoint into the repo so
the notebook/script does not need live network access during generation.

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir checkpoints/Qwen3-TTS-12Hz-1.7B-Base
```

Launch with the local checkpoint:

```bash
python scripts/run_qwen3tts_quickstart.py \
  --qwen-checkpoint-path checkpoints/Qwen3-TTS-12Hz-1.7B-Base
```

For the protection + Qwen3-TTS quickstart:

```bash
python scripts/run_protect_qwen3tts_quickstart.py \
  --qwen-checkpoint-path checkpoints/Qwen3-TTS-12Hz-1.7B-Base
```

## 3. FishSpeech S1-mini Quickstart

Use an isolated S1 checkout pinned before the S2 tokenizer change:

```bash
git clone https://github.com/fishaudio/fish-speech.git checkpoints/fish_speech_s1
git -C checkpoints/fish_speech_s1 checkout d3df50503b36314a964f66cac1af1e19e95bcfa3
python -m pip install -e checkpoints/fish_speech_s1
huggingface-cli download fishaudio/s1-mini \
  --local-dir checkpoints/fish_speech/openaudio-s1-mini
python scripts/run_fishspeech_quickstart.py \
  --fish-repo-dir checkpoints/fish_speech_s1 \
  --fish-ckpt-dir checkpoints/fish_speech/openaudio-s1-mini
```

The S1-mini release contains `tokenizer.tiktoken` and `special_tokens.json`.
Newer S2 tokenizer code expects Hugging Face tokenizer assets and can fail on
S1-mini even when all model weights load. The adapter validates the tokenizer
before starting its GPU worker. Keep upstream environments isolated because their
Torch and NumPy requirements can differ from the evaluation environment.

## 4. Fish Audio S2 Quickstart

Use a separate S2-compatible checkout and environment. Do not update the pinned
S1 checkout in place.

```bash
git clone https://github.com/fishaudio/fish-speech.git checkpoints/fish_speech
python -m pip install -e checkpoints/fish_speech
huggingface-cli download fishaudio/s2-pro \
  --local-dir checkpoints/fish_speech/s2-pro
python scripts/run_fishspeech_s2_quickstart.py \
  --fish-repo-dir checkpoints/fish_speech \
  --fish-ckpt-dir checkpoints/fish_speech/s2-pro
```

The S2 integration remains experimental while checkpoint-specific inference and
metric validation are pending. See `reproduction/plan.json` for measured status.

## 5. SafeSpeech / BertVITS2 Setup (advanced)

`grnoise_on_libritts` does not require the SafeSpeech surrogate checkpoints, but
`safespeech_on_libritts` does.

Install `.[eval]` and the protection runtime dependencies before using the
protection quickstart. These are separate from the minimal Qwen environment.

The upstream helper in
[`src/protection/safespeech/original_code/download_models.py`](../src/protection/safespeech/original_code/download_models.py)
downloads:

- `OedoSoldier/Bert-VITS2-2.3` base model files:
  `DUR_0.pth`, `D_0.pth`, `G_0.pth`, `WD_0.pth`
- `microsoft/deberta-v3-large`
- `microsoft/wavlm-base-plus`
- `speechbrain/spkrec-ecapa-voxceleb`

Install the SafeSpeech dependency stack expected by the upstream code:

```bash
python -m pip install -r src/protection/safespeech/original_code/requirements.txt
```

Then download the surrogate assets:

```bash
python src/protection/safespeech/original_code/download_models.py
```

Launch the SafeSpeech variant:

```bash
python scripts/run_protect_qwen3tts_quickstart.py \
  --protect-config safespeech_on_libritts \
  --qwen-checkpoint-path checkpoints/Qwen3-TTS-12Hz-1.7B-Base
```

## 6. Public Dataset Only

All three quickstart scripts already support the public dataset release from
`Nanboy/RVCBench`. They download only the dataset subset they need into
`data/`, then point the benchmark at the local dataset root with
`dataset.use_hf_dataset=false`.
