# Quickstart Model Setup

This repository publishes the benchmark dataset on Hugging Face, but the model
artifacts used by the quickstart notebooks are separate. Some are gated, and a
few require extra runtime packages or a local repo checkout.

The commands below make those dependencies explicit.

## 0. Environment Choice

The quickstarts and full benchmark runs should be launched from a
model-specific environment. The public release includes the same environment
specs that were maintained internally at
`/tealab-data/rjin02/AudioWatermarkBench/envs`; use the checked-in copies under
[`../envs/`](../envs/) instead of relying on that absolute path.

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
python -m pip install -U qwen-tts
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

## 3. FishSpeech Quickstart

Clone the inference repo and install it editable:

```bash
git clone https://github.com/fishaudio/fish-speech.git checkpoints/fish_speech
python -m pip install -e checkpoints/fish_speech
```

Download the gated S1-mini checkpoint into the location expected by the config:

```bash
huggingface-cli download fishaudio/s1-mini \
  --local-dir checkpoints/fish_speech/openaudio-s1-mini
```

Launch:

```bash
python scripts/run_fishspeech_quickstart.py \
  --fish-repo-dir checkpoints/fish_speech \
  --fish-ckpt-dir checkpoints/fish_speech/openaudio-s1-mini
```

## 4. Fish Audio S2 Quickstart

Fish Audio S2 ([paper](https://arxiv.org/abs/2603.08823)) ships from the same
`fishaudio/fish-speech` repo as S1, so it reuses the `fishspeech` conda
environment and code checkout above — make sure your checkout is up to date:

```bash
cd checkpoints/fish_speech && git pull && cd -
python -m pip install -e checkpoints/fish_speech
```

Download the S2-Pro checkpoint:

```bash
huggingface-cli download fishaudio/s2-pro \
  --local-dir checkpoints/fish_speech/s2-pro
```

Launch:

```bash
python scripts/run_fishspeech_s2_quickstart.py \
  --fish-repo-dir checkpoints/fish_speech \
  --fish-ckpt-dir checkpoints/fish_speech/s2-pro
```

> [!WARNING]
> S2 uses a different generation architecture from S1 (a dual autoregressive
> "slow AR / fast AR" decoder over a 10-codebook codec, vs. S1's single-AR
> LLaMA-style model). The wrapper reuses S1's `ModelManager`/`TTSInferenceEngine`
> code path with `decoder_config_name: "modded_dac_vq"` as a best-effort default —
> confirm this still matches the codec config shipped with `s2-pro` once you've
> downloaded the checkpoint, and update the `configs/ots_vc/clean/*/fishspeech_s2_ots.yaml`
> files if upstream exposes a different config name for the S2 codec.

## 5. SafeSpeech / BertVITS2 Setup

`grnoise_on_libritts` does not require the SafeSpeech surrogate checkpoints, but
`safespeech_on_libritts` does.

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
