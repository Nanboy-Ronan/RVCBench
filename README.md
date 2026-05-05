# <img src="figs/logo.png" alt="RVCBench logo" width="40" style="vertical-align: middle; margin-right: 8px;"> RVCBench

**RVCBench** is an open-source benchmark for evaluating the robustness of modern voice cloning (VC) systems against audio protection methods. It provides a unified, reproducible pipeline covering the full attack–defense cycle: protection, voice cloning, optional denoising, and multi-metric evaluation.

![RVCBench main figure](figs/main.png)

---

## Overview

Voice cloning technology poses a growing threat to speaker privacy. Audio protection methods—such as adversarial perturbations—aim to make cloned speech recognizably degraded, but their effectiveness varies widely across different VC models and datasets. RVCBench closes this gap by offering a single framework that:

- applies protection algorithms to source audio (SafeSpeech, Enkidu, EM, Gaussian noise, spectral perturbations),
- runs a broad suite of zero-shot and fine-tuning VC adversaries on clean or protected inputs,
- optionally denoises protected audio and re-evaluates,
- computes standardised fidelity and generation-quality metrics with bootstrap confidence intervals.

## Supported Models

### Voice Cloning Adversaries (Zero-Shot OTS)

| Model | Key |
|---|---|
| Qwen3-TTS | `qwen3_tts` |
| FireRedTTS-2 | `fireredtts2` |
| VoxCPM | `voxcpm` |
| F5-TTS | `f5_tts` |
| MaskGCT | `maskgct` |
| OpenVoice V2 | `openvoice` |
| Coqui XTTS-v2 | `xtts` |
| IndexTTS | `index_tts` |
| ZipVoice | `zipvoice` |
| FishSpeech | `fishspeech` |
| CosyVoice / CosyVoice 2 | `cosyvoice` |
| Higgs Audio | `higgs_audio` |
| SparkTTS | `sparktts` |
| VALL-E | `vall_e` |
| StyleTTS 2 | `styletts2` |
| GLM-TTS | `glm_tts` |
| MOSS TTSD | `moss_ttsd` |
| PlayDiffusion | `playdiffusion` |
| Bark Voice Clone | `bark_voice_clone` |
| OZSpeech | `ozspeech` |
| VibeVoice | `vibevoice` |

### Protection Methods

| Method | Description |
|---|---|
| SafeSpeech | Adversarial perturbation optimised against a surrogate VC model |
| Enkidu | Perceptual-loss adversarial perturbation |
| EM | Expectation–Maximisation perturbation |
| GRNoise | Gaussian random noise (no surrogate model required) |
| AntiFake | AntiFake adversarial perturbation |

---

## Getting Started

### Requirements

- Python 3.9+
- PyTorch ≥ 2.0 with CUDA
- `hydra-core`, `omegaconf`, `pandas`, `pyarrow`, `soundfile`, `librosa`
- Model-specific packages (installed per environment; see each model's config)

### Installation

```bash
git clone https://github.com/Nanboy-Ronan/RVCBench.git
cd RVCBench
pip install hydra-core omegaconf pandas pyarrow soundfile librosa \
            jiwer openai-whisper pymcd huggingface_hub
```

Model checkpoints and third-party inference code are **not bundled**. Download instructions are provided in the [Data & Checkpoints](#data--checkpoints) section.

---

## Quickstart Examples

Three end-to-end examples are provided as both Jupyter notebooks and standalone Python scripts. All outputs—data, generated audio, and metrics—are written **inside the repository directory**. Benchmark data is downloaded automatically from `Nanboy/RVCBench` on Hugging Face.

| Example | Notebook | Script |
|---|---|---|
| FishSpeech zero-shot VC on VCTK | `notebooks/rvcbench_fishspeech_quickstart.ipynb` | `scripts/run_fishspeech_quickstart.py` |
| Qwen3-TTS zero-shot VC on LibriTTS | `notebooks/rvcbench_qwen3tts_quickstart.ipynb` | `scripts/run_qwen3tts_quickstart.py` |
| Protection (GRNoise / SafeSpeech) + Qwen3-TTS | `notebooks/rvcbench_safespeech_qwen3tts_quickstart.ipynb` | `scripts/run_protect_qwen3tts_quickstart.py` |

Model setup for the quickstarts, including gated downloads and exact commands:
[docs/quickstart_model_setup.md](/tealab-data/rjin02/RVCBench/docs/quickstart_model_setup.md)

### Running the scripts

```bash
conda activate qwen3

# Zero-shot voice cloning only
python scripts/run_qwen3tts_quickstart.py                    # auto-downloads data from HF
python scripts/run_qwen3tts_quickstart.py \
    --no-hf-download --speaker-id 1089 --max-samples 10
python scripts/run_qwen3tts_quickstart.py \
    --qwen-checkpoint-path checkpoints/Qwen3-TTS-12Hz-1.7B-Base

# Protection + voice clone attack (two-step pipeline)
python scripts/run_protect_qwen3tts_quickstart.py            # Gaussian noise protection (no checkpoints needed)
python scripts/run_protect_qwen3tts_quickstart.py \
    --protect-config safespeech_on_libritts                    # SafeSpeech (requires surrogate-model checkpoints)

# FishSpeech zero-shot voice cloning
python scripts/run_fishspeech_quickstart.py

# Smoke-test the notebook companion scripts against local HF-formatted data only
python scripts/validate_quickstarts.py
```

Pass `--help` to either script for the full list of options.

---

## Full Pipeline

All entry points use [Hydra](https://hydra.cc) for configuration. Config values can be overridden directly on the command line.

### Experiment flow

```
source audio → [protection] → [denoising] → voice cloning → evaluation
```

### 1. Apply protection and evaluate fidelity

```bash
python run_protect.py --config-name safespeech_on_libritts
```

### 2. Run zero-shot voice cloning on clean prompts

```bash
# Using a specific model and dataset
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots

# With command-line overrides
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
    adversary.max_samples=50 dataset.speaker_id=1089
```

### 3. Run voice cloning on protected prompts

```bash
python run_vc_protect.py --config-name ots_vc/protection/safespeech/ozspeech_ots \
    protected_audio_dir=results/safespeech_on_libritts/<timestamp>/protected_audio
```

### 4. Optionally denoise protected audio and re-evaluate

```bash
python run_denoiser.py --config-name denoise/denoiser_dns64_on_protected_libritts_spec
```

---

## Running Specific VC Models

The zero-shot VC configs are under `configs/ots_vc/clean/`. Most older examples use `configs/ots_vc/clean/libritts/`, while some newer model integrations currently live under `configs/ots_vc/clean/libratts/`. Despite the directory name difference, the added configs here still default to the `libritts` dataset internally.

### If You Only Want One Model

You do not need to download every checkpoint bundle. For a single model, the workflow is:

1. Pick the Hydra config for that model under `configs/ots_vc/clean/...`.
2. Download or install only that model's runtime and checkpoints.
3. Point any local paths with Hydra overrides such as `adversary.code_path=...` or `adversary.checkpoint_path=...`.
4. Run `run_vc.py` against the dataset/config you want.

Generic pattern:

```bash
python run_vc.py --config-name <model_config> \
  dataset.speaker_id=<speaker_id> \
  adversary.max_samples=<n>
```

If the model needs a local repo checkout or checkpoint directory:

```bash
python run_vc.py --config-name <model_config> \
  dataset.speaker_id=<speaker_id> \
  adversary.code_path=/path/to/model_repo \
  adversary.checkpoint_path=/path/to/checkpoint_or_hf_id
```

Concrete examples:

```bash
# Qwen3-TTS only
python -m pip install -U qwen-tts
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir checkpoints/Qwen3-TTS-12Hz-1.7B-Base
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  dataset.speaker_id=1089 \
  adversary.checkpoint_path=checkpoints/Qwen3-TTS-12Hz-1.7B-Base

# FishSpeech only
git clone https://github.com/fishaudio/fish-speech.git checkpoints/fish_speech
huggingface-cli download fishaudio/s1-mini \
  --local-dir checkpoints/fish_speech/openaudio-s1-mini
python run_vc.py --config-name ots_vc/clean/vctk/fishspeech_ots \
  dataset.speaker_id=p226 \
  adversary.code_path=checkpoints/fish_speech \
  adversary.llama_checkpoint_path=checkpoints/fish_speech/openaudio-s1-mini \
  adversary.decoder_checkpoint_path=checkpoints/fish_speech/openaudio-s1-mini/codec.pth
```

For the full command list for the quickstart models, see
[docs/quickstart_model_setup.md](/tealab-data/rjin02/RVCBench/docs/quickstart_model_setup.md).

### Additional Examples

```bash
# FireRedTTS-2
python run_vc.py --config-name ots_vc/clean/libratts/fireredtts2_ots

# VoxCPM
python run_vc.py --config-name ots_vc/clean/libratts/voxcpm_ots
```

### Model-specific setup notes

- `Qwen3-TTS` can use either the Hugging Face model ID `Qwen/Qwen3-TTS-12Hz-1.7B-Base` directly or a local directory passed via `adversary.checkpoint_path=...`. It also needs the `qwen-tts` Python package.
- `FishSpeech` needs both a local checkout of `fishaudio/fish-speech` and the `fishaudio/s1-mini` checkpoint directory. Pass them with `adversary.code_path=...`, `adversary.llama_checkpoint_path=...`, and `adversary.decoder_checkpoint_path=...`.
- `FireRedTTS-2` expects a local upstream checkout at `checkpoints/FireRedTTS2` and pretrained weights under `checkpoints/FireRedTTS2/pretrained_models/FireRedTTS2` by default.
- `VoxCPM` defaults to the Hugging Face model ID `openbmb/VoxCPM2`. If you want to force offline/local loading, override `adversary.local_files_only=true` and optionally set `adversary.cache_dir=/path/to/cache`.
- All model-specific paths and generation knobs can be overridden at launch time with Hydra, for example: `adversary.code_path=/path/to/model_repo` or `adversary.max_samples=20`.

### Example overrides

```bash
python run_vc.py --config-name ots_vc/clean/libratts/fireredtts2_ots \
    adversary.max_samples=20 \
    dataset.speaker_id=1089

python run_vc.py --config-name ots_vc/clean/libratts/voxcpm_ots \
    adversary.local_files_only=true \
    adversary.cache_dir=/path/to/hf-cache
```

---

## Data & Checkpoints

### Benchmark dataset

The benchmark dataset is publicly available on Hugging Face. The data loader fetches it automatically when `use_hf_dataset: true` (the default in all dataset configs).

👉 **[Nanboy/RVCBench on Hugging Face](https://huggingface.co/datasets/Nanboy/RVCBench)**

A static snapshot is also available for offline use:

👉 **[Download via Google Drive](https://drive.google.com/file/d/1ZDOMorDGV8i5oVNtA5BaJLbFj2dVo5AU/view?usp=drive_link)**

### Model checkpoints

Each VC adversary requires its own inference code and pretrained weights. Clone the relevant repository into `checkpoints/` and verify the paths in the corresponding config under `configs/`.

A bundled archive with all supported model code and checkpoints is available here:

👉 **[Download pretrained checkpoints](https://arbutus.cloud.alliancecan.ca/api/swift/containers/rjin/object/AudioBench/checkpoint.zip)**

If you only need a single model, cloning just that repository is the fastest option.

---

## Dataset Format

Each dataset follows a canonical layout:

```text
data/<dataset>/
├── audios/
│   └── <speaker_id>/*.wav
├── filelists/          # legacy per-speaker JSON manifests (kept for compatibility)
└── metadata.parquet    # canonical manifest used by all loaders
```

`metadata.parquet` stores one row per benchmark pair with the following columns:

| Column | Description |
|---|---|
| `speaker_id` | Target speaker identifier |
| `prompt_file_name` | Path to the prompt (reference) audio |
| `prompt_text`, `prompt_language` | Transcript and language of the prompt |
| `target_file_name` | Path to the ground-truth target audio |
| `target_text`, `target_language` | Transcript and language of the target |
| `pair_id`, `dataset_name`, `split` | Provenance fields |

Training-oriented phoneme and alignment annotations (`prompt_phonemes`, `prompt_tone`, `prompt_word2ph`, and their `target_*` counterparts) are preserved when available. Dataset-specific fields (e.g., `spam_type` in `robotcall`) are carried as additional columns.

To rebuild canonical manifests from legacy per-speaker JSON files:

```bash
python src/datasets/build_canonical_manifests.py --force
```

Dataset selection and `speaker_id` filtering continue to work through `configs/dataset/` as before.

---

## Repository Structure

```text
RVCBench/
├── run_vc.py                  # voice cloning on clean prompts
├── run_protect.py             # apply protection + fidelity evaluation
├── run_vc_protect.py          # voice cloning on protected prompts
├── run_denoiser.py            # denoise protected audio + re-evaluate
├── configs/
│   ├── dataset/               # dataset configs (root_path, sampling_rate, …)
│   ├── model/                 # surrogate model configs for protection
│   ├── ots_vc/                # zero-shot VC configs (clean / protected)
│   └── denoise/               # denoiser configs
├── src/
│   ├── adversary/             # VC adversary wrappers
│   ├── protection/            # protection algorithm implementations
│   ├── datasets/              # dataset loaders and manifest utilities
│   ├── evaluation/            # fidelity and generation-quality metrics
│   ├── models/                # model generator wrappers
│   └── workflows/             # end-to-end pipeline orchestration
├── notebooks/                 # quickstart notebooks and runnable example scripts
└── data/                      # local dataset folders (populated at runtime)
```

---

## Outputs

Each run writes a timestamped directory under `results/`:

```text
results/<run_name>/<timestamp>/
├── generated_audio/            # cloned audio files (VC runs)
├── protected_audio/            # perturbed audio files (protection runs)
├── perturbed_noise/            # saved perturbation tensors
├── generation_sample_metrics.csv
├── fidelity_sample_metrics.csv
└── metrics.json                # aggregated metrics with bootstrap CIs
```

`metrics.json` contains fidelity metrics (SNR, STOI, MCD, WER, MOS, speaker similarity) for protection and denoising runs, and generation-quality metrics (MCD, WER, speaker similarity, MOS, emotion match rate, RTF) for VC runs.

---

## Configuration

All configs use [Hydra](https://hydra.cc). Any field can be overridden from the command line:

```bash
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
    device=cuda:1 \
    adversary.max_samples=100 \
    dataset.speaker_id=1089 \
    dataset.use_hf_dataset=false \
    dataset.root_path=/path/to/local/data
```

Key config locations:

| Path | Controls |
|---|---|
| `configs/dataset/` | Dataset root, sampling rate, speaker selection |
| `configs/ots_vc/` | VC model, generation hyperparameters, evaluation settings |
| `configs/model/` | Surrogate model used during protection |
| `configs/denoise/` | Denoiser model and paths |

---

## Contributing

Contributions are welcome. Areas of particular interest include:

- New protection or defense methods
- New VC adversary wrappers
- Dataset adapters and additional evaluation metrics
- Reproducibility and documentation improvements

Please open an issue or pull request on GitHub. For questions, contact:

**ruinanjin@alumni.ubc.ca**

---

## Citation

If you use RVCBench in your research, please cite:

```bibtex
@article{liao2026rvcbench,
  title   = {RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models},
  author  = {Liao, Xinting and Jin, Ruinan and Yu, Hanlin and Pandya, Deval and Li, Xiaoxiao},
  journal = {arXiv preprint arXiv:2602.00443},
  year    = {2026}
}
```

## License

See `LICENSE`.
