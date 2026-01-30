# RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models

Benchmarking the robustness of modern voice-cloning (VC) systems and audio generation models.

RVCBench provides a modular pipeline to:
- prepare datasets for zero-shot or fine-tuned VC,
- apply protection methods to source audio (e.g., SafeSpeech, Enkidu, EM, noise/spec perturbations),
- run VC adversaries (zero-shot or fine-tune) on clean or protected prompts,
- optionally denoise protected audio and re-evaluate,
- compute fidelity and generation metrics with bootstrapping support.

## Repository structure

- `run_vc.py`: run VC on clean prompts.
- `run_protect.py`: generate protected audio and evaluate fidelity.
- `run_vc_protect.py`: run VC using protected prompts.
- `run_denoiser.py`: denoise protected audio and evaluate fidelity.
- `configs/`: Hydra configs (datasets, models, protection, VC, denoising).
- `src/`: core library (datasets, protection, adversary wrappers, workflows, evaluation).
- `scripts/`: helpers for downloading models and collecting results.

## Setup

This repo expects a working Python environment with PyTorch, torchaudio, Hydra, and other common audio/NLP deps. We will create a folder to indicate its environment specification; install dependencies based on your environment and target models.

Model weights and some third-party components must be downloaded separately. You can either use the helper scripts:

```bash
python scripts/download_models.py
```

Or download our pre-processed checkpoints using the following S3 link.

Notes:
- Some scripts contain hard-coded paths (e.g., `scripts/download_safespeech_models.py`). Update them to your environment before running.
- Several adversary wrappers expect checkpoints and configs under `checkpoints/` or `model/`. See the corresponding YAML in `configs/` for exact paths.
- The SafeSpeech reference code lives in `src/protection/safespeech/original_code` and may require its own setup.

## Dataset Downloading
We provide our preprocessed dataset in the S3 link here.

## Quick usage

All entrypoints use Hydra. Configs live under `configs/`, and you can override any value on the command line.

### 1) Run protection only (and evaluate fidelity)

```bash
python run_protect.py --config-name safespeech_on_libritts
```

This writes protected audio and metrics under `results/<run_name>/<timestamp>/`.

### 2) Run VC on clean prompts

```bash
python run_vc.py --config-name ozspeech_ots
```

Or pick a dataset-specific zero-shot config, for example:

```bash
python run_vc.py --config-name ots_vc/clean/vctk/bert_ots
```

### 3) Run VC on protected prompts

First, generate protected audio with `run_protect.py`, then point VC to that folder:

```bash
python run_vc_protect.py --config-name ots_vc/protection/safespeech/ozspeech_ots \
  protected_audio_dir=results/safespeech_on_libritts/<timestamp>/protected_audio
```

### 4) Denoise protected audio (optional)

```bash
python run_denoiser.py --config-name denoise/denoiser_dns64_on_protected_libritts_spec
```

Update the dataset root paths inside the denoiser config to your local protected-audio directory.

## Configuration tips

- Dataset configs live in `configs/dataset/`. Update `root_path`, `sampling_rate`, and `return_path` as needed.
- VC configs are organized under `configs/ots_vc/` for clean/protected settings and datasets.
- Most models use paths defined in the config (e.g., `adversary.code_path`, `checkpoint_path`).
- You can override any config field on the CLI, e.g.:

```bash
python run_vc.py --config-name ozspeech_ots vc.max_samples=50 dataset.root_path=/path/to/data
```

## Outputs

Each run writes to:

```
results/<run_name>/<timestamp>/
├── generated_audio/
├── protected_audio/
├── purturbed_noise/
└── metrics.json
```

Metrics are stored as JSON (fidelity metrics for protection/denoiser runs, generation metrics for VC runs).


## Example Notebooks
Below, we provide runnable Colab notebooks to help illustrate our usage.


## Citation
Coming soon.

## License

See `LICENSE`.
