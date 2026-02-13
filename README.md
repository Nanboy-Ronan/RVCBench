# <img src="figs/logo.png" alt="RVCBench logo" width="40" style="vertical-align: middle; margin-right: 8px;"> RVCBench

**RVCBench** is a research benchmark for evaluating how robust modern voice cloning systems are against audio protection methods.

![RVCBench main figure](figs/main.png)

It provides an end-to-end, reproducible pipeline to:
- apply protection to source prompts (SafeSpeech, Enkidu, EM, noise/spec perturbations),
- run voice cloning adversaries on clean or protected inputs,
- optionally denoise protected audio and re-evaluate,
- report fidelity and generation quality metrics.

## Why RVCBench

- Unified benchmark for protection vs. cloning across multiple setups.
- Hydra-based configuration for reproducible experiments.
- Modular codebase for extending datasets, protection methods, and adversaries.
- Structured results with machine-readable metrics.

## Repository Layout

- `run_protect.py`: generate protected audio and evaluate fidelity.
- `run_vc.py`: run VC on clean prompts.
- `run_vc_protect.py`: run VC on protected prompts.
- `run_denoiser.py`: denoise protected audio and evaluate fidelity.
- `configs/`: Hydra configs for datasets, models, protection, VC, and denoising.
- `src/`: core modules (datasets, protection, adversaries, workflows, evaluation).
- `data/`: local data folders and manifests used by experiments.

## Dataset

We are preparing a Hugging Face release.  
For now, use the temporary download link:

**[Download dataset](https://drive.google.com/file/d/1ZDOMorDGV8i5oVNtA5BaJLbFj2dVo5AU/view?usp=drive_link)**

## Setup

Use a Python environment with `PyTorch`, `torchaudio`, and `hydra-core` plus model-specific dependencies.

Model checkpoints and third-party components are not bundled and must be downloaded separately.

- SafeSpeech reference code is under `src/protection/safespeech/original_code/`.
- SafeSpeech model downloads are handled by:

```bash
python src/protection/safespeech/original_code/download_models.py
```

- Some wrappers expect local paths like `checkpoints/` or `model/`; verify each config under `configs/`.

## Quickstart

All entrypoints use Hydra. You can override config values directly from the command line.

1. Generate protected audio and compute fidelity:

```bash
python run_protect.py --config-name safespeech_on_libritts
```

2. Run VC on clean prompts:

```bash
python run_vc.py --config-name ozspeech_ots
```

Example dataset-specific config:

```bash
python run_vc.py --config-name ots_vc/clean/vctk/bert_ots
```

3. Run VC on protected prompts:

```bash
python run_vc_protect.py --config-name ots_vc/protection/safespeech/ozspeech_ots \
  protected_audio_dir=results/safespeech_on_libritts/<timestamp>/protected_audio
```

4. Optionally denoise protected audio:

```bash
python run_denoiser.py --config-name denoise/denoiser_dns64_on_protected_libritts_spec
```

## Experiment Flow

`protect -> vc_protect -> (optional) denoise -> evaluate`

This makes it easy to compare clean, protected, and recovered performance under a single framework.

## Configuration Notes

- Dataset settings: `configs/dataset/` (`root_path`, `sampling_rate`, etc.).
- VC settings: `configs/ots_vc/` (clean/protected splits and model variants).
- Most model/checkpoint paths are defined in config fields such as `adversary.code_path` and `checkpoint_path`.

Example override:

```bash
python run_vc.py --config-name ozspeech_ots vc.max_samples=50 dataset.root_path=/path/to/data
```

## Outputs

Each run writes to:

```text
results/<run_name>/<timestamp>/
├── generated_audio/
├── protected_audio/
├── purturbed_noise/
└── metrics.json
```

`metrics.json` stores experiment metrics (fidelity for protection/denoiser runs and generation quality for VC runs).

## Citation

Coming soon.

## License

See `LICENSE`.
