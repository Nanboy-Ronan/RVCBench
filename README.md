# <img src="figs/logo.png" alt="RVCBench logo" width="40" style="vertical-align: middle; margin-right: 8px;"> RVCBench

**RVCBench** is an open benchmark for studying how robust modern voice cloning systems are against audio protection methods.

![RVCBench main figure](figs/main.png)

RVCBench gives you an end-to-end, reproducible pipeline to:
- apply protection to source prompts (SafeSpeech, Enkidu, EM, noise/spec perturbations),
- run voice cloning adversaries on clean or protected inputs,
- optionally denoise protected audio and re-evaluate,
- report fidelity and generation quality metrics.

## Why RVCBench

- Compare protection vs. cloning under one consistent framework.
- Reproduce runs with Hydra-based configs instead of one-off scripts.
- Extend quickly with your own datasets, methods, and adversary wrappers.
- Get structured outputs (`metrics.json`) that are easy to aggregate.

## Who It Is For

- Researchers benchmarking anti-voice-cloning defenses.
- Engineers evaluating attack/defense tradeoffs on new models.
- Contributors who want a common baseline for robust VC evaluation.

## Repository Layout

- `run_protect.py`: generate protected audio and evaluate fidelity.
- `run_vc.py`: run VC on clean prompts.
- `run_vc_protect.py`: run VC on protected prompts.
- `run_denoiser.py`: denoise protected audio and evaluate fidelity.
- `configs/`: Hydra configs for datasets, models, protection, VC, and denoising.
- `src/`: core modules (datasets, protection, adversaries, workflows, evaluation).
- `data/`: local data folders and manifests used by experiments.

## Canonical Dataset Layout

Datasets now use a canonical root-level manifest format to make Hugging Face uploads and local loading consistent:

```text
data/<dataset>/
├── audios/
│   └── <speaker>/*.wav
├── filelists/                  # legacy source manifests kept for compatibility/migration
└── metadata.parquet            # canonical manifest used by loaders
```

`metadata.parquet` stores one row per benchmark pair with clean Hugging Face-friendly columns such as:

- `speaker_id`
- `prompt_file_name`, `prompt_text`, `prompt_language`
- `target_file_name`, `target_text`, `target_language`
- `pair_id`, `dataset_name`, `split`

Training-related text features are also preserved in the canonical manifest when available:

- `prompt_phonemes`, `prompt_tone`, `prompt_word2ph`
- `target_phonemes`, `target_tone`, `target_word2ph`

Dataset-specific metadata such as `spam_type` in `robotcall` is preserved as additional columns.

To rebuild canonical manifests from the legacy per-speaker JSON files:

```bash
python src/datasets/build_canonical_manifests.py --force
```

Existing launch commands do not change. Dataset selection still happens through the configs under `configs/dataset/`, and `speaker_id` filtering still works per dataset.

## Dataset and Checkpoints

The public Hugging Face dataset is available here:

👉 **[Nanboy/RVCBench](https://huggingface.co/datasets/Nanboy/RVCBench)**

You can also download the dataset snapshot directly from Google Drive:

👉 **[Download the dataset](https://drive.google.com/file/d/1ZDOMorDGV8i5oVNtA5BaJLbFj2dVo5AU/view?usp=drive_link)**

We expect user to git clone the corresponding models and download their checkpoints in the `checkpoints` folder. If you just want to launch the evaluation for one model, then just clone that one is the fastest way. We also provided all pretrained checkpoints together so that you don't need to download them one by one (original code and checkpoints are all inside)
👉 **[Download the pretrained checkpoints](https://arbutus.cloud.alliancecan.ca/api/swift/containers/rjin/object/AudioBench/checkpoint.zip)**

## Setup

Create a Python environment with `PyTorch`, `torchaudio`, and `hydra-core`, then add model-specific dependencies based on the configs you plan to run.

Model checkpoints and third-party components are not bundled and must be downloaded separately.

- Some wrappers expect local paths like `checkpoints/`; verify each config under `configs/`. We will provide the link to them soon.

- If you need to add new models, simply clone their model into `checkpoints/` and write similar wrapper under `src/models/`.

## 5-Minute Start

If you only want to verify your setup quickly, run:

```bash
python run_vc.py --config-name ozspeech_ots vc.max_samples=5
```

Then check `results/<run_name>/<timestamp>/metrics.json`.

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
python run_vc.py --config-name ots_vc/clean/vctk/higgs_audio_ots
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

`prompt voice + prompt text -> (optional) vc_protect -> (optional) denoise -> voice clone --> evaluate`

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

## 🤝 Contributing

We welcome contributions of all kinds! In particular, we appreciate:

- 🛡️ New protection or defense methods  
- 🎭 New VC adversary wrappers  
- 📊 Dataset adapters and evaluation improvements  
- 🔁 Reproducibility enhancements  
- 📝 Documentation improvements and fixes  

### Questions?

If you have questions about the project, feel free to open an issue or contact:  
📧 **ruinanjin@alumni.ubc.ca**

---

Thank you for contributing and helping improve the project! 🚀


## Citation

```
@article{liao2026rvcbench,
  title={RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models},
  author={Liao, Xinting and Jin, Ruinan and Yu, Hanlin and Pandya, Deval and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2602.00443},
  year={2026}
}
```

## License

See `LICENSE`.
