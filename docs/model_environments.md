# Model Environments

RVCBench integrates many third-party voice cloning and TTS models. These
projects often pin incompatible versions of PyTorch, Transformers, ONNX Runtime,
tokenizers, or model-specific helper packages, so one global Python environment
is not expected to run every model.

During internal development, the model environment files were maintained under:

```text
/tealab-data/rjin02/AudioWatermarkBench/envs
```

For the public release, the same environment specs are included in this
repository under [`envs/`](../envs/). Use the environment that matches the model
you are launching.

## Creating an Environment

The environment YAML files reference the repository-level
[`requirements.txt`](../requirements.txt). Run the commands from the `envs/`
directory so relative requirement paths resolve correctly:

```bash
cd envs
conda env create -f qwen3-tts.yml
conda activate qwen3
cd ..
```

Then run the corresponding benchmark command, for example:

```bash
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots
```

To update an existing environment:

```bash
cd envs
conda env update -f qwen3-tts.yml --prune
cd ..
```

## Model-to-Environment Map

| Model family | Environment file | Conda environment name |
|---|---|---|
| General benchmark / fallback | `envs/audiobench.yml` | `audiobench` |
| BertVITS2 / SafeSpeech surrogate | `envs/bertvits2.yml` | `bertvits2` |
| CosyVoice | `envs/cosyvoice.yml` | `cosyvoice` |
| F5-TTS | `envs/f5-tts.yml` | `audiobench` |
| FishSpeech | `envs/fishspeech.yml` | `fishspeech` |
| GLM-TTS | `envs/glm-tts.yml` | `audiobench` |
| GlowTTS | `envs/glowtts.yml` | `glowtts` |
| Higgs Audio | `envs/higgs-audio.yml` | `higgs-audio` |
| IndexTTS | `envs/indextts.yml` | `indextts` |
| Kimi Audio | `envs/kimi-audio.yml` | `kimi-audio` |
| MaskGCT | `envs/maskgct.yml` | `maskgct` |
| MGM-Omni | `envs/mgm-omni.yml` | `mgm-omni` |
| MOSS-TTSD | `envs/moss.yml` | `moss` |
| OpenVoice | `envs/openvoice.yml` | `audiobench` |
| OZSpeech | `envs/ozspeech.yml` | `audiobench` |
| PlayDiffusion | `envs/playdiffusion.yml` | `playdiffusion` |
| Qwen3-Omni | `envs/qwen3-omni.yml` | `qwen3-omni` |
| Qwen3-TTS | `envs/qwen3-tts.yml` | `qwen3` |
| Spark-TTS | `envs/sparktts.yml` | `sparktts` |
| StyleTTS2 | `envs/styletts2.yml` | `styletts2` |
| VALL-E | `envs/vall-e.yml` | `vall-e` |
| VibeVoice | `envs/vibevoice.yml` | `vibevoice` |
| XTTS-v2 | `envs/xtts-v2.yml` | `audiobench` |
| ZipVoice | `envs/zipvoice.yml` | `audiobench` |

## Reproducibility Options

The checked-in Conda YAML files are the lowest-friction public interface. They
make model-specific dependencies visible and let users install only the
environment they need.

For stronger reproducibility, generate platform-specific lock files from these
YAML files with `conda-lock`, or publish prebuilt Docker/Apptainer images per
model family. Containers are usually the most reliable option for CUDA-heavy
third-party inference stacks, while Conda or Micromamba specs are easier for
users who need to adapt paths, CUDA versions, or local checkpoint locations.
