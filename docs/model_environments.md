# Model Environments

RVCBench integrates many third-party voice cloning and TTS models. These
projects often pin incompatible versions of PyTorch, Transformers, ONNX Runtime,
tokenizers, or model-specific helper packages, so one global Python environment
is not expected to run every model.

The model environment specs are maintained in this repository under
[`envs/`](../envs/). Use the environment that matches the model you are
launching.

## Creating an Environment

The Qwen environment installs only the core package and Qwen runtime. Other
base templates install the core package and may require a separate
upstream checkout or package. They are installation starting points, not tested
lock files. Run from `envs/` so editable package paths resolve correctly:

```bash
cd envs
conda env create -f qwen3-tts.yml
conda activate qwen3
cd ..
```

Then run the corresponding benchmark command, for example:

```bash
python scripts/run_qwen3tts_quickstart.py --max-samples 5
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
| VoxCPM2 | `envs/voxcpm.yml` (base recipe; requires the pinned upstream checkout) | `voxcpm` |
| FireRedTTS2 | `envs/fireredtts2.yml` (native monologue canary validated; requires pinned upstream checkout) | `fireredtts2` |
| BertVITS2 / SafeSpeech surrogate | `envs/bertvits2.yml` | `bertvits2` |
| CosyVoice | `envs/cosyvoice.yml` | `cosyvoice` |
| dots.tts | `envs/dots-tts.yml` | `dots-tts` |
| Fish Audio S2 native | `envs/fish-s2-native.yml` ([setup](fish_s2_native.md); base recipe) | `fish-s2-native` |
| F5-TTS | `envs/f5-tts.yml` (validated Python 3.11 / Torch 2.8 / SDK 1.1.18 pins; base recipe) | `f5-tts` |
| Fish Audio S2 (local API server) | `envs/fish-speech-s2.yml` (base env; serves the model, see [README](../README.md)) | `fish-speech-s2` |
| FishSpeech | `envs/fishspeech.yml` | `fishspeech` |
| GLM-TTS | `envs/glm-tts.yml` | `glm-tts` |
| GlowTTS | `envs/glowtts.yml` | `glowtts` |
| Higgs Audio | `envs/higgs-audio.yml` | `higgs-audio` |
| Higgs TTS 3 | `envs/vllm-omni-cu129.yml` (base env; serves the model, see [README](../README.md)) | `vllm-omni-cu129` |
| IndexTTS | `envs/indextts.yml` | `indextts` |
| Kimi Audio | `envs/kimi-audio.yml` | `kimi-audio` |
| MaskGCT | `envs/maskgct.yml` | `maskgct` |
| MGM-Omni | `envs/mgm-omni.yml` | `mgm-omni` |
| MOSS-TTS | `envs/moss-tts.yml` | `moss-tts` |
| MOSS-TTSD (native MossTTSD checkpoint) | `envs/moss-ttsd.yml` | `moss-ttsd` |
| MOSS-TTSD (legacy Asteroid checkpoint) | `envs/moss.yml` | `moss` |
| OpenVoice | `envs/openvoice.yml` | `openvoice` |
| OZSpeech | `envs/ozspeech.yml` | `ozspeech` |
| PlayDiffusion | `envs/playdiffusion.yml` | `playdiffusion` |
| Qwen3-Omni | `envs/qwen3-omni.yml` | `qwen3-omni` |
| Qwen3-TTS | `envs/qwen3-tts.yml` | `qwen3` |
| Spark-TTS | `envs/sparktts.yml` | `sparktts` |
| StyleTTS2 | `envs/styletts2.yml` | `styletts2` |
| VALL-E | `envs/vall-e.yml` | `vall-e` |
| Bark Voice Clone | `envs/bark-native.yml` (native LibriTTS16 generated/scored; base recipe; [setup](bark_native.md)) | `bark-native` |
| Amphion VALL-E (separate implementation) | `envs/amphion-valle.yml` (native LibriTTS16 generated/scored; base recipe) | `amphion-valle` |
| VibeVoice | `envs/vibevoice.yml` | `vibevoice` |
| XTTS-v2 | `envs/xtts-v2.yml` | `xtts-v2` |
| ZipVoice | `envs/zipvoice.yml` | `zipvoice` |
| ZONOS2 | not a conda env — see `envs/zonos2.yml` (`uv`-managed `checkpoints/ZONOS2-repo/.venv`) | — |

## Reproducibility Options

The Qwen quickstart and CPU smoke path are covered by offline CI. The other
model families have adapter integrations and base environment templates; they
have not all been revalidated with current upstream packages. The root
`requirements.txt` is a legacy aggregate and is not the recommended installation
path. Follow each model runtime setup before using its adapter.

Generation-only usage needs the core plus model runtime. Install `.[eval]`
explicitly for full evaluation; run `rvcbench doctor --model qwen3 --eval --imports` to
check dependency availability before launching. This check does not load models
or certify CUDA compatibility. Training/protection runtimes need their own
upstream dependencies.

For stronger reproducibility, generate platform-specific lock files from these
YAML files with `conda-lock`, or publish prebuilt Docker/Apptainer images per
model family. Containers are usually the most reliable option for CUDA-heavy
third-party inference stacks, while Conda or Micromamba specs are easier for
users who need to adapt paths, CUDA versions, or local checkpoint locations.

For models requiring a different Torch version, generate with `+vc.generate_only=true`
in that model's environment and evaluate in a separate environment containing
`.[eval]`. The Qwen and evaluation extras pin a matched Torch/Torchaudio 2.6 pair;
other model base templates do not install the evaluation stack automatically.

Generation provenance records statically discovered `unmapped_imports` separately
from distribution versions. These names may be authored upstream namespaces,
missing modules or packages without distribution metadata. The inventory
includes literal `__import__()` and `import_module()` calls; it does not resolve
nonliteral imports or certify a complete environment lock.

For SparkTTS, run `rvcbench doctor --model sparktts --imports` in the chosen
runtime before generation. It checks the main native dependencies, including
`einx`, without loading weights. Dependency availability does not validate the
upstream source tree, checkpoint compatibility, CUDA or full model inference.

OpenVoice's converter-only CPU check uses native configuration and weights, but
its full audio API also needs a compatible NumPy/Numba/Librosa combination.
Import the audio stack in the intended runtime before loading checkpoints.
Read-only environment installations may require `NUMBA_CACHE_DIR` pointing to a
writable local directory. This addresses cache placement; it does not repair
NumPy/Numba version incompatibility or establish a clean OpenVoice/Melo lock.
