# Running the built-in models

RVCBench includes adapters for 32 voice cloning integrations. Each one needs its own Python
environment, the upstream inference code and the model checkpoints; none of these are bundled with
the package. If you only want scores for a model you can already run, you do not need any of this:
[score your own outputs](adding_a_model.md#route-0-score-audio-you-generated-anywhere) instead.

## Installation

Python 3.10 or newer on Linux. A GPU is recommended for scoring.

```bash
# package only
pip install "rvcbench[eval] @ git+https://github.com/Nanboy-Ronan/RVCBench@main"

# or a source checkout, needed to run the built-in models
git clone https://github.com/Nanboy-Ronan/RVCBench.git
cd RVCBench
python -m pip install -e '.[eval]'
rvcbench doctor                               # check dependencies
rvcbench smoke --output results/smoke         # synthetic CPU pipeline check, no downloads
```

| Extra | Adds |
| --- | --- |
| *(none)* | Runner, run records, suites and the `rvcbench` command |
| `eval` | Metrics: Whisper, SpeechBrain ECAPA and emotion, SpeechMOS, MCD, STOI |
| `qwen3` | Qwen3-TTS runtime for the quickstart |
| `http` | Clients for server-backed models |
| `enkidu` | Enkidu protection |
| `dev` | Tests, lint, pre-commit and build tools |

- **FFmpeg** is needed for the compression tasks and by several models.
- **Hugging Face login** (`hf auth login`) is recommended: anonymous downloads are rate-limited.
- **Scorer models** are stored in `$RVCBENCH_ASSET_DIR`, else `./checkpoints/` when it exists, else
  `~/.cache/rvcbench/`; see [model environments](model_environments.md).
- If installing `[eval]` reports that no `pysptk` version matches, see
  [building the evaluation extras](model_environments.md#building-the-evaluation-extras-from-source).

## Supported models

"Paper" marks the 18 models of the paper's main results and the four reported in its appendix (arXiv v3). "v2 subset run" means real generation and core scoring were
recorded on a fixed subset during the v2 refactor; see [validation coverage](validation.md).

| Model | `vc.model` | Paper | v2 subset run |
| --- | --- | :---: | :---: |
| BertVITS2 | `bertvits2` |  | pending |
| Qwen3-TTS | `qwen3_tts` | main | ✓ |
| Qwen3-Omni | `qwen3_omni` |  | pending |
| FireRedTTS-2 | `fireredtts2` |  | ✓ |
| VoxCPM | `voxcpm` |  | ✓ |
| F5-TTS | `f5_tts` | main | ✓ |
| MaskGCT | `maskgct` | main | ✓ |
| OpenVoice V2 | `openvoice` | main | ✓ |
| Coqui XTTS-v2 | `xtts` | main | ✓ |
| IndexTTS | `index_tts` | main | ✓ |
| ZipVoice | `zipvoice` | main | ✓ |
| FishSpeech | `fishspeech` | main | ✓ |
| Fish Audio S2 (in-proc) | `fishspeech_s2` | appendix | ✓ |
| Fish Audio S2 (server) | `fish_audio_s2` | appendix | ✓ |
| CosyVoice / 2 | `cosyvoice` | main | ✓ |
| Higgs Audio | `higgs_audio` | main | ✓ |
| Higgs TTS 3 | `higgs_tts_3` | appendix | pending |
| SparkTTS | `sparktts` | main | ✓ |
| VALL-E | `vall_e` |  | pending |
| StyleTTS 2 | `styletts2` | main | ✓ |
| GLM-TTS | `glm_tts` | main | ✓ |
| GlowTTS | `glowtts` |  | pending |
| Kimi Audio | `kimi_audio` |  | ✓ |
| MGM-Omni | `mgm_omni` | main | ✓ |
| MOSS TTSD | `moss_ttsd` | main | ✓ |
| MOSS-TTS | `moss_tts` | appendix | ✓ |
| dots.tts | `dots_tts` | appendix | ✓ |
| ZONOS2 | `zonos2` |  | ✓ |
| PlayDiffusion | `playdiffusion` | main | ✓ |
| Bark Voice Clone | `bark_voice_clone` |  | ✓ |
| OZSpeech | `ozspeech` | main | ✓ |
| VibeVoice | `vibevoice` | main | ✓ |

## One model, step by step

1. Pick the model's config under [`src/rvcbench/configs/ots_vc/clean/`](../src/rvcbench/configs/ots_vc/clean/),
   for example `ots_vc/clean/libritts/qwen3_tts_ots`.
2. Create and activate the model's environment from [`envs/`](../envs/); see
   [model environments](model_environments.md) for the map.
3. Install the model's runtime and download its checkpoints (notes below).
4. Point local paths at them with Hydra overrides such as `adversary.code_path=...` or
   `adversary.checkpoint_path=...`.
5. Run it:

```bash
rvcbench run --config-name <model_config> \
  dataset.speaker_id=<speaker_id> \
  adversary.max_samples=<n>
```

`rvcbench run` is the same as `python run_vc.py` in a source checkout; both take Hydra arguments.
Results go to `results/<run_name>/<timestamp>/` (see [run guide](run_protocol.md)).

### Examples

```bash
# Qwen3-TTS
python -m pip install -e '.[qwen3]'
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir checkpoints/Qwen3-TTS-12Hz-1.7B-Base
rvcbench run --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  dataset.speaker_id=1089 \
  adversary.checkpoint_path=checkpoints/Qwen3-TTS-12Hz-1.7B-Base

# FishSpeech
git clone https://github.com/fishaudio/fish-speech.git checkpoints/fish_speech_s1
git -C checkpoints/fish_speech_s1 checkout d3df50503b36314a964f66cac1af1e19e95bcfa3
python -m pip install -e checkpoints/fish_speech_s1
huggingface-cli download fishaudio/s1-mini --local-dir checkpoints/fish_speech/openaudio-s1-mini
rvcbench run --config-name ots_vc/clean/vctk/fishspeech_ots \
  dataset.speaker_id=p226 \
  adversary.code_path=checkpoints/fish_speech_s1 \
  adversary.llama_checkpoint_path=checkpoints/fish_speech/openaudio-s1-mini \
  adversary.decoder_checkpoint_path=checkpoints/fish_speech/openaudio-s1-mini/codec.pth

# FireRedTTS-2 and VoxCPM
rvcbench run --config-name ots_vc/clean/libritts/fireredtts2_ots
rvcbench run --config-name ots_vc/clean/libritts/voxcpm_ots

# dots.tts (env: dots-tts) and MOSS-TTS (env: moss-tts)
rvcbench run --config-name ots_vc/clean/libritts/dots_tts_ots device=cuda:<gpu>
rvcbench run --config-name ots_vc/clean/libritts/moss_tts_ots device=cuda:<gpu>

# ZONOS2 (uv-managed environment; run with its own interpreter, GPU pinned with CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=<gpu> checkpoints/ZONOS2-repo/.venv/bin/python run_vc.py \
  --config-name ots_vc/clean/libritts/zonos2_ots
```

More overrides:

```bash
rvcbench run --config-name ots_vc/clean/libritts/fireredtts2_ots adversary.max_samples=20 dataset.speaker_id=1089
rvcbench run --config-name ots_vc/clean/libritts/voxcpm_ots adversary.local_files_only=true adversary.cache_dir=/path/to/hf-cache
```

[Quickstart model setup](quickstart_model_setup.md) has the exact commands for the quickstart models,
including gated downloads.

## Model-specific notes

- **Qwen3-TTS** accepts the Hugging Face ID `Qwen/Qwen3-TTS-12Hz-1.7B-Base` or a local directory via
  `adversary.checkpoint_path`, and needs the `qwen-tts` package (`.[qwen3]` extra).
- **FishSpeech** needs a checkout of `fishaudio/fish-speech` and the `fishaudio/s1-mini` checkpoint, passed
  with `adversary.code_path`, `adversary.llama_checkpoint_path` and `adversary.decoder_checkpoint_path`.
- **Fish Audio S2** ([paper](https://arxiv.org/abs/2603.08823)) needs a separate S2-compatible checkout and
  `fishaudio/s2-pro`. Keep the pinned S1 checkout separate: newer upstream tokenizer code is incompatible
  with the released S1-mini tiktoken files. The opt-in [stable HTTP codec variant](fish_s2_codec_stability.md)
  produced identical waveforms across two service instances; the original HTTP path has a cold/warm
  numerical difference. See [native setup](fish_s2_native.md) and [HTTP setup](fish_s2_http.md).
- **FireRedTTS-2** expects the upstream checkout at `checkpoints/FireRedTTS2` and weights under
  `checkpoints/FireRedTTS2/pretrained_models/FireRedTTS2` by default.
- **VoxCPM** defaults to `openbmb/VoxCPM2`; set `adversary.local_files_only=true` (and optionally
  `adversary.cache_dir`) to load offline.
- **dots.tts** (`rednote-hilab/dots.tts-soar`) needs its own environment (`envs/dots-tts.yml`, which lists the
  install order; it requires `torch>=2.8.0`). `device: cuda:N` is honoured.
- **MOSS-TTS** (`OpenMOSS-Team/MOSS-TTS-v1.5`, `transformers` with `trust_remote_code`) needs `envs/moss-tts.yml`;
  install its packages in the order listed there.
- **ZONOS2** (`Zyphra/ZONOS2`) is a `uv` project (`envs/zonos2.yml` lists the clone and `uv sync` steps). Its
  scheduler ignores `device: cuda:N`, so pin the GPU with `CUDA_VISIBLE_DEVICES`; it pre-allocates a large KV
  cache (about 55 GB on an 80 GB A100), so run it alone on its GPU.
- **dots.tts, ZONOS2 and MOSS-TTS** pass the sample's `target_language` to the model, which matters for the
  AISHELL, French and cross-lingual configs.
- **Fish Audio S2 (server) and Higgs TTS 3** call a local inference server instead of loading weights; see below.

## Server-backed models

These adapters send requests to a local HTTP server (`adversary.endpoint_url`). Start the server, then run
the config from an environment with `python -m pip install -e '.[http]'`.

**Fish Audio S2** (`fishaudio/s2-pro`, Fish Speech `/v1/tts` API):

```bash
git clone https://github.com/fishaudio/fish-speech.git checkpoints/fish_speech
conda env create -f envs/fish-speech-s2.yml
conda activate fish-speech-s2
uv pip install 'numba==0.63.1' 'llvmlite==0.46.0'
cd checkpoints/fish_speech && uv pip install -e '.[cu126]' && uv pip install 'protobuf>=6.31.1,<7' && cd ../..
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
CUDA_VISIBLE_DEVICES=<gpu> python checkpoints/fish_speech/tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --listen 0.0.0.0:8001 --half
# in another shell:
rvcbench run --config-name ots_vc/clean/libritts/fish_audio_s2_ots
```

**Higgs TTS 3** (`bosonai/higgs-tts-3-4b`, vLLM-Omni `/v1/audio/speech` API):

```bash
conda env create -f envs/vllm-omni-cu129.yml
conda activate vllm-omni-cu129
uv pip install --torch-backend=cu129 --extra-index-url https://wheels.vllm.ai/0.24.0/cu129 vllm==0.24.0
uv pip install --torch-backend=cu129 vllm-omni==0.24.0
hf download bosonai/higgs-tts-3-4b
CUDA_VISIBLE_DEVICES=<gpu> vllm-omni serve bosonai/higgs-tts-3-4b \
  --host 0.0.0.0 --port 8000 --trust-remote-code --omni --allowed-local-media-path "$(pwd)"
# in another shell:
rvcbench run --config-name ots_vc/clean/libritts/higgs_tts3_ots
```

Both configs set `device: cpu` for the client process and `evaluation.device: cuda:0` for scoring; the
server's GPU is chosen with `CUDA_VISIBLE_DEVICES`.

## Known metric gaps in some model environments

These come from package versions in the model environments, not from the models (see the
`emotion_pairs` and `speechmos_pairs` fields of a run's `metrics.json`):

- **dots-tts, moss-tts**: no emotion scores; the emotion recognizer needs `AutoModelWithLMHead`, which the
  `transformers` version these models require has removed.
- **zonos2**: no SpeechMOS or emotion scores; `torchcodec` fails to load its native library with this
  environment's torch and FFmpeg. WER, MCD, SIM and DNSMOS are unaffected.

Scoring the saved audio from a separate evaluation environment (`+vc.evaluate_only=true`, see the
[run guide](run_protocol.md#score-saved-audio)) avoids these gaps.

## Checkpoints

A bundle of the supported models' code and checkpoints is available
[here](https://object-arbutus.alliancecan.ca/swift/v1/86581f3bb67c4c04bbccbcb839de730a/rjin/AudioBench/checkpoint.zip)
(about 58 GB). For a single model, cloning only that model's repository is faster. Check the paths in the
model's config before running.

## Protection and denoising

The protection pipeline perturbs reference audio, clones from the protected references, and optionally
denoises them first:

```text
source audio → protection → (denoising) → voice cloning → evaluation
```

```bash
rvcbench protect --config-name safespeech_on_libritts                     # protect and measure fidelity
rvcbench run-protected --config-name ots_vc/protection/safespeech/ozspeech_ots \
  protected_audio_dir=results/safespeech_on_libritts/<timestamp>/protected_audio
rvcbench denoise --config-name denoise/denoiser_dns64_on_protected_libritts_spec
```

Protection configs: `grnoise_on_libritts` (Gaussian noise), `spec_on_libritts` (SPEC),
`safespeech_on_libritts` (SafeSpeech), `em_on_libritts` (the paper's POP) and `enkidu_on_libritts` (Enkidu).
SPEC and SafeSpeech need surrogate-model checkpoints; see [quickstart model setup](quickstart_model_setup.md).

## Quickstart scripts and notebooks

| Example | Notebook | Script |
| --- | --- | --- |
| Qwen3-TTS on LibriTTS | [`notebooks/rvcbench_qwen3tts_quickstart.ipynb`](../notebooks/rvcbench_qwen3tts_quickstart.ipynb) | `scripts/run_qwen3tts_quickstart.py` |
| FishSpeech on VCTK | [`notebooks/rvcbench_fishspeech_quickstart.ipynb`](../notebooks/rvcbench_fishspeech_quickstart.ipynb) | `scripts/run_fishspeech_quickstart.py` |
| Fish Audio S2 on VCTK | — | `scripts/run_fishspeech_s2_quickstart.py` |
| Gaussian noise or SafeSpeech protection, then Qwen3-TTS | [`notebooks/rvcbench_safespeech_qwen3tts_quickstart.ipynb`](../notebooks/rvcbench_safespeech_qwen3tts_quickstart.ipynb) | `scripts/run_protect_qwen3tts_quickstart.py` |

The scripts download the selected speakers from the Hugging Face dataset unless `--no-hf-download` is
passed; `--help` lists their options. `python scripts/validate_quickstarts.py` checks their commands and
layouts without running a model.
