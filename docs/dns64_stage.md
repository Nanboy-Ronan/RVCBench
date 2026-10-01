# DNS64 reference production

`rvcbench denoise-dns64` enhances the references selected by a frozen manifest.
It requires an explicit local DNS64 state dict and an explicit reference
directory. It does not select the latest protection run, silently fall back
to another model, or download weights during inference.

```bash
rvcbench denoise-dns64 \
  --dataset-root /absolute/path/to/data/Libritts \
  --subset-manifest /absolute/path/to/reproduction/subsets/libritts16_v1/metadata.json \
  --reference-directory /absolute/path/to/protection-run/protected_audio \
  --weights /absolute/path/to/dns64-a7761ff99a7d5bb6.th \
  --device cuda:0 \
  --runtime-python /absolute/path/to/dns64-worker/bin/python \
  --output /absolute/path/to/new/denoising-run
```

The output directory must be new. The loader constructs the upstream DNS64
architecture without pretrained downloads, loads the local state dict with
`strict=True`, and uses evaluation mode. The upstream
[pretrained definitions](https://github.com/facebookresearch/denoiser/blob/main/denoiser/pretrained.py)
identify DNS64 as a 16 kHz Demucs model with 64 hidden channels.

The `dns64_dataset_rate_v1` recipe mirrors the historical wrapper: normalize
PCM input through torchaudio, resample to the declared dataset rate (default
16 kHz), resample to the model rate if necessary, infer, optionally mix the
input using `--dry` in [0, 1], resample back, restore the dataset-rate length,
clamp and write WAV using torchaudio's default writer. It preserves all pair
IDs when several pairs share a reference, enhancing that reference once.

`stage_manifest.json` records the clean and protected input lineage, weight
hash, source hashes, runtime packages, resampling/mixing settings and output
hashes. Missing or ambiguous references fail before loading the model.
Invalid inference produces a failed stage manifest. The clone runner checks
the known producer format's completion counts and selected output hashes
before accepting `denoised_audio` as its reference directory.
With `--runtime-python`, inference runs in a finite subprocess using a JSON
request/result protocol. The core process validates request, weight, worker and
kernel hashes, sample identities, output content, rates and frame counts.
Interpreter identity retains the virtual environment entry path even when
several environments share the same underlying Python binary.
`--timeout-seconds` defaults to 600; timeout terminates the worker and marks the
stage failed. `worker_result.json` records incremental worker progress, and
`worker.log` retains its diagnostics. Both direct and worker execution use the
same model-only inference kernel.

## Validation and remaining environment work

Two executions on `libritts16_v1` produced 16/16 identical output WAVs.
Comparison against the historical DNS64 run found equal rates and lengths for
all 16; the maximum difference was 5 PCM16 units. Historical WAV hashes do not
match, so this is a measured numeric comparison, not bitwise reproduction.
The new outputs also completed 16/16 Qwen3-TTS generations and core scores.
The retained evidence is
`reproduction/comparisons/dns64_gr_archived_noise_libritts16.json`.

The validated runtime is an isolated model-only overlay with denoiser 0.1.5,
julius 0.2.7 and Torch/Torchaudio 2.6.0. Its `pip check` fails because upstream
denoiser requires Hydra below 1.0, while the benchmark requires Hydra 1.3.
The overlay does not establish a dependency-clean installation. A separate
Python 3.10 environment with no system-site packages has passed `pip check`
and strict DNS64 weight loading. Its exact package set is retained in
`envs/dns64-worker-py310-cu124.txt`; this is a model-worker environment, not
the benchmark's core environment. This worker now runs through the public
CLI. Two independent worker executions produced 16/16 identical WAVs, all
matching the earlier direct-run WAVs. The core runtime records Hydra 1.3.2
while the worker records Hydra 0.11.3. A real 2-pair cloning canary verifies
consumption of the worker producer manifest; the matching direct-run reference
bytes were already used in the retained full 16-pair generation/scoring.
Worker evidence is retained in
`reproduction/comparisons/dns64_clean_worker_libritts16.json`.
This increment leaves the existing `run_denoiser.py` edits and shared
environments untouched.

The isolated model environment can be reconstructed with Python 3.10:

```bash
python3.10 -m venv /absolute/path/to/dns64-worker
/absolute/path/to/dns64-worker/bin/python -m pip install pip==24.0 typing_extensions==4.15.0
/absolute/path/to/dns64-worker/bin/python -m pip install \
  -r envs/dns64-worker-py310-cu124.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
/absolute/path/to/dns64-worker/bin/python -m pip check
```
