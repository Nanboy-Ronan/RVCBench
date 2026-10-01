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

## Isolated model environment

DNS64 requires legacy Hydra, which conflicts with the benchmark core.
Use a separate Python 3.10 worker environment with no system-site packages.
The pinned packages in `envs/dns64-worker-py310-cu124.txt` have passed
`pip check` and real subset inference. Direct and worker outputs match on
the fixed 16-pair selection. Historical DNS64 output differs by up to
5 PCM16 units, so bitwise historical equivalence remains unestablished.

The isolated model environment can be reconstructed with Python 3.10:

```bash
python3.10 -m venv /absolute/path/to/dns64-worker
/absolute/path/to/dns64-worker/bin/python -m pip install pip==24.0 typing_extensions==4.15.0
/absolute/path/to/dns64-worker/bin/python -m pip install \
  -r envs/dns64-worker-py310-cu124.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
/absolute/path/to/dns64-worker/bin/python -m pip check
```
