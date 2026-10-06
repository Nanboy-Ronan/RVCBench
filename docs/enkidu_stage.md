---
title: "Produce Enkidu-protected voice references"
description: "Create Enkidu protected references for RVCBench using a defined speaker cohort, frozen sample selection and recorded training and generation provenance."
---

# Enkidu reference production

Install the model dependencies with `python -m pip install -e '.[enkidu]'`. For the exact versions used
for the published references, create the environment from `envs/enkidu.yml` instead.

`protect-enkidu` trains universal spectral perturbations on the dataset's full
speaker cohort, then writes only the references selected by a frozen manifest.
It uses local ECAPA assets and an audio-only loader, avoiding training text
features and their random fallbacks.

```bash
rvcbench protect-enkidu \
  --dataset-root /absolute/path/to/data/Libritts \
  --subset-manifest /absolute/path/to/reproduction/subsets/libritts16_v1/metadata.json \
  --model-directory /absolute/path/to/SpeakerRecognition \
  --device cuda:0 --seed 42 --epochs 10 \
  --output /absolute/path/to/new/protection-run
```

The model directory must contain `hyperparams.yaml`, `embedding_model.ckpt`,
`classifier.ckpt`, `mean_var_norm_emb.ckpt` and `label_encoder.ckpt`. The output
directory must be new; model cache files stay inside that run.

The `enkidu_audio_only_cohort_v1` protocol processes mono PCM16 references at
16 kHz with one reference per batch and retains the legacy cumulative-gradient
optimization. Training uses the complete filtered `speaker` cohort from
`metadata.parquet`, regardless of the number of selected output pairs.
The selected identities, paths and transcripts must match that cohort.

`stage_manifest.json` records training-reference hashes, model assets, settings,
source/runtime provenance, completed training steps, current losses, elapsed
time and an estimated training time remaining. Completion requires all training
steps, unchanged inputs/assets and valid selected output files. Failed stages
cannot be consumed by the cloning runner. Noise is saved at
`protected_audio/enkidu.noise`, with WAVs under `protected_audio/<speaker>/`.

This audio-only variant changes the historical loader's RNG consumption.
Historical WAV and metric equivalence therefore require a separate comparison;
a successful production run does not by itself reproduce the paper's scores.
