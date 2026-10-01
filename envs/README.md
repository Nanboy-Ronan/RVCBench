# Per-Model Environments

Each model family may require a different Python environment. These files are
base templates. The Qwen template installs core plus Qwen dependencies;
other templates require the matching upstream model runtime. These are not
validated lock files for every integration.

Create environments from this directory so relative requirement paths resolve:

```bash
conda env create -f qwen3-tts.yml
conda activate qwen3
```

See [`../docs/model_environments.md`](../docs/model_environments.md) for the
model-to-environment map and reproducibility notes.
