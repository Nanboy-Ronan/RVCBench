# Evaluate your own model

There are two ways to run a voice cloning model through RVCBench.

| Route | Use it when | You change |
| --- | --- | --- |
| [External adapter](#route-1-external-adapter) | You want scores for your own model | One Python file of your own |
| [Built-in integration](#route-2-built-in-integration) | You want the model listed in this repository | The package, via a pull request |

Both routes use the same runner, run records and metrics.

## Route 1: external adapter

### 1. Install

```bash
git clone --branch v2 https://github.com/Nanboy-Ronan/RVCBench.git
python -m pip install -e RVCBench            # runner and generation
python -m pip install -e 'RVCBench[eval]'    # add the scoring stack (Whisper, speaker verification, ...)
rvcbench doctor
```

Install into the environment that already runs your model.

### 2. Write the adapter

Subclass `rvcbench.VoiceCloningAdapter` and implement `clone`:

```python
# my_model_adapter.py
from rvcbench import VoiceCloningAdapter


class MyModelAdapter(VoiceCloningAdapter):
    def load(self):
        # Called once per run. self.config is the `adversary` block of the run config.
        self.model = load_my_model(self.config.checkpoint, device=self.device)

    def clone(self, *, text, reference_audio, reference_text, language):
        # Speak `text` in the voice of the audio file `reference_audio`.
        waveform = self.model.synthesize(text, prompt_wav=str(reference_audio), prompt_text=reference_text)
        return waveform, self.model.sample_rate   # mono float waveform in [-1, 1], sample rate in Hz
```

The contract:

- `clone` is called once per benchmark sample. `reference_audio` is a `pathlib.Path`; `reference_text` is its transcript (may be empty); `language` is the dataset's language tag or `None`.
- Return `(waveform, sample_rate)`. The runner writes the WAV file, names it, and checks that it is nonempty and finite.
- Raise an exception when an utterance cannot be generated. The runner records the failure for that sample and continues; it does not skip silently.
- The runner seeds Python, NumPy and PyTorch before each call (run seed plus the sample's source index). Do not reseed inside `clone`.
- `load` and `unload` are optional and run once per run.
- The adapter must be importable from a `.py` file, because each run records a hash of the adapter's source and of the modules it imports from the same package.

[`examples/echo_adapter.py`](../examples/echo_adapter.py) is a complete adapter that returns the reference audio; it is run by the test suite.

### 3. Run

```bash
PYTHONPATH=. rvcbench run --config-name ots_vc/clean/libritts/custom_ots \
    vc.adapter=my_model_adapter:MyModelAdapter \
    vc.model=my_model run_name=my_model_on_libritts \
    +adversary.checkpoint=/path/to/checkpoint.pt \
    adversary.seed=42 adversary.max_samples=20
```

- `vc.adapter` is `package.module:ClassName`. The module has to be importable: install your package, or put its directory on `PYTHONPATH` as above.
- `vc.model` is the name recorded in the run and shown in reports.
- `custom_ots` is a template config. Keys under `adversary` reach your adapter as `self.config`; prefix a key with `+` when the template does not define it.
- Add `+vc.generate_only=true` to generate without scoring.
- The first run downloads the selected dataset from the Hugging Face Hub. To use a local copy, pass `dataset.use_hf_dataset=false dataset.root_path=/path/to/dataset`.
- Other datasets: copy the template next to the other configs of that dataset, or keep your own configs in a directory and pass `--config-dir /path/to/your/configs`.

### 4. Read the results

Each run writes `results/<run_name>/<timestamp>/`:

| File | Content |
| --- | --- |
| `run_manifest.json` | Per-sample status, input and output hashes, seeds, resolved config, environment and source provenance |
| `generated_audio/` | One WAV per sample |
| `metrics.json` | Aggregate metrics with coverage over all requested samples |

```bash
rvcbench status results/my_model_on_libritts/<timestamp>
rvcbench report results/my_model_on_libritts/<timestamp> --output my_model_report.json
```

`rvcbench report` refuses runs with missing outputs or missing required metrics. See the [run guide](run_protocol.md) for resume, retries, evaluation-only scoring and the conditions under which two runs may be compared.

## Route 2: built-in integration

A pull request that adds a model to this repository contains:

| Piece | Location | Notes |
| --- | --- | --- |
| Adapter | `src/rvcbench/adversary/<model>_ots.py` | Subclass `BaseAdversary`. Keep model imports inside methods so that importing the module needs no model dependency. |
| Model wrapper | `src/rvcbench/models/<model>/` | Loading and inference, with explicit checkpoint paths or pinned Hub revisions. |
| Registry entry | `src/rvcbench/benchmark/registry.py` | `"<model>": "rvcbench.adversary.<model>_ots:<ClassName>"` |
| Configs | `src/rvcbench/configs/ots_vc/clean/<dataset>/<model>_ots.yaml` | Start from an existing config of the same dataset. |
| Environment | `envs/<model>.yml` | Models have incompatible dependency stacks; each gets its own environment. |
| Catalog entry | `src/rvcbench/benchmark/model_catalog.json` | Use `experimental_adapter` until a subset validation is recorded. |
| Tests | `tests/test_<model>_runtime.py` | Use fake modules or stub processes. Tests must not download or load a model. |
| Documentation | `docs/validation.md`, `README.md` | Add a validation row; state what was and was not run. |

Requirements for the adapter:

- Fail loudly. Missing references, empty text, incomplete checkpoints and invalid audio raise an error; they are not skipped or patched.
- Use the sample's original source index for seeding (`self._sample_seed(sample)`), not its position in a filtered list.
- Release owned resources in `close`, including worker processes.
- Do not write absolute paths of your machine into tracked files; `tests/test_public_paths.py` rejects them.

A model may additionally get a direct backend in `src/rvcbench/benchmark/backends.py` (an adapter `generate_sample` method plus a backend class with its own timing scope). `tests/test_direct_backends.py` shows the seed and failure checks those backends must pass.

Before opening the pull request, run the model on a fixed subset (for example `reproduction/subsets/libritts16_v1`) and attach the `rvcbench report` output. Describe the result as subset validation; it does not establish paper-table reproduction.
