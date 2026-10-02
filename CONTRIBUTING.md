# Contributing to RVCBench

## Branches

- `v2` is the development branch. Open pull requests against `v2`.
- `main` holds the v1 codebase (tag `v1.0`) and only receives notices.

## Development setup

```bash
git clone --branch v2 https://github.com/Nanboy-Ronan/RVCBench.git
cd RVCBench
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,http]'
pre-commit install
```

This is the CPU setup used by CI. Model integrations need their own environment from [`envs/`](envs/); see [docs/model_environments.md](docs/model_environments.md).

## Checks

```bash
ruff check .                              # lint baseline
python -m pytest -q                       # unit and contract tests, no model downloads
python scripts/validate_quickstarts.py    # quickstart commands and layouts, no inference
rvcbench smoke --output results/smoke     # synthetic end-to-end pipeline on CPU
python docs/site-src/build.py             # regenerate docs/index.html and docs/llms.txt
```

CI runs these on Python 3.10 and 3.12, builds the wheel, installs it and runs the smoke check outside the checkout.

## Rules for changes

- **Tests do not load models.** Use fake modules, stub worker processes and the fixtures in `tests/test_benchmark.py`.
- **Fail loudly.** Inputs that cannot be processed raise an error instead of being skipped, clipped or replaced.
- **No machine-specific paths** in tracked files. `tests/test_public_paths.py` enforces this.
- **State what was verified.** A subset validation is not a paper-table reproduction; a contract test is not a model run. Say which one a change has.
- **Generated site files** (`docs/index.html`, `docs/llms.txt`) are rebuilt with `docs/site-src/build.py` and committed with the change that affects them.
- **Commit messages** are one imperative sentence describing the change.

## Adding a model

See [docs/adding_a_model.md](docs/adding_a_model.md). You can evaluate a model with an external adapter without changing this repository, or contribute a built-in integration.

## Reporting a problem

Open an issue with the command you ran, the config name and overrides, the last lines of the log, and the output of `rvcbench doctor`. For a failed run, attach `run_manifest.json`; it records per-sample errors and the environment.
