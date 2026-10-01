# Running and reproducing RVCBench

Install the benchmark with `python -m pip install -e .`. Use a separate model
runtime for generation and `python -m pip install -e '.[eval]'` for scoring.
Model checkpoints are downloaded or supplied separately. See
[model environments](model_environments.md) and the
[validation coverage](validation.md) before selecting an integration.

## Generate a fixed subset

The released selections under `reproduction/subsets/` contain pair identities,
transcripts and input hashes. `libritts16_v1` selects 16 pairs across eight
speakers. Use the same selection for generation, scoring and comparisons.
A manifest filename resolves inside the dataset; a relative path such as
`reproduction/subsets/...` can resolve from the project directory. Ambiguous
paths and missing explicit manifests fail instead of selecting another population.

```bash
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  run_name=qwen3_libritts16 \
  dataset.use_hf_dataset=false \
  +dataset.manifest_filename=reproduction/subsets/libritts16_v1/metadata.json \
  +vc.generate_only=true +seed=42
```

Supply the dataset locally under `data/Libritts`, or enable
`dataset.use_hf_dataset=true` to use the configured Hugging Face dataset.
Set model checkpoint/source overrides for the selected runtime; see
[Qwen setup](quickstart_model_setup.md) and
[pinned Hub snapshots](hub_revisions.md). A run creates a timestamped directory
under `results/<run_name>/` containing its manifest and generated audio.

Qwen3-TTS accepts individual samples directly through the request backend.
Each native seed must match the request's run seed plus original source index.
Model NaN/Inf output fails that sample, and prompt caches are released when the
backend closes. Its `qwen3_generate_excluding_prompt_encoding_and_io_v1` timing
measures the native generation call, excluding prompt encoding and file writing.
This scope alone does not establish timing comparability with another adapter.
Other integrations currently use the legacy adapter bridge.

To freeze a new subset before examining model outcomes:

```bash
python scripts/freeze_reproduction_subset.py \
  --dataset-config configs/dataset/libritts.yaml \
  --speakers 8 --pairs-per-speaker 2 \
  --output results/my_subset
```

The output directory must be new. The dataset configuration must point to a
locally available dataset; this command does not download it.

## Score saved audio

Run the same model configuration and frozen manifest in the evaluation
environment, supplying the generation run's actual `generated_audio` directory:

```bash
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  run_name=qwen3_libritts16_score \
  dataset.use_hf_dataset=false \
  +dataset.manifest_filename=reproduction/subsets/libritts16_v1/metadata.json \
  +vc.evaluate_only=true \
  +vc.evaluation.generated_audio_dir=/absolute/path/to/generated_audio
```

Evaluation-only mode scores existing audio without loading the generation model.
The scoring configuration must match the original input population and generation
protocol. Scorer definitions, devices and assets are recorded in the run.
Missing metrics and failed samples remain visible in coverage.

```bash
rvcbench status /absolute/path/to/scoring-run
rvcbench report /absolute/path/to/scoring-run --output results/report.json
rvcbench compare-check /absolute/path/to/run-a /absolute/path/to/run-b
```

`report` requires a complete, valid run; synthetic smoke runs cannot be exported
as benchmark results. `compare-check` verifies protocol compatibility for the
requested metrics. It does not by itself establish equality of results or
comparability of synthesis timing.

## Reference protection and denoising

Use an explicit reference directory and stage name to clone protected or
denoised voices. The original targets and sample identities are preserved.
See [reference binding](reference_stages.md),
[archived Gaussian-noise replay](gr_noise_replay.md) and
[DNS64 production](dns64_stage.md). Replay requires the original noise archive
and historical audio; these experimental assets are not included in the code.

## LibriTTS manifest variants

The subset contains 40 speakers and 4,000 distinct waveforms: 100 waveforms per
speaker form 50 reference–target pairs, giving 2,000 pairs overall. The metadata
also retains two annotation variants for those pairs: `speaker.json` and
`speaker_text.json`. They share audio paths but can differ in punctuation and
phonetic annotations. The default `manifest_variant: speaker` matches the
historical evaluation; selecting another variant changes the protocol.
Both variants are preserved and contribute to sample identity.

The paper's Table 9 counts waveforms. Its B.1 wording of 100 paired entries per
speaker differs from the available manifests, which contain 50 pairs. Do not
infer the evaluation denominator from the exported metadata row count.

## Interpreting reproduction

A successful subset run verifies that an integration generates and scores the
selected inputs. Matching historical rows additionally requires matching
reference/target audio, transcripts and metric protocols. Historical runs can
lack immutable checkpoint revisions, seeds or runtime details; their scores do
not establish bitwise generation reproducibility. Full paper-table reproduction
requires the complete population, rather than the onboarding subsets.

Generated audio, diagnostics and local audit snapshots belong under `results/`
and are excluded from source control. Public source retains reusable configs,
frozen selections, concise setup documentation and regression tests.
