# Run protocol (zero-shot v1)

Use `pip install -e .` from the repository root. Model runtimes and evaluation
models are separate dependencies. The initial supported onboarding path is
`pip install -e '.[qwen3]'` followed by the Qwen quickstart. Model checkpoints
are not part of the package. Other environment files are base templates requiring
the matching upstream runtime; they are not tested dependency lock files.

## Generate, evaluate, recover

```bash
python scripts/run_qwen3tts_quickstart.py --max-samples 5
```

The default quickstart only generates audio. It prints the generated audio and
metrics paths; `run_manifest.json` is in that timestamped run directory. No
Whisper, speaker-recognition, or perceptual evaluation model is loaded.

Install evaluation dependencies and score an existing run:

```bash
python -m pip install -e '.[eval]'
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  dataset.use_hf_dataset=false dataset.speaker_id=1089 adversary.max_samples=5 \
  +vc.evaluate_only=true \
  +vc.evaluation.generated_audio_dir=/absolute/path/to/run/generated_audio
```

Use exactly the same dataset selection, transcripts, and audio as the source run.
This command creates a new run; it does not overwrite the generation run. Full
metrics can also be requested with `--evaluate` on the Qwen quickstart.

```bash
python scripts/run_qwen3tts_quickstart.py --max-samples 5 \
  --resume-from /absolute/path/to/previous/run
```

Resume checks the input fingerprint, generation settings, runtime source digest,
Python environment, and generated audio hashes. Verified successful audio is copied into a new run.
Missing/failed samples are retried. Use `+vc.retries=1` with `run_vc.py` for an
additional attempt within a run. A per-sample seed is set before each adapter call.
Adapters may additionally apply their own configured seed policy, captured in the
config. GPU kernels and external services may still be nondeterministic.

The v1 runner dispatches one sample per adapter call, prioritizing explicit
failure accounting and resumability. It does not promise high-throughput batching
or multi-node scheduling. Model objects are reused and released before evaluation.

Inspect a running or interrupted job with `rvcbench status /path/to/run`.

## Artifacts and status

- `run_manifest.json`: atomic initial/final snapshots containing schema and
  protocol versions, selected sample IDs, transcripts, input hashes, output hashes,
  status, attempts, errors/warnings, code digest, Git commit, package versions,
  effective settings and coverage. Qwen Hub checkpoints are resolved to an immutable
  commit; local `checkpoint_path` directories record model/config hashes. Other
  adapters retain their configured references and need explicitly pinned upstream
  assets for strict reproducibility.
- `sample_events.jsonl`: durable per-sample updates, replayed on resume after
  interruption. A truncated final journal line is ignored; earlier corruption is
  rejected. This avoids rewriting the entire manifest for each sample.
- `generation_sample_metrics.csv`: sample IDs and individual scores, with missing
  metrics retained as missing rather than assigned successful scores.
- `generated_audio/synthesis_timings.csv`: adapter synthesis timing when provided;
  otherwise the full single-sample adapter call time. RTF must be compared under
  matched hardware/runtime conditions.
- Existing `metrics.json`, logs and audio remain available for compatibility.

`generated` means all requested audio exists but has not been scored. `complete`
requires all requested audio and every required metric for every requested sample.
`partial` includes failures or missing metrics; `failed` and `interrupted` record
exceptions or interrupt signals handled by the process. A forcibly killed process
may leave `running`/`generating`; these statuses must not be interpreted as success.

The default required metrics are MCD, WER and SIM. Configure a stricter gate with
`+evaluation.required_metrics=[mcd,wer,sim,sva,speechmos,dnsmos,emotion]`.
The full evaluator still computes all available metrics; this option sets the
comparison gate, not the model-loading selection. `metric_valid` always uses the
requested population as its denominator. Diagnostic averages remain available
for partial runs, but the report exporter rejects them. The existing bootstrap
uses utterance resampling, not speaker-cluster resampling; choose a statistical
protocol appropriate to the claim before publishing.

Completeness alone does not make two runs comparable: match the input fingerprint,
model version, protection settings, metric protocol, required metrics, and relevant
runtime/hardware settings. Historical and v1 per-sample execution timings must not
be merged without accounting for the protocol change.

## Export and website

```bash
rvcbench report /absolute/path/to/run --output docs/validated_runs/my-run.json
python docs/site-src/build.py
```

Export verifies coverage and the audio hashes and recomputes means from sample
records. Website builds recompute report means and reject incomplete or synthetic
runs. Published reports contain paths, transcripts, environment and provenance;
use this route for the public benchmark data, not private recordings. Historical
tables remain historical snapshots and are not automatically relabeled as v1 runs.

The output is a self-contained JSON report linked from the website's validated-run
section. No report is published until its JSON is committed and the site deployed.

## Historical audio and fine-tuning

To inspect pre-v1 audio without a manifest, explicitly use:

```bash
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  +vc.evaluate_only=true +vc.legacy_evaluation=true \
  +vc.evaluation.generated_audio_dir=/path/to/historical/audio
```

This uses the compatibility filename matcher and marks the result as legacy and
ineligible for the new report export. Never infer v1 provenance for these files.
The fine-tuning, protection-training, and denoising entry points retain their
existing protocols; v1 sample-level retry/resume applies to the zero-shot cloning
stage. They need their own model-specific training/protection dependencies.

For models requiring a different Torch version, generate with `+vc.generate_only=true`
in that model's environment and evaluate in a separate environment containing
`.[eval]`. The Qwen and evaluation extras pin a matched Torch/Torchaudio 2.6 pair;
other model base templates do not install the evaluation stack automatically.

Use `rvcbench doctor --model qwen3 --eval --imports` to detect import-time
compatibility errors before launching. Qwen/evaluation extras constrain NumPy to
1.26.4; the evaluator also constrains Numba. Install these in an isolated environment
rather than upgrading a shared environment used by other experiments.
