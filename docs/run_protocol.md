# Run protocol (zero-shot v2)

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
This command creates a new run; it does not overwrite the generation run. Generation and scoring can also be requested with `--evaluate` on the Qwen quickstart.

XTTS-v2 with Coqui TTS 0.22.0 requires the Transformers 4.40.2 / tokenizers
0.19.1 combination recorded in `envs/xtts-v2.yml`. The local subset check uses
Python 3.11 and NumPy 1.26.4 in an isolated overlay; this validates that runtime
combination, not a fully locked environment installation. The runner retains
the XTTS model across samples and releases it before scoring.

```bash
python scripts/run_qwen3tts_quickstart.py --max-samples 5 \
  --resume-from /absolute/path/to/previous/run
```

Resume checks the input fingerprint, generation settings, runtime source digest,
Python environment, and generated audio hashes. Verified successful audio is copied into a new run.
For subprocess adapters, the worker interpreter and its installed package versions
are recorded independently and checked on resume. Evaluation-only runs preserve
source generation provenance and do not probe or start the model interpreter.
Configured auxiliary source trees such as `melo_code_path`, converter configs,
vocabularies and base speaker embeddings also contribute to the generation
fingerprint. This prevents resuming after those inputs change. Implicit upstream
downloads remain outside this local-asset coverage and must be captured or pinned
separately; earlier runs are not retroactively assigned the expanded coverage.
CosyVoice can pin an external Matcha-TTS checkout with
`adversary.matcha_code_path=/absolute/path/to/Matcha-TTS`. Its source is included
in the generation fingerprint, and a conflicting already-imported checkout is
rejected. Use this when the dependency is outside the CosyVoice source tree.
Missing/failed samples are retried. Use `+vc.retries=1` with `run_vc.py` for an
additional attempt within a run. A per-sample seed is set before each adapter call.
The effective run seed is propagated into adapters; seeds use the preserved source
index rather than the index within a retry batch. Subprocess workers receive explicit
per-sample seeds. GPU kernels and external services may still be nondeterministic.

The v2 runner dispatches one sample per adapter call, prioritizing explicit
failure accounting and resumability. It does not promise high-throughput batching
or multi-node scheduling. Model setup runs before generation so missing runtime dependencies fail the run
early. Owned subprocess workers are explicitly closed before evaluation.

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
- `scoring_manifest.json`: metric-specific implementation, package and model asset
  provenance. Core MCD, Whisper WER and ECAPA scorers load separately.
- `metric_cache/`: atomic sample-by-scorer results. Successful records can be reused
  from the source run when inputs, scorer code, weights and settings match. Failed
  records are retried; each completed metric is also journaled.
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
Only requested metrics are loaded and evaluated. SIM and SVA share one ECAPA
scorer. Optional SpeechMOS, DNSMOS and emotion metrics have separate lifetimes;
their current wrappers retain historical implementations and record cached asset hashes. `metric_valid` always uses the
requested population as its denominator. Diagnostic averages remain available
for partial runs, but the report exporter rejects them. The existing bootstrap
uses utterance resampling, not speaker-cluster resampling; choose a statistical
protocol appropriate to the claim before publishing.

Completeness alone does not make two runs comparable: match the input fingerprint,
model version, protection settings, metric protocol, required metrics, and relevant
runtime/hardware settings. Historical and current per-sample execution timings must not
be merged without accounting for the protocol change.

Run `rvcbench compare-check /path/to/run-a /path/to/run-b --metrics mcd wer sim`
before comparing current model quality. This checks both runs' audio hashes,
population, seed policy and metric-specific scorer fingerprints, and returns a
nonzero exit code when they differ. The old coverage field
`eligible_for_comparison` means only that coverage is complete; it does not prove
pairwise comparability. New runs record `comparison_status` separately. Timing
comparisons and intervention comparisons require their additional protocols.

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

## Manifest variants and subset reproduction

LibriTTS contains 4,000 distinct waveforms forming 2,000 reference–target pairs.
Older exports also contain a second transcript representation in `speaker_text`
rows. Both representations are preserved and receive distinct sample identities.
`configs/dataset/libritts.yaml` explicitly selects `manifest_variant: speaker`,
matching the original `<speaker>.json` evaluation loader. Use
`dataset.manifest_variant=speaker_text` to select the alternate representation,
or `dataset.manifest_variant=null` to retain all variants. A custom manifest can
store `manifest_variant` explicitly. No source audio or manifest is rewritten by
selection. See `docs/audits/libritts_manifest_20260930.json` for the source audit.

The frozen `reproduction/subsets/libritts16_v1` selection contains eight speakers
and two pairs per speaker, selected by a fixed hash ranking before model outcomes
were examined. It records content hashes, original indices and transcript versions.
Load it using the original dataset audio root:

```bash
python run_vc.py --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  dataset.use_hf_dataset=false \
  +dataset.manifest_filename=/absolute/path/to/reproduction/subsets/libritts16_v1/metadata.json \
  +vc.generate_only=true +seed=42
```

`scripts/compare_reproduction_subset.py` validates exact historical pair filenames,
speaker identities, target transcripts and both input audio hashes before comparing
scores. It reports paired deltas and a speaker-cluster bootstrap interval.
`scripts/replay_historical_metrics.py` separately checks whether current scorers
reproduce scores on the original generated audio. These are different checks:
stochastic new generation can differ even when historical metric replay agrees.
Neither a 16-pair check nor a confidence interval alone establishes full-table
reproduction or statistical equivalence. Per-model progress and remaining
architecture requirements are recorded in `reproduction/plan.json`.

Historical English WER used more than one normalization formula. Set
`+evaluation.wer_normalization=lowercase_v1` to preserve punctuation, or
`+evaluation.wer_normalization=ascii_punctuation_removed_v2` for the current
default. The choice is recorded in the scorer version and cache fingerprint.
The historical replay script verifies the formula against every saved transcript
before selecting it. The paired comparator rejects different formulas.
Target-only legacy output names also require a complete, ordered generation log
that verifies reference audio identity; target audio alone is insufficient.
Even with matching formulas, Whisper predictions can differ across runtime
versions. Replay reports retain those differences and do not claim equivalence.
