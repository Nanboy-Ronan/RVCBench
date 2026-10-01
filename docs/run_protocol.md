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
CosyVoice also checks the exact Transformers pin in its upstream
`requirements.txt` before model loading. The validated local compatibility
combination is Transformers 4.51.3 / tokenizers 0.21.4 on Python 3.11. A 4.57.3 /
0.22.2 runtime completed inference but produced a severe quality regression;
see `reproduction/comparisons/cosyvoice_transformers451_canary.json` for the
matched diagnostic samples. The environment recipe remains a template; the
local check used an isolated overlay, not a clean dependency lock installation.
Upstream requirement files and statically imported distributions now contribute
to the generation fingerprint, including auxiliary source trees. This is
conservative and can include optional training dependencies.
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

ZipVoice defaults to retained native inference: weights and vocoder are loaded
once in preparation and released before scoring. Setting `adversary.runtime_python`
selects CLI inference in that interpreter; `adversary.execution_backend` can
explicitly select `native` or `cli`. Native inference requires the current Python
interpreter. Interpreter paths preserve virtual-environment symlinks so that the
requested environment is actually used, including for MaskGCT and IndexTTS workers.
CLI inference starts a process and loads weights per sample; its reported synthesis
time includes that overhead, while native preparation is outside per-sample timing.
Do not compare their RTF as equivalent timing protocols.

The published ZipVoice checkpoint uses the `emilia` tokenizer even when evaluated
on LibriTTS. Tokenizer selection follows the trained checkpoint, not the dataset
name. The historical March 2026 clean run also used `emilia`; the former `libritts`
default produced a severe content regression. Explicit tokenizer overrides remain
available for custom checkpoints. Invalid names fail before weight loading.

MOSS-TTS v1.5 uses Transformers 5.x processor APIs and an auxiliary audio
tokenizer. Set `adversary.codec_path` to a local tokenizer snapshot to avoid an
implicit Hub download. Configured checkpoint directories now hash their Python
implementations alongside weights, vocabulary and configuration files; changing
checkpoint code or codec weights changes the model fingerprint. Unconfigured
downloads remain outside this asset coverage.

Higgs Audio loads `examples/generation.py` from the configured checkout by file
path, so another installed `examples` package cannot shadow that entrypoint.
Before loading it, the wrapper checks Transformers against that checkout's declared
requirement. The validated overlay uses Transformers 4.46.3 and tokenizers 0.20.3.
Model assets also hash configured `audio_tokenizer_path` and `scene_prompt_path`.
The historical November 2025 log omitted `reference_role`; reproduction explicitly
uses `assistant`, the default in the nearest preceding source revision. This is
source-derived historical alignment, not an immutable record of the executed code.
The current clean YAML's explicit `user` reference role remains a distinct setting.

StyleTTS2's model fingerprint follows `ASR_config`, `ASR_path`, `F0_path` and
`PLBERT_dir` in its YAML, resolving relative paths against the configured upstream
checkout. This includes `.t7` checkpoints and PL-BERT implementation/configuration.
Legacy pickle loading is scoped to serial upstream model preparation, and the
original `torch.load` is restored on success or failure. The primary checkpoint
uses an explicit `weights_only=False` argument. Other implicit Hub assets remain
outside the configured local asset coverage.

GLM-TTS preparation and generation both run inside the configured upstream working
directory and CUDA device context, restoring the previous directory/device afterward.
This keeps relative frontend rules and upstream `.cuda()` calls consistent with the
requested runtime. Model fingerprints include `ckpt_dir`, `frontend_dir` and upstream
`configs/`, including JSONL text rules and FST files where present. Normalizer caches
inside installed third-party packages remain outside this configured asset coverage.
The historical January 2026 checkpoint directory is no longer present locally;
matching its current replacement to those historical weights is unproven.

MOSS-TTSD inspects the local checkpoint's `auto_map` and uses its declared
`AutoModel` class when present. The legacy Asteroid loader remains available for
older checkpoints without that declaration. Missing or mismatched model parameters
are rejected; explicitly declared tied weights must refer to the loaded source
parameter. The validated native checkpoint runtime uses Transformers 5.0.0:
4.53.2 lacks its `tie_weights` interface, and 5.14.1 removed a cache helper used
by its custom generation code. `envs/moss-ttsd.yml` records the key runtime pins;
the isolated overlay has been tested, but a clean installation is unverified.
The wrapper supplies the tokenizer's actual padding ID to the native config and
requests a full generated sequence, because the upstream batch helper already
slices off the reference prefix.

MOSS-TTSD preserves historical reference-ASR behavior by default
(`use_prompt_transcript=false`). Set `reference_asr_model` to an explicit local
Whisper checkpoint for asset hashing, or select `use_prompt_transcript=true` to
use the manifest's reference text. These are distinct generation protocols.
ASR loads during preparation and is released with the generator; an unused ASR
asset is excluded when manifest text is selected. The current native checkpoint
differs from the legacy loader's architecture, and immutable historical weights
are absent, so a matched-input comparison does not establish historical weight
equivalence.

Fatal CUDA device assertions and illegal memory accesses terminate a run after
journaling the failed sample. Subsequent samples stay pending, and scorers are
not launched in the invalid context. Ordinary per-sample errors continue to use
the configured retry policy. An empty dataset now reports the dataset root and
asks the caller to check `manifest_filename`, before accessing variant metadata.

MGM-Omni's legacy loader chooses its architecture from the checkpoint directory
name. For local checkpoints declaring `model_type=MGMTTS`, RVCBench selects the
TTS branch from that metadata, so a Hub snapshot directory named by revision hash
loads correctly. The upstream name hook, Torch initialization overrides and
Transformers generation initializer are restored after preparation, including
on errors. Generation re-enters the upstream initializer within a scoped CUDA
device context. Main model, tokenizer and fallback Whisper references are released
before scoring. This remains a serial compatibility bridge.

Configured `repo_root` checkouts now contribute Python source and dependency
files to generation provenance, and `cosyvoice_path` contributes local auxiliary
weights/configuration to the model fingerprint. The validated MGM runtime uses
Torch 2.6.0, Transformers 4.52.3 and flash-attn 2.7.4.post1; its recipe records
key pins rather than a fully resolved clean-install lock. Historical runs selected
a Hub model name without an immutable revision, so current snapshot hashes alone
do not establish equality to historical weights. Content failures remain in the
fixed subset and are included in aggregate metrics.

PlayDiffusion resolves its preset before initializing the upstream engine. A
configured local preset bypasses default Hub checkpoint selection and initializes
one model manager. The wrapper overrides the engine's preset method on its own
subclass, leaving the upstream class unchanged, and releases engine references
before scoring. Hub presets launched through the benchmark runner resolve an
immutable revision before download; `cache_dir` is passed directly to Hub APIs
without changing process-wide cache environment variables.

All six named preset assets contribute individual hashes, including the extensionless
vocoder, `.npy` k-means centers and `.pkl` inpainter. Filename overrides participate
in both loading and fingerprinting. The validated runtime uses Torch 2.6.0,
Transformers 4.57.3 and fairseq2 0.4.4; its key-pin recipe has not been tested as
a clean installation. The upstream constructor still checks/downloads its NLTK
tagger resource, which is not yet included in configured asset hashes. Historical
Hub revision and environment equivalence are also unproven; these limits prevent
a claim of fully frozen generation provenance.

OZSpeech places ZACT and both FACodec modules explicitly on the requested device
and logs their actual parameter devices after loading. Loading tensors with a
CUDA `map_location` alone does not move a newly constructed model's parameters.
Main/codec references are released before scoring. Its OmegaConf checkpoint
allowlist exists only during the upstream `weights_only=True` load; existing
caller-owned allowlist entries are preserved on success and failure.

Both codec path spellings (`codec_*_path`, `facodec_*_path`) are accepted, including
an alias when the primary YAML field is null. Local codec weights and the upstream
`zact/lexicon/librispeech-lexicon.txt` are hashed. Missing codec paths resolve the
two default Hub files at one immutable revision before loading. The validated
runtime uses Torch 2.8.0, Transformers 4.57.3 and Lightning 2.5.3. Its environment
recipe records key pins rather than a clean-install lock. G2P/NLTK package resource
assets and historical immutable weights/runtime are not yet established, so
matched-input generation and scoring comparisons remain evidence-bounded.
