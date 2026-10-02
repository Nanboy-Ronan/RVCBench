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

Run seeds must be integers; booleans, floats and numeric strings are rejected
before backend preparation. The default remains 42 when no seed is configured.
Direct backends require a native run seed and reject disagreements between the
request, adapter configuration and generator configuration before changing RNG
state or running inference. Original source indices must be nonnegative integers. Manifests may omit
`source_index` entirely to use the existing row-order policy; when declared,
every loaded row must provide a valid integer. Validation precedes variant and
speaker filtering, so filtering does not hide malformed declarations.
`max_samples` must be a positive integer when provided.
The declared sample seed remains `run_seed + source_index`. These direct backends
support the `source_index` native seed policy. Other explicitly named historical
seed variants retain their own adapter and comparison protocol.

Qwen3-TTS accepts individual samples directly through the request backend.
Each native seed must match the request's run seed plus original source index.
Model NaN/Inf output fails that sample, and prompt caches are released when the
backend closes. Its `qwen3_generate_excluding_prompt_encoding_and_io_v1` timing
measures the native generation call, excluding prompt encoding and file writing.
This scope alone does not establish timing comparability with another adapter.
F5-TTS also accepts individual samples directly. Its
`f5_sequential_inference_excluding_transcription_and_output_write_v1` timing
includes reference preprocessing, file reading, chunk generation and vocoding;
explicit reference transcription and output WAV writing occur outside this call.
A missing, empty, nonfinite or incorrectly sampled text chunk fails the sample.
VoxCPM and VoxCPM2 use explicit sample requests while preserving their separate
reference-conditioning configurations. The recorded requested seed uses the
original source index; the effective seed may differ after a native bad-case
retry. Its `voxcpm_generate_including_native_retries_excluding_output_write_v1`
timing includes the native generation call and retries, excluding WAV writing.
Missing reference audio fails the sample.

XTTS-v2 accepts explicit samples with the same language selection, target-text
fallback and original-index seed rule as its compatibility adapter. Native
inference exceptions, missing references and empty, nonfinite or non-mono audio
fail the sample. Its
`xtts_synthesize_including_conditioning_excluding_output_write_v1` timing includes
reference conditioning and native synthesis, excluding WAV writing. The migrated
interface has contract checks; native frozen-subset validation remains pending.

ZipVoice accepts explicit samples and preserves its prompt transcript, language
selection and original source index. Missing references, missing text, native
exceptions and empty, nonfinite or non-mono audio fail the sample. Native timing
`zipvoice_native_generation_including_temporary_wav_io_excluding_output_write_v1`
includes generation and temporary WAV writing/reading, excluding final output
writing. CLI timing
`zipvoice_cli_request_including_process_startup_model_loading_and_temporary_wav_io_v1`
also includes starting a process and loading models for each request. CLI waits
are bounded by `adversary.request_timeout_sec` (900 seconds); this setting does
not interrupt in-process native inference. Native requests record the applied
seed. CLI requests record `native_requested_seed` and policy
`source_index_cli_argument`, with no claimed effective seed receipt from the
child. These scopes must not be treated as interchangeable inference timings.
Native frozen-subset validation after migration remains pending.

GLM-TTS's adapter accepts explicit samples with original-index seeds and requires
nonempty target text and the actual reference transcript. Missing conditioning
does not fall back to a default sentence. Invalid mono audio fails the sample,
and standalone calls flush existing timing records on failure. Closing releases
the LLM, flow/vocoder, frontend and speech tokenizer. At 24 kHz, HiFT must exist
under the configured checkpoint directory; its temporary environment binding is
restored after successful or failed loading. Foreign cached native namespaces
are rejected so another model's `utils` or `cosyvoice` cannot silently supply GLM
code. The pinned native speech tokenizer requires CUDA; CPU model loading fails
before imports or weight allocation. Adapter contracts and native import origins are checked; direct backend
registration and migrated native generation/scoring remain pending.

FireRedTTS2 accepts explicit monologue requests with original-index seeds and
requires both reference audio and its actual transcript. Empty target text does
not fall back to the reference transcript. Missing conditioning and invalid mono
audio fail the sample; the runner records the failure and continues. Standalone
calls flush timing records even on failure. Timing
`fireredtts2_prompted_monologue_including_token_retries_and_cleanup_excluding_output_write_v1`
includes conditioning, bounded native token retries and cleanup, excluding final
WAV writing. Decoded audio is saved at 24 kHz; the native 16 kHz value describes
prompt input. The fixed short10 `retry20` validation retains its explicit protocol
variant and does not replace the default three-attempt protocol. The current
native CPU short10/retry20 run generated all 10 samples and completed MCD/WER/SIM
scoring with audio/input/seed and row/CSV/aggregate checks. Historical GPU
scorer/runtime fingerprints differ, so it does not establish historical or
full-paper equivalence. A second native CPU generation and complete runtime
resource locking remain unverified.

StyleTTS2 accepts explicit samples with the original-index seed applied before
reference-style extraction and diffusion sampling. Text selection and the
configured alpha, beta, diffusion steps, embedding scale and tail trim retain
the existing behavior. Missing references, empty text and invalid or empty mono
audio fail the sample. Closing clears the model, sampler and cached reference
styles. Timing
`styletts2_synthesis_including_reference_style_and_diffusion_excluding_output_write_v1`
includes conditioning and diffusion, excluding final WAV writing. Reference
cache hits affect this measurement.

The main checkpoint must contain complete finite state for every constructed
module. DataParallel prefixes are normalized without collisions; missing or
unexpected tensor keys and mismatched shapes fail initialization. Extra checkpoint
modules absent from the constructed model are recorded separately and cannot
supply missing weights. Native CPU loading verified exact final state for all
13 modules, including ASR, F0 and PL-BERT after their initial pretraining loads.
The initialization receipt is retained in each generation result. Two fresh native
CPU generations of the frozen LibriTTS16 subset produce byte-identical WAVs;
MCD/WER/SIM scoring is complete for all 16 pairs with row/CSV/aggregate checks.
Historical GPU scorer/runtime fingerprints differ, so those results do not
establish historical or full-paper equivalence. Tokenization fallback and full
runtime/asset closure still require separate validation.

OpenVoice accepts explicit samples, preserving reference audio, text fallback,
language choice and source-speaker selection. The sample seed is applied before
generation using the original source index. Melo language models and reference
embeddings remain cached during the run and are released on close. Owned
Melo text modules also release their global Torch model caches. Missing
references, empty text, native exceptions and empty, nonfinite or non-mono audio
fail the sample. Temporary source WAVs are removed after success or failure.
The native base-class constructor compatibility shim is scoped to construction
and restored after success or failure.

Its native timing
`openvoice_generation_including_conditioning_lazy_language_initialization_and_temporary_io_excluding_output_write_v1`
includes reference conditioning, deferred Melo initialization and temporary
source audio I/O, excluding final output writing. Cache state and deferred
initialization affect this measurement; it is not a pure inference timer.
The migrated OpenVoice runner generated and scored all 16 frozen samples on CPU,
with every WAV matching a second fresh CPU generation. Historical GPU equivalence
remains unproven. Converter checkpoint
loading requires complete finite state with matching
shapes. Each generated row records its checked loading result in
`native_initialization`; resume and saved-audio scoring preserve it. The check
is scoped to the owned Torch model and restores its method after startup or
failure. The default LibriTTS configuration binds a pinned English Melo checkpoint and
configuration through `adversary.melo_models`; additional languages need their
own entries. The English configuration also captures pronunciation-resource hashes before
loading. The `text_models` mapping also fixes the eagerly imported tokenizers and English
BERT assets to local snapshots. Native English text features repeat for the
selected 16 inputs. Other languages and historical result
equivalence still require separate verification.
See [Melo asset setup](hub_revisions.md#openvoice-melo-language-models).

CosyVoice accepts explicit samples, preserving the original-index seed, prompt
transcript fallback and mono 16 kHz reference preprocessing. Model discovery is
limited to the supplied asset directory, including nested payloads; a missing
path does not trigger searches across unrelated checkouts. Each streaming chunk
must contain nonempty finite mono audio representable in float32. Missing or
invalid chunks fail the whole sample rather than saving partial or repaired
speech. Closing releases the model. Native timing
`cosyvoice_native_inference_and_stream_collection_excluding_reference_loading_and_output_write_v1`
includes the inference call and stream collection, excluding reference loading
and final WAV writing. Native frozen-subset validation after migration remains
pending.

SparkTTS uses each explicit sample's own prompt transcript and original-index
seed. References shared by distinct annotation rows do not share a transcript
lookup entry. Empty text, missing reference or invalid audio fails the sample.
A native failure with a prompt transcript fails by default. To replay the
historical transcript-removal retry, set
`adversary.retry_without_prompt_text=true` and give the run a distinct name,
for example `run_name=sparktts_legacy_transcript_fallback`. The fallback preserves
the existing RNG stream; it does not reseed the second inference call.
Each successful row records `conditioning_variant` as `prompt_audio_and_text`,
`prompt_audio_only` or `prompt_audio_only_after_transcript_failure`. Resume and
saved-audio scoring retain this receipt. `native_initialization` also records
the checked BiCodec loading receipt, including allowed derived mel buffer
value hashes; see [asset validation](hub_revisions.md). Controlled timing comparison rejects
transcript retries. The
`sparktts_inference_including_enabled_transcript_retry_excluding_output_write_v1`
time includes native conditioning/inference and an explicitly enabled retry,
excluding final WAV writing. Closing releases the model; imports do not create
caches inside the upstream source tree or override `HF_HOME`. The migrated
native frozen-subset campaign remains pending.

IndexTTS uses explicit requests to its persistent worker. Startup and response
waits are bounded by `adversary.startup_timeout_sec` (600 seconds) and
`adversary.request_timeout_sec` (900 seconds). Each response must confirm the
request ID, seed and exact output path; invalid or missing audio fails the
sample. Protocol errors and timeouts reap the worker and clear its ready state.
Custom worker scripts must implement this request/response contract, including
`event`, `ok`, `request_id`, `seed` and `output_path` fields on success.
Its `indextts_worker_request_including_ipc_inference_and_output_validation_v1`
time includes IPC, native inference, output writing, file readiness waiting and
audio validation. It cannot be compared with in-process inference-only timing.
MaskGCT's runner resolves and hashes its primary weights and semantic snapshot
before starting the worker. The worker requires `checkpoint_dir` and
`semantic_model_path`; its semantic model and processor use the same local snapshot.
See [asset pins and remaining auxiliary dependencies](hub_revisions.md).
MaskGCT uses the same bounded response reader and response contract as IndexTTS,
with its own prompt transcript, language and generation settings. Its startup
and response timeout settings have the same defaults. The
`maskgct_worker_request_including_ipc_inference_and_output_validation_v1` timing
includes the worker call and output validation. Closing or failing either worker
clears its model-ready state. The response reader requires POSIX selectable
process pipes. Other integrations currently use the legacy adapter bridge.

To freeze a new subset before examining model outcomes:

```bash
python scripts/freeze_reproduction_subset.py \
  --dataset-config src/rvcbench/configs/dataset/libritts.yaml \
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

## Compare request timing

New generation runs record `request_wall_time_sec` around the backend request,
including reference processing, synthesis, output writing and any deferred setup.
The runner synchronizes its requested CUDA device at both boundaries; backend
preparation is recorded separately as `backend_prepare_wall_time_sec`.
Native `synthesis_time_sec` retains its adapter-specific scope. The CSV exports
both times and the first/subsequent request phase.

```bash
rvcbench compare-timing /absolute/path/to/generation-a /absolute/path/to/generation-b
```

The check accepts generation-only runs and keeps the first request after prepare
separate from subsequent requests. It verifies matched inputs and artifact hashes,
fresh single-attempt measurements, timing-profile integrity and unchanged CPU/GPU
framework settings. Resumed or evaluation-only copies, older runs without request
profiles, and declared workers/services without hardware evidence are rejected.
This compares observed host request latency under recorded settings; it does not
establish isolated-device performance, inference-only speed or historical table
equivalence. Full model initialization is outside the request measurement, and
first/subsequent phases do not imply a standardized warmup policy.

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
