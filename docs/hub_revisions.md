---
title: "Pinned model assets and offline reproduction"
description: "Resolve fixed Hugging Face revisions and verify model assets for RVCBench. Understand offline behavior and explicit checkpoint validation requirements."
---

# Fixed Hub revisions and offline reproduction

Revision resolution preserves an explicit full 40-character lowercase Git
commit without a separate `model_info` request. Branches, tags, abbreviated commits
and unspecified revisions are resolved online. Offline use of such mutable
references fails with an instruction to pin a full commit first. This applies
to Qwen3-TTS, F5-TTS checkpoints/vocoders, PlayDiffusion presets and
OzSpeech's downloaded codec assets, MaskGCT's primary and semantic checkpoints, XTTS-v2 and ZipVoice.
Snapshot downloads may still contact the Hub when online.

The pin identifies a version; it does not prove that the required files exist
in the local cache. A missing file still fails during snapshot or model loading.
The Hub's [download documentation](https://huggingface.co/docs/huggingface_hub/guides/download)
describes full commit revisions and versioned snapshots.

For Qwen3-TTS Hub checkpoints, the runner resolves the pinned snapshot to a local
directory before loading the upstream wrapper. The installed upstream wrapper
forwards loading options to its model, but loads its processor separately
without forwarding `revision`. Using the local snapshot keeps both components
on the selected version and avoids the processor's repo metadata request in
offline mode. No shared upstream package is modified.

The runner records the Hub repo and commit under `model_reference.assets`,
alongside hashes of supported asset files in the snapshot. Its effective
`adversary.checkpoint_path` is the local snapshot directory; the original
configured repo remains in the saved run configuration. Explicit local
checkpoints continue to use their actual file hashes.

With the model already cached, a fixed LibriTTS subset can be generated using:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_vc.py \
  --config-name ots_vc/clean/libritts/qwen3_tts_ots \
  dataset.use_hf_dataset=false \
  +dataset.manifest_filename=/absolute/path/to/reproduction/subsets/libritts16_v1/metadata.json \
  adversary.revision=fd4b254389122332181a7c3db7f27e918eec64e3 \
  adversary.max_samples=16 +vc.generate_only=true +seed=42
```

This is an example revision used in the retained reproduction artifacts, not
a claim that every upstream revision supports the same cloning interface.

## F5-TTS assets

F5-TTS resolves checkpoint and vocoder revisions independently. The runner
supplies local versioned assets to the upstream API, records their hashes,
and hashes the installed preset configuration and default vocabulary.
Explicit `ckpt_file`, `vocab_file` and `vocoder_local_path` overrides are retained.
Checkpoint paths preserve the logical `.safetensors` filename in Hub caches;
resolving the symlink to a bare blob filename can select the wrong loader.

With both snapshots cached, the validated default preset can run offline:

```bash
HF_HUB_OFFLINE=1 python run_vc.py \
  --config-name ots_vc/clean/libritts/f5_tts_ots \
  dataset.use_hf_dataset=false \
  +dataset.manifest_filename=reproduction/subsets/libritts16_v1/metadata.json \
  adversary.revision=84e5a410d9cead4de2f847e7c9369a6440bdfaca \
  adversary.vocoder_revision=0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21 \
  +vc.generate_only=true +seed=42
```

These pins apply to `F5TTS_v1_Base` and its Vocos vocoder. Another preset or
vocoder needs compatible revisions or explicit local assets. Current asset
provenance does not reconstruct missing checkpoint identities from legacy runs.

## MaskGCT checkpoints

The runner resolves `adversary.repo_id` and `adversary.revision` into an explicit
`adversary.checkpoint_dir` containing all six native checkpoint files. It records
their hashes before starting the worker; missing or empty files fail asset
resolution. The worker loads these local paths without downloading replacement
primary weights. Explicit local checkpoint directories bypass Hub resolution.

Offline resolution requires a full commit and all six files in the cache. The
cached revision `265c6cef07625665d0c28d2faafb1415562379dc` has been resolved and
hashed locally. This verifies asset availability, not native generation after
the backend migration or equivalence to historical weights.

The runner also resolves `adversary.semantic_revision` for
`facebook/w2v-bert-2.0` to `adversary.semantic_model_path`, containing its model
configuration, feature-extractor configuration and safetensors weights. Both
upstream semantic loaders use this same snapshot with `local_files_only=True`.
The redirection is scoped to startup inside the isolated worker and restores
the original loading methods afterward. Upstream files are not modified.
Explicit local semantic snapshots bypass Hub resolution and must contain all
three required files.

The cached semantic revision `da985ba0987f70aaeb84a80f2851cfac8c697a7b` has been
hashed and loaded with the native SDK on CPU without missing, unexpected or
mismatched weights. Its feature extractor matches the currently cached default
processor on all 16 frozen references. These checks do not establish complete
MaskGCT generation after migration or historical weight equivalence.

For offline runs, provide both full revisions after caching their required files:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_vc.py \
  --config-name ots_vc/clean/libritts/maskgct_ots \
  dataset.use_hf_dataset=false \
  +dataset.manifest_filename=reproduction/subsets/libritts16_v1/metadata.json \
  adversary.revision=265c6cef07625665d0c28d2faafb1415562379dc \
  adversary.semantic_revision=da985ba0987f70aaeb84a80f2851cfac8c697a7b \
  +vc.generate_only=true +seed=42
```

The local dataset root must contain the referenced audio. Configure the
MaskGCT worker environment through `MASKGCT_RUNTIME_PYTHON` or
`adversary.runtime_python` before running. The runner hashes local semantic normalization statistics when
`code_path` is configured. It also hashes the G2P source/data files, including
the eagerly imported Chinese ONNX model and dictionaries. An eSpeak probe uses
the worker interpreter and native-library search path; its version, selected
shared library and complete data directory are recorded, including compiled
dictionary files without extensions. Before model loading, the worker recomputes
the eSpeak binary/data hashes and checks them against the resolved configuration.
The parent also checks the startup receipt; changed resources or a missing
receipt reject startup and reap the worker. Native English tokenization has been
checked on the 32 prompt/target texts in the frozen 16-pair subset. Other-language
package resources and a clean environment lock remain incomplete; these checks
do not establish full generation or historical equivalence.

## XTTS-v2 checkpoints

The runner resolves `adversary.checkpoint` and `adversary.revision` to a local
snapshot before creating XTTS. It validates and hashes the native model, JSON
configuration and vocabulary; a preset speaker bank is also captured when
present. The native cloning path supports checkpoints without a preset speaker
bank. Explicit file overrides are validated, hashed and passed to the loader.
Missing or empty required files fail before loading weights.

The cached revision below resolves offline, and its four consumed file hashes
match the retained LibriTTS16 run. Migrated native generation and scoring remain
pending; these asset checks do not establish output equivalence or a complete
runtime environment lock.

```bash
HF_HUB_OFFLINE=1 python run_vc.py \
  --config-name ots_vc/clean/libritts/xtts_ots \
  dataset.use_hf_dataset=false \
  +dataset.manifest_filename=reproduction/subsets/libritts16_v1/metadata.json \
  adversary.revision=6c2b0d75eae4b7047358e3b6bd9325f857d43f77 \
  adversary.local_files_only=true +vc.generate_only=true +seed=42
```

Standalone `XttsGeneratorConfig` accepts the same `revision` and
`local_files_only` options. Explicit local checkpoint directories bypass Hub
resolution.

## ZipVoice assets

The runner resolves the ZipVoice model and Vocos vocoder independently, using
`adversary.revision` and `adversary.vocoder_revision`. The selected
`zipvoice` or `zipvoice_distill` directory contains the configured checkpoint,
`model.json` and `tokens.txt`. The vocoder contains `config.yaml` and
`pytorch_model.bin`. Missing or empty required files fail before loading models.
Their local paths and hashes are recorded, and both native and CLI inference
receive those explicit directories. Local `model_dir`, `checkpoint_name` and
`vocoder_path` overrides are retained.

The following cached model and vocoder snapshots resolve offline; all five
consumed file hashes match the retained Emilia-tokenizer LibriTTS16 run. This
does not establish migrated native generation equivalence or freeze all
tokenizer/package resources. The model pin below has been checked for
`zipvoice`; another model variant needs its own compatible cached files.

```bash
HF_HUB_OFFLINE=1 python run_vc.py \
  --config-name ots_vc/clean/libritts/zipvoice_ots \
  dataset.use_hf_dataset=false \
  +dataset.manifest_filename=reproduction/subsets/libritts16_v1/metadata.json \
  adversary.revision=4ed45fb6e7e9527b780bef9e097a04bf13fe4e6b \
  adversary.vocoder_revision=0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21 \
  adversary.local_files_only=true +vc.generate_only=true +seed=42
```

Provide the native ZipVoice source via `adversary.code_path`. Pin resolution
applies to the benchmark runner. Standalone `ZipVoiceGenerator` callers need
explicit local `model_dir` and `vocoder_path` to avoid implicit upstream
downloads; both runtimes recheck configured local required files before model
startup.

## SparkTTS local assets

SparkTTS's relative `model_dir` resolves under its `code_path`, matching the
native loader even if a different directory with the same name exists in the
current working directory. The runner records the resolved absolute directory
and hashes its assets. Before native imports, both runner and generator check
configuration, BiCodec weights, LLM weights/tokenizer and wav2vec2
weights/feature-extractor files. Standard single-file or indexed Hugging Face
weight layouts are accepted; every referenced shard must be present and nonempty.

Native BiCodec loading rejects missing learned state, unexpected checkpoint
entries and shape mismatches. Only the two configuration-derived registered
mel buffers, `mel_transformer.spectrogram.window` and
`mel_transformer.mel_scale.fb`, may be absent. A missing entry is accepted only as a registered non-trainable
finite buffer. Their shapes, dtypes and value hashes are recorded in each
successful sample's `native_initialization` receipt and retained during reuse.
The scoped check restores the native class method after startup or failure;
upstream source files remain unchanged.

The current local bundle's 14 asset hashes match the retained LibriTTS16 run.
Actual native BiCodec CPU loading accepts its 840 checkpoint entries with only
those two derived buffers absent, and rejects removal of a learned encoder
tensor in memory. This component check used the existing `audiobench` runtime;
the default controller environment lacks `einx`. It does not establish full
SparkTTS generation after migration, LLM/wav2vec loading completeness, a clean
runtime lock or immutable historical weights. Use a compatible SparkTTS runtime
with a complete explicit local model bundle for full generation.

## CosyVoice local assets

CosyVoice requires an explicit local `model_dir`. Discovery stays inside that
directory, and the runner binds and hashes the effective payload directory.
Runner and generator reject missing or empty core checkpoints, the selected
variant's YAML configuration and speech tokenizer before importing the native
SDK. A configured `zero_shot_spk_id` also requires a nonempty `spk2info.pt` bank;
its actual speaker membership is still checked by native inference.

CosyVoice2's native constructor overrides its Qwen pretrained path to
`CosyVoice-BlankEN` under the payload directory. The preflight therefore also
requires its configuration, vocabulary, merges and weights. Standard single-file
or indexed Transformers weights are accepted, with every declared shard present
and nonempty. The generator rechecks these files at model loading. CosyVoice1
receives only the arguments accepted by its constructor; `load_vllm` is restricted
to CosyVoice2.

These checks validate local file completeness, not tensor loading or speech
quality. CosyVoice1's YAML-selected text assets, custom YAML dependencies,
JIT/TensorRT/vLLM assets and text-frontend resources need further runtime evidence.
The previously measured native subset uses CosyVoice2 with accelerators disabled;
post-migration native subset generation remains pending.

## OpenVoice converter loading

OpenVoice's native converter uses `strict=False`. The wrapper requires a complete
finite checkpoint with matching tensor shapes instead. Missing or unexpected
keys, shape errors and nonfinite tensors fail startup. The check is scoped to
the converter's owned Torch model; other Torch modules retain their behavior,
and the original method is restored after success or failure. Startup must
perform exactly one checked load. Successful rows record the converter's key
count, strictness and compatibility result in `native_initialization`, retained
on resume and saved-audio scoring.

Actual native converter CPU startup accepts the local 486-entry checkpoint with
no missing or unexpected state. Its 32,792,226 parameters load without changing
any checkpoint tensor, and removing a learned decoder parameter in memory is
rejected. This check used the existing `audiobench` environment with a writable
Numba cache and bypassed the Melo import stage to isolate the converter.

This verifies the converter component only. Melo checkpoints, BERT/text assets,
speaker embeddings and post-migration native subset generation still require
separate evidence. A compatible full OpenVoice/Melo runtime is required; loading
the standalone converter architecture does not validate its audio API imports.

## OpenVoice Melo language models

The LibriTTS OpenVoice configuration pins the English Melo model to
`myshell-ai/MeloTTS-English` revision
`bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef`. The runner resolves its configuration
and checkpoint from that same snapshot, checks both files are nonempty, records
their hashes, and supplies their explicit local paths to the native constructor.
`adversary.local_files_only=true` uses an already cached pinned snapshot.

Configure additional uppercase language entries in `adversary.melo_models`
with `repo_id` and `revision`, or both `config_path` and `checkpoint_path` for
explicit local files. Once this mapping is configured, an absent language fails
rather than falling back to an implicit download. Standalone generator calls
require resolved local paths in the mapping. Older configurations without a
mapping retain the native implicit asset behavior as a separate unpinned path.

The actual pinned English architecture accepts all 1,051 checkpoint entries
with no missing or unexpected state; its 51,874,097 parameters match every
checkpoint tensor and all tensors are finite. This CPU component check excludes
Melo's text API and speech inference. The earlier native subset did not record
Melo asset hashes, so it does not establish historical checkpoint identity.
Transformer text assets use the separate `text_models` mapping described below.
Full native text and speech validation remains pending.

## OpenVoice English pronunciation resources

The LibriTTS configuration enables
`adversary.capture_english_text_resources=true`. Before model loading, the runner
records Melo's CMU dictionary and its existing pickle cache, the trained
`g2p_en` parameter archive, and NLTK's CMU/tagger resources. It includes both
archives checked by `g2p_en` during import and the active corpus/tagger used by
the installed NLTK version. Directory resources include extensionless files.
Missing resources fail preflight; this path does not call a downloader.

These are resources discovered in the generation interpreter. Configure
`melo_code_path` for the intended checkout and install the needed NLTK data in
that runtime before running. Matching hashes identify the observed resource
files; they do not pin their installation source or guarantee a clean runtime.
The local inventory records seven resource entries and ten files. BERT and
its tokenizers, non-English resources and migrated speech generation remain
outside this check.

## OpenVoice Transformer text assets

The LibriTTS configuration lists fixed snapshots in `adversary.text_models`,
keyed by the upstream repository IDs used by Melo. Melo eagerly imports six
language tokenizers even for English generation. The mapping provides these
tokenizers and enables weights only for English `bert-base-uncased` with
`load_model: true`. The runner validates and hashes the selected files; cached
weights for other languages are not included unless enabled.

During native imports and text inference, scoped loading redirects configured
repository IDs to their resolved local snapshot paths with `local_files_only`
and remote custom code disabled. Unlisted IDs and weight loading on a
tokenizer-only entry fail explicitly. Loader methods are restored after success
or failure. Configurations without `text_models` retain the separate legacy
implicit-loading path. Standalone generator calls require resolved local paths.

This fixes asset selection, not Transformer version compatibility or all text
resources. The observed runtime also emits a tokenizer regex warning for the
multilingual snapshot. Actual native English G2P, phone/tone/language tensors and expanded BERT features
have repeated identically for the 16 selected target texts on CPU. Historical
repo-ID tokenization equivalence and speech reproduction remain unproven.

OpenVoice tracks Melo text module objects imported by its own serial operations.
On close, it releases their native global Torch model caches as well as its
language models and reference embeddings. A replacement module or a previously
imported foreign module is not selected for cleanup. Partial import failures
still record owned modules for cleanup. The native LibriTTS16 text check verifies
that closing the generator releases the actual English BERT cache without
changing any of the checked text feature hashes.
