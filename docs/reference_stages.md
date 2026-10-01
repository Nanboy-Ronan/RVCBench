# Binding protected and denoised references

The zero-shot runner accepts `vc.reference_audio_dir` as an explicit directory
of replacement reference waveforms. It keeps the selected sample IDs, target
waveforms and transcripts from the clean manifest. `vc.reference_stage` names
the protocol variant; the name does not attest how the audio was produced.

For example, append these overrides to a `run_vc.py` zero-shot command:

```bash
+vc.reference_audio_dir=/absolute/path/to/protected_audio
+vc.reference_stage=protected_grnoise_legacy
```

Each selected reference must resolve uniquely under that directory as
`<speaker>/<filename>`, `<filename>`, or its manifest-relative path. Missing
references, ambiguous paths and flat files shared by different clean identities
fail before model loading. The dataset object is not modified. When a legacy
wrapper also supplies a protected directory, it must agree with the explicit
directory.

`run_manifest.json` records a `reference_stage` section with the clean reference
and replacement reference paths and SHA-256 hashes for every sample. Its
fingerprint enters the generation configuration. Resume and evaluation reject
changed lineage, including changes to the clean reference when the replacement
and target bytes remain unchanged.

An adjacent `stage_manifest.json` is hashed and recorded when present. This is
recorded producer metadata; unknown producer formats do not attest their claims.
A producer declaring an unfinished or failed status is rejected. For the known
`gr_archived_noise_replay_v1`, `gr_seeded_batch_rng_v1`,
`dns64_dataset_rate_v1` and `enkidu_audio_only_cohort_v1` formats, the runner
additionally requires complete verification counts and checks selected sample
identities and clean/output hashes against the producer rows. GR replay also
requires historical output hashes; Enkidu requires complete cohort training.
Such bindings are labeled `verified_selected_output_hashes`. Historical
directories without a producer manifest are labeled
`legacy_directory_content_only`; binding them does not reproduce the protection
or denoising algorithm.

## Produce reference audio

Use [Enkidu cohort production](enkidu_stage.md),
[archived Gaussian-noise replay](gr_noise_replay.md) when the historical
noise archive is available, or [DNS64 production](dns64_stage.md) to denoise
an explicit reference directory. These commands write a stage manifest that
the cloning runner verifies against the selected inputs and output bytes.

Legacy protectors store their noise archive at the protection run root and
write WAVs under `<speaker>/<filename>`. Enkidu requires batch size 1 and
16 kHz processing; its output WAVs retain that rate even when source audio
is 24 kHz. Its surrogate optimization retains the historical gradient
accumulation behavior. Validate full production separately from binding
an existing protected directory.

SafeSpeech/SPEC/EM surrogate loading requires the canonical 112-symbol table
and matching complete checkpoints. The default `model.checkpoint_load_policy=strict`
rejects missing or differently shaped weights before applying them. An explicit
`model.checkpoint_load_policy=legacy_partial` retains legacy initialization
fallbacks and records the affected tensors; this is a distinct diagnostic
protocol and does not establish trained-surrogate equivalence.
Text symbol imports do not prepare language models or download checkpoints.

The training loader uses `dataset.text_feature_policy=strict_cached` by default:
current-language `.bert.pt` features must be finite floating tensors with 1024
channels and the expected phoneme length. Missing or invalid caches fail instead
of fabricating features. `legacy_random` explicitly restores the old fallback;
non-current-language random channels retain the historical construction in both
policies. A shape check does not establish a cache's tokenizer/model provenance.
