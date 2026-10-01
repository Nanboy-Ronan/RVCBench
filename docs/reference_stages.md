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
`gr_archived_noise_replay_v1` and `dns64_dataset_rate_v1` formats, the runner additionally requires complete
verification counts and checks selected sample identities and clean/output/
historical hashes against the producer rows. Such bindings are labeled
`verified_selected_output_hashes`. Historical
directories without a producer manifest are labeled
`legacy_directory_content_only`; binding them does not reproduce the protection
or denoising algorithm.

## Produce reference audio

Use [archived Gaussian-noise replay](gr_noise_replay.md) when the historical
noise archive is available, or [DNS64 production](dns64_stage.md) to denoise
an explicit reference directory. Both commands write a stage manifest that
the cloning runner verifies against the selected inputs and output bytes.
