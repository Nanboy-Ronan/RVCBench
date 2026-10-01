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

## Historical LibriTTS coverage

The frozen `libritts16_v1` subset has 16 matching references in each directory:

- GR-Noise: `grnoise_on_libritts/20251130-144554/protected_audio`
- DNS64 on GR-Noise:
  `denoiser_dns64_on_grnoise_libritts/20260124-145829/denoised_audio/protected_audio`

Both paths are under the read-only AudioWatermarkBench results directory.
The extra `protected_audio` level in the DNS64 output is part of the historical
directory layout. Pass that exact directory rather than searching recursively
by basename. Binding audits are retained in
`reproduction/comparisons/protected_grnoise_libritts16_bindings.json` and
`reproduction/comparisons/denoised_grnoise_dns64_libritts16_bindings.json`.
These audits prove selected-file coverage and hashes, not end-to-end stage
reproduction or metric equivalence.

The GR-Noise archive can now be used to produce verified reference outputs;
see [historical noise replay](gr_noise_replay.md). This records a producer
manifest while preserving the original experimental noise.
For explicit DNS64 production, see [the denoising stage](dns64_stage.md).

Both historical directories have now been used in real Qwen3-TTS runs on the
same 16 pairs, with all 16 outputs valid for each core metric:

| Reference variant | Generated / scored | MCD | WER | SIM |
| --- | --- | --- | --- | --- |
| GR-Noise historical audio | 16 / 16 | 6.767655 | 0.022489 | 0.452234 |
| DNS64 on GR-Noise historical audio | 16 / 16 | 5.419403 | 0.034989 | 0.456988 |

`reproduction/comparisons/reference_stage_qwen3_libritts16.json` contains run
paths, hashes, per-sample metrics and checks against the clean subset. These
are diagnostic subset results. They do not establish paper-table equivalence
or isolated synthesis timing, and producer reproduction remains pending.
