# Replaying historical Gaussian protection

The historical GR-Noise run retains `protected_audio/gr.noise`, a frozen
dictionary of padded batch tensors. The benchmark can reconstruct selected
protected waveforms from that archive without loading a surrogate voice model
or training text features:

```bash
rvcbench replay-gr \
  --dataset-root /absolute/path/to/data/Libritts \
  --subset-manifest /absolute/path/to/reproduction/subsets/libritts16_v1/metadata.json \
  --noise-archive /absolute/path/to/historical/protected_audio/gr.noise \
  --historical-directory /absolute/path/to/historical/protected_audio \
  --output /absolute/path/to/new/stage-run
```

The output directory must be new. The command does not modify the clean
dataset, archive or historical results. Its default protocol uses batches of
8, 24 kHz mono PCM16 references, a PCM divisor of 32768 and a spectral hop
length of 512. The full cohort is loaded from the dataset's `metadata.parquet`
using the historical `speaker` manifest variant. The selected subset must
retain the same sample identities, paths and texts.

The full cohort matters: the original loader grouped rows per speaker, then
sorted each batch by descending reference spectrogram length. Its archived
noise slots therefore do not correspond to the positions in a smaller subset.
The replay reconstructs those positions, validates archive speaker/batch counts
and tensor shapes, adds the selected noise to the clean PCM waveform, clamps
to [-1, 1] and writes PCM16. Each resulting WAV must have the same SHA-256 as
the historical protected WAV. It does not search alternative permutations to
force a match.

Missing cohort files, changed subset records and incompatible archive shapes
fail before creating the output directory. An output mismatch produces a
failed `stage_manifest.json`; it cannot be reported as a verified stage.
The manifest records input hashes, original archive batch/slot, output and
historical hashes, runtime packages and source hashes. The generated
`protected_audio` directory can be bound through `vc.reference_audio_dir`;
the clone runner records the adjacent producer manifest hash, requires complete
verification and checks the selected input/output hashes against its rows
before model loading. Failed or stale producer outputs cannot be consumed.

This is reproduction from frozen experimental noise, not regeneration of the
original random-number stream. Historical batch construction, worker RNGs,
text-feature fallbacks and runtime versions can affect newly drawn noise.
The archived replay does not replace that outstanding reproducibility work,
nor does it establish the denoising or other protection methods' production.
