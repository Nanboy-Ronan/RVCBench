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

## Regenerate the original RNG stream

Add `--regenerate-rng --seed 42 --epsilon 0.03137255 --device cuda:0` to the
command to draw new Gaussian noise for every batch in the full cohort.
Archive values serve only as comparison data in this mode. Every regenerated
batch must match its archived tensor exactly before any selected WAV is written;
a mismatch fails the stage. The generated noise then produces the selected
references, whose WAV hashes must also match the historical files.

The `gr_seeded_batch_rng_v1` manifest records the seed, standard deviation,
device and full-cohort batch verification counts. The clone runner rejects
incomplete RNG verification. Device and Torch/CUDA version can affect RNG
output; CPU is not assumed equivalent to the historical CUDA run.

On the released LibriTTS cohort, the seed-42 CUDA stream matches all 280 archived
batches, and the fixed 16 selected outputs match their historical WAVs.
This establishes the checked Gaussian-noise production path; other protection
methods require their own production and protocol validation.
