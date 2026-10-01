# Fish S2 codec numerical stability

Cold and warm requests to the pinned official S2 server initially returned
different audio for the same input and seed. Token tracing showed that the
first reference's codec tokens already differed before semantic generation.
An independent codec-only run reproduced this on two consecutive encodes.
Startup seeds, deterministic cuDNN settings, deterministic Torch algorithms,
and a fixed cuBLAS workspace did not remove it.

Layer hashes locate the first divergence at
`encoder.block.1.block.0.block.0`, a DAC `Snake1d` activation: its input hash
matches across calls while its output hash differs. All trained parameter
hashes match before and after the three-encode control. The underlying DAC
implementation calls a TorchScript function for the Snake expression.
Disabling JIT optimization makes repeated reference token hashes identical.

The opt-in `eager_snake_v1` execution variant replaces owned codec `Snake1d`
modules with that exact expression evaluated eagerly. Parameter objects and
state dictionary keys are preserved; no upstream class or shared function is
patched. The measured codec contains 58 such activations. Its repeated
encodes, encode after decode, and math-only/default attention controls match
the JIT-disabled token hash. This changes the numerical execution path, so
it remains a named variant rather than rewriting historical artifacts.

Start the owned HTTP service with
`python scripts/serve_fish_s2.py --code-path /path/to/fish-speech --stable-codec-activations -- ...`.
Use the same official arguments documented in [HTTP setup](fish_s2_http.md).
The default launcher leaves the upstream codec activation implementation
intact. Client provenance should declare the variant and both local source
files:

```text
+adversary.service_codec_variant=eager_snake_v1
+adversary.service_launcher_code_path=/absolute/path/to/scripts/serve_fish_s2.py
+adversary.service_codec_code_path=/absolute/path/to/src/models/stable_codec_activations.py
```

These declarations record code and configuration. They do not attest the
implementation of an arbitrary remote server. The controlled run verifies
the locally owned process and exact source hashes separately.

Evidence is retained in `reproduction/comparisons/fish_s2_codec_*_probe.json`:
the initial probe's unsupported Flash-only kernel failure, deterministic
controls, full layer hashes, JIT-disabled controls, and eager controls. Probe
source text is embedded in the artifacts. The full layer probe synchronizes
CUDA heavily; it is diagnostic evidence and provides no timing measurement.
HTTP validation generated and scored all 16 fixed LibriTTS pairs. The cold
two-sample canary matches the corresponding full-run outputs. A second
independent service instance generated all 16 again with identical WAV hashes.
Its only source change was an added preflight guard against unexpected Snake
state; both loaded source versions are captured. Scoring was not repeated for
byte-identical waveforms: evidence remains attached to the first scored run.
Mean MCD/WER/SIM are 5.664465/0.052010/0.570020. All three speaker-bootstrap
intervals for differences from the matched historical population include zero.

The eager runtime audit and comparison are
`reproduction/comparisons/fish_s2_http_eager_runtime_audit.json` and
`fish_s2_http_eager_libritts16.json`. This validates the measured subset and
environment, not universal determinism or full-paper table reproduction.
Both owned services were stopped after finite validation; existing services
were preserved. The original HTTP numerical path and all historical results
remain available.
