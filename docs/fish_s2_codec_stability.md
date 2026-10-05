---
title: "Fish Audio S2 codec stability"
description: "Understand the numerical stability variant for the Fish Audio S2 codec, its motivation, explicit runtime settings and limits of comparison with native results."
---

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
