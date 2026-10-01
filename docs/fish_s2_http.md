# Fish S2 HTTP validation

The `fish_audio_s2` integration sends requests to an independently managed
Fish Speech S2 server. Install the client with `pip install -e '.[http]'`.
For server setup and checkpoint provisioning, use [the S2 environment and
source revision](fish_s2_native.md). The measured source revision is
`214da3cd841bda85da2496b96cd3c4d7edb1337e`.

Run the official service from that checkout in its own environment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.api_server \
  --listen 127.0.0.1:18011 --workers 1 --device cuda:0 \
  --llama-checkpoint-path /absolute/path/to/s2-pro \
  --decoder-checkpoint-path /absolute/path/to/s2-pro/codec.pth \
  --decoder-config-name modded_dac_vq
```

Use a free port and retain ownership of the server process. A successful
`GET /v1/health` confirms readiness; it does not attest server weights or code.
The benchmark checks readiness before generation, requires actual reference
and target transcripts, sends seed plus original sample index, and rejects
non-WAV success responses and permanent HTTP errors. Transient retries reuse
the exact serialized request. Remote inference failure after acceptance can
still leave an unknown server-side outcome.

```bash
python run_vc.py --config-name ots_vc/clean/libritts/fish_audio_s2_ots \
  run_name=fish_s2_http_libritts16 dataset.use_hf_dataset=false \
  +dataset.manifest_filename=/absolute/path/to/reproduction/subsets/libritts16_v1/metadata.json \
  adversary.endpoint_url=http://127.0.0.1:18011/v1/tts \
  +adversary.service_code_path=/absolute/path/to/fish-speech \
  +adversary.service_checkpoint_path=/absolute/path/to/s2-pro \
  +adversary.service_codec_checkpoint_path=/absolute/path/to/s2-pro/codec.pth \
  +vc.generate_only=true +seed=42
```

Those optional local paths record source and asset hashes; they must refer to
the assets actually used by the owned server. They do not verify an arbitrary
remote server. The official server's permissive loader was independently
checked against strict native loading of the same local assets for this run.

Score saved audio in the evaluation environment with `+vc.evaluate_only=true`
and `+vc.evaluation.generated_audio_dir=/absolute/path/to/generated_audio`.
`vc.evaluation.device` takes precedence over `evaluation.device`, followed by
the generation device. HTTP generation uses CPU on the client while the
default scoring device is CUDA. Scoring provenance includes execution device,
Torch/CUDA versions, CPU thread count, and CUDA device properties, preventing
cross-device cache reuse.

All 16 fixed LibriTTS pairs were generated and scored. Their historical inputs
are matched by audio hashes and transcripts. Historical server weights,
runtime, and RNG state are unavailable, so the comparison remains diagnostic.
The historical S2 entry is experimental and outside the paper's 18-model set.

Deterministic HTTP generation remains unresolved: the first cold-service
request differs from subsequent identical-seed requests; both two-sample
warm repeats agree with the corresponding full-run outputs. These differences
are retained in `reproduction/comparisons/fish_s2_http_repeatability.json`.
Setting a request seed alone is insufficient evidence of repeatability.
Reference encoding before seed setup is a possible cause requiring a separate
controlled cold-start experiment. The validation server has been stopped;
existing services were preserved.
