# Fixed Hub revisions and offline reproduction

Revision resolution preserves an explicit full 40-character lowercase Git
commit without a separate `model_info` request. Branches, tags, abbreviated commits
and unspecified revisions are resolved online. Offline use of such mutable
references fails with an instruction to pin a full commit first. This applies
to Qwen3-TTS, PlayDiffusion presets and OzSpeech's downloaded codec assets.
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
