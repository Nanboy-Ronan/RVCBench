# Native Fish Audio S2 runtime

The S2 integration uses its own native checkout, separate from the historical
S1 adapter. The source revision is `214da3cd841bda85da2496b96cd3c4d7edb1337e`.
Its native conversion maps the released `fish_qwen3_omni` sharded checkpoint
into DualARTransformer weights. Both the main model and codec load strictly.
The six legacy codec RoPE/causal-mask caches are accepted only when they are
registered nonpersistent buffers and equal the corresponding region of the
native reconstruction, including dtype. Missing trained weights remain errors.

```bash
git clone https://github.com/fishaudio/fish-speech.git checkpoints/fish-speech-s2-native
git -C checkpoints/fish-speech-s2-native checkout 214da3cd841bda85da2496b96cd3c4d7edb1337e
cd envs
conda env create -f fish-s2-native.yml
conda activate fish-s2-native
cd ..
```

Provision the released S2-Pro checkpoint in `checkpoints/s2-pro`, including
the tokenizer, config, shard index, both safetensors shards, and `codec.pth`.
Alternatively override `adversary.llama_checkpoint_path` and
`adversary.decoder_checkpoint_path` to existing local assets. The environment
recipe is a starting point, not a tested clean-install lock. The measured
environment is a private overlay on an existing Python 3.12/Torch 2.8 install.

```bash
python run_vc.py --config-name ots_vc/clean/libritts/fishspeech_s2_ots \
  run_name=fish_s2_libritts16 dataset.use_hf_dataset=false \
  +dataset.manifest_filename=reproduction/subsets/libritts16_v1/metadata.json \
  +vc.generate_only=true +seed=42
```

Native reference loading, semantic generation and codec decoding run serially
on the calling thread. This preserves native generation functions while
avoiding the upstream worker's unbounded initialization wait and global Hydra
reset. Resources belong to the adapter and are released on close; no shared
endpoint is restarted. Use `compile=false` and `half=false` (BF16 main model).
The native sampler's `top_k` default is 30. Its repetition-penalty request
argument is accepted upstream but has no use in the current generation function;
the adapter does not claim that changing it alters generation. The native
engine also accepts `normalize` without applying a normalization transform.

Actual reference and target transcripts are required. Empty target text is an
error, rather than a request to synthesize the reference sentence. Scoring runs
separately in the evaluation environment, after generation resources close.
The HTTP S2 integration remains a separate entry requiring a verified S2 server.
