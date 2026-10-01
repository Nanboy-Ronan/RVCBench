# Native Bark voice cloning

The measured implementation uses the serp-ai Bark fork at commit
`3b567365f650ee481c52dc5c32e55d4fddf5b6d6`, native Fairseq HuBERT layer 9,
the released learned semantic tokenizer, and Encodec reference codes.
All assets must exist locally before model preparation. Missing files fail
explicitly; the adapter does not substitute reference tokens or download weights.

Create the environment from `envs/bark-native.yml`, then install Fairseq:

```bash
cd envs
conda env create -f bark-native.yml
conda activate bark-native
cd ..
python -m pip install fairseq==0.12.2 --no-deps
git clone https://github.com/serp-ai/bark-with-voice-clone.git checkpoints/bark-with-voice-clone
git -C checkpoints/bark-with-voice-clone checkout 3b567365f650ee481c52dc5c32e55d4fddf5b6d6
```

Fairseq declares older Hydra/OmegaConf constraints. The measured runtime uses
Hydra 1.3.2 and OmegaConf 2.3.0 through native legacy task/model factories.
The recipe is an installation starting point, not a tested clean installation
lock. The measured environment was an isolated overlay on an existing Python
3.10 environment. The factory control checks all 214 state tensors and layer-9
features on one second of reference audio; it does not establish paper scores.

Supply these assets through the YAML defaults or explicit `adversary.*` overrides:

| Option | Required local asset |
| --- | --- |
| `models_dir` | Directory containing `text_2.pt`, `coarse_2.pt`, `fine_2.pt` from `suno/bark` |
| `hubert_checkpoint` | `hubert_base_ls960.pt` from `https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt` |
| `hubert_tokenizer` | `quantifier_hubert_base_ls960_14.pth` from `GitMylo/bark-voice-cloning` revision `c26e70f3311c6973ca86511dd18b6a8ee073e830` |
| `text_tokenizer_path` | Local tokenizer directory for `google-bert/bert-base-multilingual-cased`, revision `3f076fdb1ab68d5b2880cb87a0886f315b8146f8` |

The multilingual tokenizer needs `vocab.txt`, `tokenizer.json`,
`tokenizer_config.json`, and `config.json`. Provision Encodec's
`encodec_24khz-d7cc33bc.th` in the Torch Hub checkpoint cache before an offline
run. The audit records actual asset hashes; the three Bark weights were reused
read-only from existing local assets, without a claimed immutable Hub revision.
The quantizer filename's `14` does not select HuBERT layer 14: native reference
encoding uses layer 9. LoRA weights require a separate runtime and are rejected.

Run the frozen subset after placing assets at the default paths:

```bash
python run_vc.py --config-name ots_vc/clean/libritts/bark_voice_clone_ots \
  run_name=bark_libritts16 dataset.use_hf_dataset=false \
  +dataset.manifest_filename=reproduction/subsets/libritts16_v1/metadata.json \
  +vc.generate_only=true +seed=42
```

Score saved outputs using the [evaluation-only command](run_protocol.md#score-saved-audio)
in a separate evaluation environment. Subset validation does not establish
historical paper equivalence.
