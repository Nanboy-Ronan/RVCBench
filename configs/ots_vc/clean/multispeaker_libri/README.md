# Multispeaker Libri OTS VC Configs

Configs in this folder mirror the LibriTTS OTS setups but target the 10-speaker Multispeaker Libri dataset.

- **Dataset config**: `configs/dataset/multispeaker_libri.yaml` (root `./data/audiobenchmarkdataset/Multispeaker_libri`, 16 kHz, 10 speakers). Set `speaker_id` if you want a single speaker; `null` uses all.
- **Models covered**: bertvits2, cosyvoice2, fishspeech, glowtts, higgs_audio, kimi_audio, mgm_omni, moss_ttsd, ozspeech, playdiffusion, qwen3_omni, sparktts, styletts2, vall_e, vibevoice, plus `eval_only`.
- **Run naming/output**: each config uses `*_on_multispeaker_libri` in `run_name` and writes metrics under `results/<run_name>/metrics.json`.

## Usage

Run VC with Hydra pointing to a config in this directory, e.g.:

```bash
python run_vc.py --config-name higgs_audio_ots.yaml --config-path configs/ots_vc/clean/multispeaker_libri
```

To evaluate pre-generated audio only, set `generated_audio_dir` in `eval_only.yaml` to your path, then:

```bash
python run_vc.py --config-name eval_only.yaml --config-path configs/ots_vc/clean/multispeaker_libri
```

## Notes

- Model-specific prompts/decoding match the LibriTTS counterparts; adjust if you want dataset-specific wording.
- Keep `return_path: true` in the dataset config so file paths are available during evaluation.
