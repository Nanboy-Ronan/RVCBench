"""Canonical content data for the RVCBench homepage.

Single source of truth for every number on the page (sourced from the root
README.md). `render.py` turns this into server-rendered HTML at build time
so the numbers are present in the raw document — no JavaScript required to
see them (crawlers, AI answer engines, and no-JS browsers all get the real
content; JS only adds sorting/hover polish on top).
"""

SITE = {
    "url": "https://nanboy-ronan.github.io/RVCBench/",
    "repo": "https://github.com/Nanboy-Ronan/RVCBench",
    "paper": "https://arxiv.org/abs/2602.00443",
    "arxiv_id": "2602.00443",
    "dataset": "https://huggingface.co/datasets/Nanboy/RVCBench",
    "demo": "https://huggingface.co/spaces/Nanboy/RVCBench",
}

# The benchmark's four robustness dimensions (from the paper's framework figure,
# figs/main.png). Each maps to a real, already-built visualization further down
# the page — this section is the map, not a new chart.
DIMENSIONS = [
    {
        "key": "input",
        "token": "judge",
        "name": "Input Robustness",
        "question": "Does it still work when the reference audio or text prompt isn't clean studio speech?",
        "subtests": [
            "Reference-audio shifts — accents, ages, multi-speaker clips, café/station/train noise",
            "Text-prompt shifts — unusual, robocall-style, or hallucination-inducing prompts",
        ],
        "demo_anchor": "generalisation",
        "demo_label": "See it in the cross-dataset heatmap",
    },
    {
        "key": "generation",
        "token": "signal",
        "name": "Generation Robustness",
        "question": "Does cloning quality hold up across model architectures, languages, and utterance length?",
        "subtests": [
            "32 integration entries spanning codec-LM, diffusion, and hybrid architectures",
            "Multilingual (EN/ZH/FR), long-form generation, and emotion preservation",
        ],
        "demo_anchor": "leaderboard",
        "demo_label": "See it in the leaderboard",
    },
    {
        "key": "output",
        "token": "counter",
        "name": "Output Robustness",
        "question": "Does the cloned output survive real-world post-processing, and can it be told apart from the real speaker?",
        "subtests": [
            "Post-processing resilience — MP3/AAC/Opus compression, phone-narrowband simulation",
            "Deepfake detectability — ground-truth vs. cloned speech classification",
        ],
        "demo_anchor": None,
        "demo_label": "In the codebase (data/compression/, src/rvcbench/datasets/deepfake_preprocess.py) — not yet on the public leaderboard",
    },
    {
        "key": "perturbation",
        "token": "protect",
        "name": "Audio Perturbation Robustness",
        "question": "Can a protection method actually stop a clone — and survive an attacker trying to denoise it back out?",
        "subtests": [
            "Passive perturbation — natural multi-speaker interference and environmental noise",
            "Proactive perturbation (5 methods) and counteract perturbation (adaptive denoising)",
        ],
        "demo_anchor": "robustness",
        "demo_label": "See it in the protection-robustness chart",
    },
]

# Cross-dataset heatmap columns, regrouped by which robustness dimension each
# condition tests (same underlying numbers as the source README table — see
# CROSS_DATASET below — just organised to make the taxonomy visible).
CROSS_DATASET_GROUPS = [
    {"label": "Baseline", "dim": None, "cols": ["LibriTTS"]},
    {"label": "Input Robustness", "dim": "input", "cols": ["VCTK", "BG-clean", "BG-noise", "Halluc."]},
    {"label": "Generation Robustness", "dim": "generation", "cols": ["Long", "AISHELL", "French", "Bilingual"]},
]

# "Why RVCBench" comparison — a real table (not two parallel lists) so each
# row pairs its two values directly, both visually and when read as plain text.
WHY_COMPARISON = [
    {"dim": "Adversary models", "typical": "1–3", "rvcbench": "32 integration entries; 22 models with paper results"},
    {"dim": "Datasets / languages", "typical": "1", "rvcbench": "10, incl. ZH / FR / bilingual / noisy"},
    {"dim": "Protection methods compared", "typical": "usually own only", "rvcbench": "5, equal footing"},
    {"dim": "Denoising-adaptive attacker", "typical": "rarely modeled", "rvcbench": "built into the pipeline"},
    {"dim": "Metrics", "typical": "ad hoc", "rvcbench": "standardised + bootstrap CIs"},
    {"dim": "Reproducibility", "typical": "custom scripts", "rvcbench": "one Hydra pipeline, public HF data"},
]

STATS = [
    {"n": "32", "l": "VC / TTS models"},
    {"n": "5", "l": "protection methods"},
    {"n": "10", "l": "dataset configs"},
    {"n": "204", "l": "speakers (paper)"},
    {"n": "14,370", "l": "utterances (paper)"},
    {"n": "3", "l": "languages — EN / ZH / FR"},
]

# Results from the paper (arXiv v3), two decimals as published; † marks models in its appendix.
# Leaderboard — LibriTTS, clean prompts (Table 13). Pre-sorted by SIM desc (the default view).
LEADERBOARD = [
    {"m": "Qwen3-TTS", "sim": 0.61, "wer": 0.05, "mos": 4.39, "mcd": 5.79, "rtf": 2.02, "sva": 0.97, "emo": 0.73},
    {"m": "IndexTTS", "sim": 0.61, "wer": 0.05, "mos": 4.06, "mcd": 6.61, "rtf": 2.23, "sva": 0.97, "emo": 0.69},
    {"m": "dots.tts \u2020", "sim": 0.6, "wer": 0.06, "mos": 4.17, "mcd": 6.11, "rtf": 0.67, "sva": 0.96, "emo": 0.71},
    {"m": "CosyVoice 2", "sim": 0.58, "wer": 0.05, "mos": 4.37, "mcd": 6.02, "rtf": 4.81, "sva": 0.97, "emo": 0.7},
    {"m": "ZipVoice", "sim": 0.58, "wer": 0.05, "mos": 4.13, "mcd": 7.09, "rtf": 1.46, "sva": 0.95, "emo": 0.68},
    {"m": "MOSS-TTS v1.5 \u2020", "sim": 0.57, "wer": 0.06, "mos": 4.32, "mcd": 6.66, "rtf": 0.76, "sva": 0.95, "emo": 0.7},
    {"m": "GLM-TTS", "sim": 0.57, "wer": 0.09, "mos": 4.08, "mcd": 6.41, "rtf": 1.74, "sva": 0.95, "emo": 0.68},
    {"m": "MaskGCT", "sim": 0.57, "wer": 0.09, "mos": 3.93, "mcd": 6.91, "rtf": 1.36, "sva": 0.94, "emo": 0.68},
    {"m": "Higgs TTS 3 \u2020", "sim": 0.56, "wer": 0.05, "mos": 4.23, "mcd": 6.32, "rtf": 0.64, "sva": 0.96, "emo": 0.71},
    {"m": "F5-TTS", "sim": 0.56, "wer": 0.12, "mos": 3.99, "mcd": 6.96, "rtf": 0.61, "sva": 0.94, "emo": 0.68},
    {"m": "Higgs Audio", "sim": 0.56, "wer": 0.25, "mos": 4.3, "mcd": 6.06, "rtf": 1.42, "sva": 0.94, "emo": 0.72},
    {"m": "Fish Audio S2 \u2020", "sim": 0.54, "wer": 0.04, "mos": 4.37, "mcd": 6.16, "rtf": 4.67, "sva": 0.95, "emo": 0.71},
    {"m": "MGM-Omni", "sim": 0.54, "wer": 0.09, "mos": 4.28, "mcd": 5.82, "rtf": 0.84, "sva": 0.93, "emo": 0.68},
    {"m": "PlayDiffusion", "sim": 0.51, "wer": 0.05, "mos": 4.15, "mcd": 8.06, "rtf": 0.73, "sva": 0.94, "emo": 0.68},
    {"m": "MOSS-TTSD", "sim": 0.49, "wer": 0.38, "mos": 4.1, "mcd": 7.09, "rtf": 0.62, "sva": 0.88, "emo": 0.67},
    {"m": "VibeVoice", "sim": 0.48, "wer": 0.23, "mos": 3.83, "mcd": 6.76, "rtf": 1.86, "sva": 0.85, "emo": 0.62},
    {"m": "FishSpeech", "sim": 0.47, "wer": 0.17, "mos": 4.37, "mcd": 6.47, "rtf": 3.61, "sva": 0.91, "emo": 0.68},
    {"m": "XTTS-v2", "sim": 0.45, "wer": 0.07, "mos": 3.81, "mcd": 8.62, "rtf": 0.62, "sva": 0.91, "emo": 0.64},
    {"m": "Spark-TTS", "sim": 0.41, "wer": 0.33, "mos": 4.06, "mcd": 5.83, "rtf": 1.56, "sva": 0.76, "emo": 0.67},
    {"m": "OZSpeech", "sim": 0.39, "wer": 0.06, "mos": 3.21, "mcd": 6.87, "rtf": 8.75, "sva": 0.84, "emo": 0.64},
    {"m": "OpenVoice V2", "sim": 0.24, "wer": 0.07, "mos": 4.3, "mcd": 7.06, "rtf": 0.08, "sva": 0.47, "emo": 0.6},
    {"m": "StyleTTS 2", "sim": 0.23, "wer": 0.05, "mos": 4.3, "mcd": 6.81, "rtf": 0.11, "sva": 0.39, "emo": 0.59},
]

# Protection robustness — SIM, LibriTTS (Tables 35-37). ss=SafeSpeech ek=Enkidu sp=Spectral gr=GR-Noise em=EM (POP)
ROBUSTNESS = [
    {"m": "Qwen3-TTS", "clean": 0.61, "ss": 0.38, "ek": 0.5, "sp": 0.36, "gr": 0.41, "em": 0.58},
    {"m": "IndexTTS", "clean": 0.61, "ss": 0.35, "ek": 0.47, "sp": 0.32, "gr": 0.39, "em": 0.57},
    {"m": "dots.tts \u2020", "clean": 0.6, "ss": 0.41, "ek": 0.49, "sp": 0.39, "gr": 0.44, "em": 0.57},
    {"m": "CosyVoice 2", "clean": 0.58, "ss": 0.32, "ek": 0.45, "sp": 0.3, "gr": 0.38, "em": 0.55},
    {"m": "ZipVoice", "clean": 0.58, "ss": 0.29, "ek": 0.44, "sp": 0.26, "gr": 0.26, "em": 0.54},
    {"m": "MOSS-TTS v1.5 \u2020", "clean": 0.57, "ss": 0.33, "ek": 0.43, "sp": 0.31, "gr": 0.33, "em": 0.53},
    {"m": "GLM-TTS", "clean": 0.57, "ss": 0.33, "ek": 0.44, "sp": 0.31, "gr": 0.39, "em": 0.53},
    {"m": "MaskGCT", "clean": 0.57, "ss": 0.3, "ek": 0.41, "sp": 0.28, "gr": 0.31, "em": 0.53},
    {"m": "Higgs TTS 3 \u2020", "clean": 0.56, "ss": 0.48, "ek": 0.49, "sp": 0.48, "gr": 0.34, "em": 0.53},
    {"m": "F5-TTS", "clean": 0.56, "ss": 0.21, "ek": 0.43, "sp": 0.18, "gr": 0.14, "em": 0.52},
    {"m": "Higgs Audio", "clean": 0.56, "ss": 0.26, "ek": 0.43, "sp": 0.24, "gr": 0.27, "em": 0.52},
    {"m": "Fish Audio S2 \u2020", "clean": 0.54, "ss": 0.32, "ek": 0.43, "sp": 0.3, "gr": 0.34, "em": 0.52},
    {"m": "MGM-Omni", "clean": 0.54, "ss": 0.18, "ek": 0.32, "sp": 0.17, "gr": 0.23, "em": 0.49},
    {"m": "PlayDiffusion", "clean": 0.51, "ss": 0.17, "ek": 0.34, "sp": 0.15, "gr": 0.16, "em": 0.47},
    {"m": "MOSS-TTSD", "clean": 0.49, "ss": 0.24, "ek": 0.34, "sp": 0.22, "gr": 0.25, "em": 0.08},
    {"m": "VibeVoice", "clean": 0.48, "ss": 0.27, "ek": 0.37, "sp": 0.25, "gr": 0.28, "em": 0.45},
    {"m": "FishSpeech", "clean": 0.47, "ss": 0.24, "ek": 0.33, "sp": 0.21, "gr": 0.23, "em": 0.01},
    {"m": "XTTS-v2", "clean": 0.45, "ss": 0.26, "ek": 0.31, "sp": 0.24, "gr": 0.24, "em": 0.41},
    {"m": "Spark-TTS", "clean": 0.41, "ss": 0.13, "ek": 0.14, "sp": 0.11, "gr": 0.06, "em": 0.36},
    {"m": "OZSpeech", "clean": 0.39, "ss": 0.16, "ek": 0.19, "sp": 0.15, "gr": 0.15, "em": 0.34},
    {"m": "OpenVoice V2", "clean": 0.24, "ss": 0.18, "ek": 0.19, "sp": 0.18, "gr": 0.18, "em": 0.24},
    {"m": "StyleTTS 2", "clean": 0.23, "ss": 0.09, "ek": 0.12, "sp": 0.08, "gr": 0.03, "em": 0.21},
]
ROBUSTNESS_METHOD_NAME = {"ss": "SafeSpeech", "ek": "Enkidu", "sp": "Spectral", "gr": "GR-Noise", "em": "EM"}

# Cross-dataset generalisation — SIM, clean prompts (Tables 11, 13, 33, 38, 39, 41). The paper reports
# multi-speaker results per SNR level only (Tables 29-32), so they are not a column here.
CROSS_DATASET_COLUMNS = ["LibriTTS", "VCTK", "Long", "AISHELL", "French", "Bilingual", "BG-clean", "BG-noise", "Halluc."]
CROSS_DATASET = [
    {"m": "Qwen3-TTS", "v": [0.61, 0.62, 0.56, 0.72, 0.54, 0.67, 0.69, 0.57, 0.51]},
    {"m": "IndexTTS", "v": [0.61, 0.57, 0.78, 0.72, 0.4, 0.67, 0.59, 0.53, 0.53]},
    {"m": "dots.tts \u2020", "v": [0.6, 0.57, 0.72, 0.68, 0.46, 0.61, 0.65, 0.56, 0.54]},
    {"m": "CosyVoice 2", "v": [0.58, 0.58, 0.53, 0.72, 0.38, 0.65, 0.63, 0.51, 0.52]},
    {"m": "ZipVoice", "v": [0.58, 0.55, 0.73, 0.71, 0.36, 0.63, 0.62, 0.46, 0.51]},
    {"m": "MOSS-TTS v1.5 \u2020", "v": [0.57, 0.51, 0.66, 0.68, 0.46, 0.65, 0.58, 0.46, 0.46]},
    {"m": "GLM-TTS", "v": [0.57, 0.57, 0.76, 0.69, 0.4, 0.66, 0.62, 0.53, 0.53]},
    {"m": "MaskGCT", "v": [0.57, 0.56, 0.76, 0.67, 0.49, 0.63, 0.61, 0.49, 0.5]},
    {"m": "Higgs TTS 3 \u2020", "v": [0.56, 0.45, 0.74, 0.63, 0.47, 0.3, 0.49, 0.31, 0.4]},
    {"m": "F5-TTS", "v": [0.56, 0.54, 0.61, 0.7, 0.3, 0.65, 0.58, 0.41, 0.46]},
    {"m": "Higgs Audio", "v": [0.56, 0.52, 0.52, 0.58, 0.35, 0.54, 0.59, 0.42, 0.42]},
    {"m": "Fish Audio S2 \u2020", "v": [0.54, 0.51, 0.5, 0.66, 0.46, 0.62, 0.58, 0.48, 0.41]},
    {"m": "MGM-Omni", "v": [0.54, 0.45, 0.44, 0.71, 0.23, 0.63, 0.52, 0.33, 0.4]},
    {"m": "PlayDiffusion", "v": [0.51, 0.43, 0.64, 0.44, 0.28, 0.46, 0.43, 0.31, 0.41]},
    {"m": "MOSS-TTSD", "v": [0.49, 0.44, 0.64, 0.44, 0.33, 0.44, 0.49, 0.04, 0.42]},
    {"m": "VibeVoice", "v": [0.48, 0.44, 0.62, 0.56, 0.34, 0.53, 0.51, 0.36, 0.41]},
    {"m": "FishSpeech", "v": [0.47, 0.43, 0.57, 0.61, 0.37, 0.57, 0.5, 0.39, 0.35]},
    {"m": "XTTS-v2", "v": [0.45, 0.45, 0.61, 0.57, 0.45, 0.51, 0.55, 0.39, 0.49]},
    {"m": "Spark-TTS", "v": [0.41, 0.53, 0.35, 0.57, 0.16, 0.48, 0.59, 0.33, 0.34]},
    {"m": "OZSpeech", "v": [0.39, 0.25, 0.42, 0.2, 0.11, 0.17, 0.27, 0.16, 0.28]},
    {"m": "OpenVoice V2", "v": [0.24, 0.39, 0.28, 0.43, 0.27, 0.3, 0.48, 0.36, 0.36]},
    {"m": "StyleTTS 2", "v": [0.23, 0.24, 0.2, 0.11, 0.11, 0.21, 0.2, 0.17, 0.18]},
]

import json as _json
from pathlib import Path as _Path
MODELS = _json.loads((_Path(__file__).resolve().parents[2] / "src/rvcbench/benchmark/model_catalog.json").read_text())

PROTECTIONS = [
    {"n": "SafeSpeech", "desc": "Adversarial perturbation optimised against a surrogate VC model."},
    {"n": "Enkidu", "desc": "Perceptual-loss adversarial perturbation."},
    {"n": "EM", "desc": "Expectation–Maximisation perturbation."},
    {"n": "GRNoise", "desc": "Gaussian random noise — no surrogate model required."},
    {"n": "Spectral", "desc": "SafeSpeech's spectral perturbation mode."},
]

DATASETS = [
    {"k": "Libritts", "lang": "EN", "desc": "English zero-shot VC/TTS benchmark prompts."},
    {"k": "VCTK", "lang": "EN", "desc": "Multi-speaker English voice cloning."},
    {"k": "Multispeaker_libri", "lang": "EN", "desc": "Multi-speaker LibriSpeech-style evaluation."},
    {"k": "Long_context", "lang": "EN", "desc": "Longer-context voice-cloning prompts."},
    {"k": "AISHELL1_dev", "lang": "ZH", "desc": "Mandarin speech evaluation."},
    {"k": "CommonVoiceFR_dev", "lang": "FR", "desc": "French speech evaluation."},
    {"k": "Bilingual_uedin", "lang": "EN/ZH", "desc": "Bilingual speech evaluation."},
    {"k": "Background_noise", "lang": "EN", "desc": "Noisy-prompt robustness."},
    {"k": "robotcall", "lang": "EN", "desc": "Robocall-style speech robustness."},
    {"k": "vctk_text_robust", "lang": "EN", "desc": "Text robustness on VCTK-style prompts."},
]

FAQ = [
    {
        "q": "What four dimensions of robustness does RVCBench test?",
        "a": "Input Robustness (reference-audio and text-prompt shifts), Generation Robustness (model "
             "architecture, multilingual, long-form, and expressive generalisation), Output Robustness "
             "(post-processing resilience and deepfake detectability), and Audio Perturbation Robustness "
             "(passive noise, proactive protection methods, and counteract/denoising attacks).",
    },
    {
        "q": "What is RVCBench?",
        "a": "RVCBench is a benchmark for voice-cloning robustness, speaker privacy, and audio-protection methods. "
             "It provides 32 integration entries and paper results for 22 models (18 in the main results, 4 in the appendix), with 5 audio-protection methods across "
             "10 dataset configurations, scoring speaker similarity, intelligibility, perceptual quality, and runtime.",
    },
    {
        "q": "How many voice-cloning models does RVCBench evaluate?",
        "a": "The RVCBench codebase includes 32 TTS/VC integration entries. The paper (arXiv v3) reports "
             "results for 18 of those models across 18 robustness evaluations, 204 speakers, and 14,370 utterances.",
    },
    {
        "q": "What audio-protection methods does RVCBench compare?",
        "a": "Five methods on equal footing: SafeSpeech (adversarial perturbation against a surrogate VC model), "
             "Enkidu (perceptual-loss adversarial perturbation), EM (Expectation–Maximisation perturbation), "
             "Spectral (SafeSpeech's spectral perturbation mode), and GR-Noise (Gaussian random noise).",
    },
    {
        "q": "Which model is hardest to clone under protection, according to RVCBench?",
        "a": "Across the LibriTTS leaderboard, StyleTTS 2 and OpenVoice V2 have the lowest clean speaker similarity "
             "and drop furthest under protection — GR-Noise pushes StyleTTS 2's similarity from 0.23 down to 0.03.",
    },
    {
        "q": "Is the RVCBench dataset public?",
        "a": "Yes. The benchmark dataset is hosted on Hugging Face at huggingface.co/datasets/Nanboy/RVCBench under "
             "a CC0-1.0 license, with 10 dataset configurations spanning English, Mandarin, and French.",
    },
    {
        "q": "How do I cite RVCBench?",
        "a": "Cite the NeurIPS 2026 paper: Jin, Ruinan; Liao, Xinting; Yu, Hanlin; Pandya, Deval; Li, Xiaoxiao. "
             "“RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models.” "
             "Advances in Neural Information Processing Systems (NeurIPS), 2026. arXiv:2602.00443.",
    },
]

CITATION = {
    "title": "RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models",
    "authors": ["Ruinan Jin", "Xinting Liao", "Hanlin Yu", "Deval Pandya", "Xiaoxiao Li"],
    "year": "2026",
    "venue": "Advances in Neural Information Processing Systems",
    "conference": "NeurIPS 2026",
    "arxiv_id": "2602.00443",
}
