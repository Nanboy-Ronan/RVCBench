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
            "27 adversary models spanning codec-LM, diffusion, and hybrid architectures",
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
        "demo_label": "In the codebase (data/compression/, src/datasets/deepfake_preprocess.py) — not yet on the public leaderboard",
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
    {"label": "Input Robustness", "dim": "input", "cols": ["VCTK", "Multi-spk", "BG-clean", "BG-noise", "Halluc."]},
    {"label": "Generation Robustness", "dim": "generation", "cols": ["Long", "AISHELL", "French", "Bilingual"]},
]

# "Why RVCBench" comparison — a real table (not two parallel lists) so each
# row pairs its two values directly, both visually and when read as plain text.
WHY_COMPARISON = [
    {"dim": "Adversary models", "typical": "1–3", "rvcbench": "26, zero-shot + fine-tuning"},
    {"dim": "Datasets / languages", "typical": "1", "rvcbench": "10, incl. ZH / FR / bilingual / noisy"},
    {"dim": "Protection methods compared", "typical": "usually own only", "rvcbench": "5, equal footing"},
    {"dim": "Denoising-adaptive attacker", "typical": "rarely modeled", "rvcbench": "built into the pipeline"},
    {"dim": "Metrics", "typical": "ad hoc", "rvcbench": "standardised + bootstrap CIs"},
    {"dim": "Reproducibility", "typical": "custom scripts", "rvcbench": "one Hydra pipeline, public HF data"},
]

STATS = [
    {"n": "27", "l": "VC / TTS models"},
    {"n": "5", "l": "protection methods"},
    {"n": "10", "l": "dataset configs"},
    {"n": "225", "l": "speakers (paper)"},
    {"n": "14,370", "l": "utterances (paper)"},
    {"n": "3", "l": "languages — EN / ZH / FR"},
]

# Leaderboard — LibriTTS, clean prompts. Pre-sorted by SIM desc (the default view).
LEADERBOARD = [
    {"m": "Qwen3-TTS",    "sim": .614, "wer": .052, "mos": 4.39, "mcd": 5.79, "rtf": 2.02, "sva": .974, "emo": .731},
    {"m": "IndexTTS",     "sim": .606, "wer": .052, "mos": 4.06, "mcd": 6.61, "rtf": 2.23, "sva": .972, "emo": .693},
    {"m": "CosyVoice 2",  "sim": .602, "wer": .175, "mos": 4.39, "mcd": 6.17, "rtf": 4.58, "sva": .974, "emo": .729},
    {"m": "ZipVoice",     "sim": .579, "wer": .053, "mos": 4.13, "mcd": 7.09, "rtf": 1.46, "sva": .952, "emo": .675},
    {"m": "MaskGCT",      "sim": .570, "wer": .088, "mos": 3.93, "mcd": 6.91, "rtf": 1.36, "sva": .939, "emo": .682},
    {"m": "GLM-TTS",      "sim": .570, "wer": .087, "mos": 4.08, "mcd": 6.41, "rtf": 1.74, "sva": .951, "emo": .678},
    {"m": "F5-TTS",       "sim": .559, "wer": .116, "mos": 3.99, "mcd": 6.96, "rtf": 0.61, "sva": .937, "emo": .676},
    {"m": "Higgs Audio",  "sim": .559, "wer": .250, "mos": 4.30, "mcd": 6.06, "rtf": 1.42, "sva": .941, "emo": .717},
    {"m": "MGM-Omni",     "sim": .539, "wer": .095, "mos": 4.28, "mcd": 5.82, "rtf": 0.84, "sva": .933, "emo": .676},
    {"m": "PlayDiffusion","sim": .506, "wer": .055, "mos": 4.15, "mcd": 8.06, "rtf": 0.73, "sva": .936, "emo": .681},
    {"m": "MOSS-TTSD",    "sim": .492, "wer": .383, "mos": 4.10, "mcd": 7.09, "rtf": None, "sva": .876, "emo": .667},
    {"m": "VibeVoice",    "sim": .480, "wer": .228, "mos": 3.83, "mcd": 6.76, "rtf": 1.86, "sva": .852, "emo": .624},
    {"m": "FishSpeech",   "sim": .472, "wer": .166, "mos": 4.37, "mcd": 6.47, "rtf": 3.61, "sva": .907, "emo": .682},
    {"m": "XTTS-v2",      "sim": .454, "wer": .073, "mos": 3.81, "mcd": 8.62, "rtf": 0.62, "sva": .908, "emo": .639},
    {"m": "SparkTTS",     "sim": .408, "wer": .326, "mos": 4.06, "mcd": 5.83, "rtf": 1.56, "sva": .764, "emo": .672},
    {"m": "OZSpeech",     "sim": .388, "wer": .060, "mos": 3.21, "mcd": 6.87, "rtf": 8.75, "sva": .840, "emo": .636},
    {"m": "OpenVoice V2", "sim": .244, "wer": .075, "mos": 4.30, "mcd": 7.06, "rtf": 0.08, "sva": .474, "emo": .601},
    {"m": "StyleTTS 2",   "sim": .228, "wer": .049, "mos": 4.30, "mcd": 6.81, "rtf": 0.11, "sva": .388, "emo": .589},
]

# Protection robustness — SIM, LibriTTS. ss=SafeSpeech ek=Enkidu sp=Spectral gr=GR-Noise em=EM
ROBUSTNESS = [
    {"m": "Qwen3-TTS",     "clean": .614, "ss": .384, "ek": .502, "sp": .363, "gr": .408, "em": .582},
    {"m": "IndexTTS",      "clean": .606, "ss": .346, "ek": .475, "sp": .318, "gr": .392, "em": .572},
    {"m": "CosyVoice 2",   "clean": .602, "ss": .321, "ek": .447, "sp": .301, "gr": .384, "em": .549},
    {"m": "ZipVoice",      "clean": .579, "ss": .287, "ek": .435, "sp": .262, "gr": .258, "em": .543},
    {"m": "MaskGCT",       "clean": .570, "ss": .303, "ek": .407, "sp": .281, "gr": .312, "em": .530},
    {"m": "GLM-TTS",       "clean": .570, "ss": .330, "ek": .445, "sp": .311, "gr": .388, "em": .532},
    {"m": "F5-TTS",        "clean": .559, "ss": .207, "ek": .431, "sp": .176, "gr": .137, "em": .520},
    {"m": "Higgs Audio",   "clean": .559, "ss": .264, "ek": .435, "sp": .236, "gr": .272, "em": .521},
    {"m": "MGM-Omni",      "clean": .539, "ss": .184, "ek": .316, "sp": .166, "gr": .229, "em": .491},
    {"m": "PlayDiffusion", "clean": .506, "ss": .173, "ek": None, "sp": .149, "gr": .162, "em": .466},
    {"m": "MOSS-TTSD",     "clean": .492, "ss": .242, "ek": .335, "sp": .216, "gr": .247, "em": .453},
    {"m": "VibeVoice",     "clean": .480, "ss": .272, "ek": .367, "sp": .253, "gr": .280, "em": .442},
    {"m": "FishSpeech",    "clean": .472, "ss": .238, "ek": .334, "sp": .212, "gr": .235, "em": .439},
    {"m": "XTTS-v2",       "clean": .454, "ss": .260, "ek": .308, "sp": .241, "gr": .237, "em": .414},
    {"m": "SparkTTS",      "clean": .408, "ss": .129, "ek": .137, "sp": .108, "gr": .062, "em": .359},
    {"m": "OZSpeech",      "clean": .388, "ss": .156, "ek": .187, "sp": .147, "gr": .148, "em": .337},
    {"m": "OpenVoice V2",  "clean": .244, "ss": .185, "ek": .188, "sp": .180, "gr": .175, "em": .236},
    {"m": "StyleTTS 2",    "clean": .228, "ss": .089, "ek": .125, "sp": .081, "gr": .030, "em": .207},
]
ROBUSTNESS_METHOD_NAME = {"ss": "SafeSpeech", "ek": "Enkidu", "sp": "Spectral", "gr": "GR-Noise", "em": "EM"}

# Cross-dataset generalisation — SIM, clean prompts.
CROSS_DATASET_COLUMNS = ["LibriTTS", "VCTK", "Multi-spk", "Long", "AISHELL", "French", "Bilingual", "BG-clean", "BG-noise", "Halluc."]
CROSS_DATASET = [
    {"m": "Qwen3-TTS",    "v": [.614, .618, .495, .561, .721, .536, .673, .689, .572, .515]},
    {"m": "IndexTTS",     "v": [.606, .567, .473, .775, .721, .397, .673, .589, .528, .529]},
    {"m": "CosyVoice 2",  "v": [.602, .582, .448, .530, .717, .378, .653, .626, .515, .518]},
    {"m": "ZipVoice",     "v": [.579, .554, .531, .729, .712, .363, .322, .625, .462, .509]},
    {"m": "MaskGCT",      "v": [.570, .555, .431, .194, .674, .494, None, .610, .487, .499]},
    {"m": "GLM-TTS",      "v": [.570, .573, .445, .757, .690, .398, .657, .622, .528, .533]},
    {"m": "F5-TTS",       "v": [.559, .537, .507, .607, .696, .304, .653, .582, .414, .455]},
    {"m": "Higgs Audio",  "v": [.559, .516, .418, .520, .581, .349, .543, .592, .421, .425]},
    {"m": "MGM-Omni",     "v": [.539, .447, .370, .442, .713, .227, .630, .523, .332, .396]},
    {"m": "PlayDiffusion","v": [.506, .426, .360, .637, .441, .283, .465, .433, .305, .408]},
    {"m": "MOSS-TTSD",    "v": [.492, .440, .379, .644, .437, .327, .471, .494, .488, .416]},
    {"m": "VibeVoice",    "v": [.480, .436, .348, .625, .564, .343, .531, .513, .364, .408]},
    {"m": "FishSpeech",   "v": [.472, .430, .383, .572, .611, .374, .566, .495, .387, .351]},
    {"m": "XTTS-v2",      "v": [.454, .454, .328, .613, .569, .445, .506, .546, .394, .488]},
    {"m": "SparkTTS",     "v": [.408, .532, .228, .345, .569, .164, .480, .588, .332, .336]},
    {"m": "OZSpeech",     "v": [.388, .253, .271, None, None, .109, None, .272, .164, .281]},
    {"m": "OpenVoice V2", "v": [.244, .392, .192, .278, .431, .271, .298, .484, .358, .365]},
    {"m": "StyleTTS 2",   "v": [.228, .236, .162, None, None, None, .213, .196, .166, .184]},
]

MODELS = [
    {"n": "BertVITS2", "key": "bert", "b": False},
    {"n": "Qwen3-TTS", "key": "qwen3_tts", "b": True},
    {"n": "Qwen3-Omni", "key": "qwen3_omni", "b": False},
    {"n": "FireRedTTS-2", "key": "fireredtts2", "b": False},
    {"n": "VoxCPM", "key": "voxcpm", "b": False},
    {"n": "F5-TTS", "key": "f5_tts", "b": True},
    {"n": "MaskGCT", "key": "maskgct", "b": True},
    {"n": "OpenVoice V2", "key": "openvoice", "b": True},
    {"n": "Coqui XTTS-v2", "key": "xtts", "b": True},
    {"n": "IndexTTS", "key": "index_tts", "b": True},
    {"n": "ZipVoice", "key": "zipvoice", "b": True},
    {"n": "FishSpeech", "key": "fishspeech", "b": True},
    {"n": "Fish Audio S2", "key": "fishspeech_s2", "b": False},
    {"n": "CosyVoice / 2", "key": "cosyvoice", "b": True},
    {"n": "Higgs Audio", "key": "higgs_audio", "b": True},
    {"n": "SparkTTS", "key": "sparktts", "b": True},
    {"n": "VALL-E", "key": "vall_e", "b": False},
    {"n": "StyleTTS 2", "key": "styletts2", "b": True},
    {"n": "GLM-TTS", "key": "glm_tts", "b": True},
    {"n": "GlowTTS", "key": "glowtts", "b": False},
    {"n": "Kimi Audio", "key": "kimi_audio", "b": False},
    {"n": "MGM-Omni", "key": "mgm_omni", "b": True},
    {"n": "MOSS TTSD", "key": "moss_ttsd", "b": True},
    {"n": "PlayDiffusion", "key": "playdiffusion", "b": True},
    {"n": "Bark Voice Clone", "key": "bark_voice_clone", "b": False},
    {"n": "OZSpeech", "key": "ozspeech", "b": True},
    {"n": "VibeVoice", "key": "vibevoice", "b": True},
]

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
             "It evaluates 27 zero-shot and fine-tuning TTS/VC models against 5 audio-protection methods across "
             "10 dataset configurations, scoring speaker similarity, intelligibility, perceptual quality, and runtime.",
    },
    {
        "q": "How many voice-cloning models does RVCBench evaluate?",
        "a": "The RVCBench codebase includes wrappers for 27 TTS/VC adversary models. The arXiv v2 paper reports "
             "results for 18 of those models across 18 robustness evaluations, 225 speakers, and 14,370 utterances.",
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
             "and drop furthest under protection — GR-Noise pushes StyleTTS 2's similarity from 0.228 down to 0.030.",
    },
    {
        "q": "Is the RVCBench dataset public?",
        "a": "Yes. The benchmark dataset is hosted on Hugging Face at huggingface.co/datasets/Nanboy/RVCBench under "
             "a CC0-1.0 license, with 10 dataset configurations spanning English, Mandarin, and French.",
    },
    {
        "q": "How do I cite RVCBench?",
        "a": "Cite the arXiv preprint: Jin, Ruinan; Liao, Xinting; Yu, Hanlin; Pandya, Deval; Li, Xiaoxiao. "
             "“RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models.” "
             "arXiv:2602.00443, 2026.",
    },
]

CITATION = {
    "title": "RVCBench: Benchmarking the Robustness of Voice Cloning Across Modern Audio Generation Models",
    "authors": ["Ruinan Jin", "Xinting Liao", "Hanlin Yu", "Deval Pandya", "Xiaoxiao Li"],
    "year": "2026",
    "arxiv_id": "2602.00443",
}
