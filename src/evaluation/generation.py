import csv
import hashlib
import inspect
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from matplotlib import backends

import jiwer
import torch
import torchaudio
import whisper
from pymcd.mcd import Calculate_MCD
from hydra.utils import to_absolute_path
# Lazy import inside helper to reduce optional deps
from tqdm import tqdm

try:
    from huggingface_hub.errors import RemoteEntryNotFoundError
except ImportError:  # pragma: no cover - fallback when HF Hub unavailable
    RemoteEntryNotFoundError = RuntimeError

from . import bootstrap as bootstrap_utils
from .fidelity import (
    _load_dnsmos_predictor,
    _load_speechmos_predictor,
    _predict_dnsmos_scores,
    _predict_speechmos_mos,
)

_SANITIZE_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
_EMOTION_MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
_EMOTION_SR = 16000


def _sanitize_component(value: str) -> str:
    return _SANITIZE_PATTERN.sub("_", str(value))


def _normalize_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        try:
            return torch.device(device)
        except Exception:
            return torch.device("cpu")
    if isinstance(device, int):
        return torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
    if device is not None:
        try:
            return torch.device(device)
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def _load_synthesis_timings(generated_audio_dir: Path) -> dict[str, float]:
    timing_path = Path(generated_audio_dir) / "synthesis_timings.csv"
    if not timing_path.is_file():
        return {}
    timings: dict[str, float] = {}
    try:
        with open(timing_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raw_path = row.get("generated_path")
                raw_time = row.get("synthesis_time_sec")
                if not raw_path or raw_time in (None, ""):
                    continue
                try:
                    timings[str(Path(raw_path).resolve())] = float(raw_time)
                except Exception:
                    continue
    except Exception:
        return {}
    return timings


def _format_duration_tag(seconds: float) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    return f"{value:g}".replace(".", "p")


def _maybe_cap_generated_audio(
    gen_file: Path,
    gen_dir: Path,
    max_duration_sec: Optional[float],
    logger,
    cache: dict,
) -> Path:
    if not max_duration_sec:
        return gen_file

    try:
        max_duration_sec = float(max_duration_sec)
    except (TypeError, ValueError):
        logger.warning("Invalid generated_audio_max_seconds=%r; skipping cap.", max_duration_sec)
        return gen_file

    if max_duration_sec <= 0:
        return gen_file

    cache_key = (str(gen_file), max_duration_sec)
    if cache_key in cache:
        return cache[cache_key]

    try:
        info = torchaudio.info(str(gen_file))
    except Exception as exc:
        logger.warning("Failed to inspect %s for duration cap: %s", gen_file, exc)
        cache[cache_key] = gen_file
        return gen_file

    sample_rate = info.sample_rate or 0
    num_frames = info.num_frames or 0
    if sample_rate <= 0 or num_frames <= 0:
        cache[cache_key] = gen_file
        return gen_file

    max_frames = int(max_duration_sec * sample_rate)
    if max_frames <= 0 or num_frames <= max_frames:
        cache[cache_key] = gen_file
        return gen_file

    trimmed_dir = gen_dir / "_eval_trimmed"
    trimmed_dir.mkdir(parents=True, exist_ok=True)
    duration_tag = _format_duration_tag(max_duration_sec)
    hash_tag = hashlib.md5(str(gen_file).encode("utf-8")).hexdigest()[:8]
    trimmed_path = trimmed_dir / f"{gen_file.stem}_cap{duration_tag}_{hash_tag}{gen_file.suffix}"

    if trimmed_path.exists():
        cache[cache_key] = trimmed_path
        return trimmed_path

    try:
        waveform, sr = torchaudio.load(str(gen_file), frame_offset=0, num_frames=max_frames)
    except Exception as exc:
        logger.warning("Failed to load %s for duration cap: %s", gen_file, exc)
        cache[cache_key] = gen_file
        return gen_file

    if waveform.numel() == 0:
        logger.warning("Loaded empty waveform while capping %s; using original file.", gen_file)
        cache[cache_key] = gen_file
        return gen_file

    try:
        torchaudio.save(str(trimmed_path), waveform.cpu(), sample_rate=sr)
    except Exception as exc:
        logger.warning("Failed to save capped audio for %s: %s", gen_file, exc)
        cache[cache_key] = gen_file
        return gen_file

    logger.info(
        "Capped generated audio %s to %.2fs for evaluation.",
        gen_file,
        max_duration_sec,
    )
    cache[cache_key] = trimmed_path
    return trimmed_path


class _EmotionFeatureAdapter(torch.nn.Module):
    def __init__(self, wav2vec_module):
        super().__init__()
        self.wav2vec_module = wav2vec_module

    def forward(self, wavs):
        if wavs.dim() == 1:
            wavs = wavs.unsqueeze(0)
        batch_size = wavs.shape[0]
        device = wavs.device
        wav_lens = torch.ones(batch_size, device=device)
        return self.wav2vec_module(wavs, wav_lens)


class _EmotionPassThrough(torch.nn.Module):
    def forward(self, tensor, *_args, **_kwargs):  # pragma: no cover - trivial
        return tensor


class _EmotionEmbeddingAdapter(torch.nn.Module):
    def __init__(self, pooling_module):
        super().__init__()
        self.pooling_module = pooling_module

    def forward(self, feats, wav_lens=None):
        pooled = self.pooling_module(feats, wav_lens)
        if pooled.dim() == 3:
            pooled = pooled.squeeze(1)
        return pooled


class _EmotionClassifierHead(torch.nn.Module):
    def __init__(self, head_module):
        super().__init__()
        self.head = head_module

    def forward(self, embeddings):  # pragma: no cover - thin wrapper
        return self.head(embeddings)


# -----------------------------
# Mandarin-friendly text utils
# -----------------------------
_ZH_PUNCT_MAP = {
    "，": ", ",
    "。": ".",
    "：": ":",
    "；": ";",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "、": ",",
    "—": "-",
    "…": "...",
    "·": ".",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
}


def _is_cjk_char(ch: str) -> bool:
    try:
        return "\u4e00" <= ch <= "\u9fff"
    except Exception:
        return False


def _to_simplified_zh(text: str) -> str:
    """Convert Chinese text to Simplified Chinese when possible.
    Tries OpenCC first, then hanziconv, otherwise returns the input unchanged.
    """
    s = str(text or "")
    try:
        # opencc-python-reimplemented
        from opencc import OpenCC  # type: ignore

        try:
            cc = OpenCC("t2s")
        except Exception:
            # Fallback conversion map commonly available
            cc = OpenCC("hk2s")
        return cc.convert(s)
    except Exception:
        pass

    try:
        # hanziconv
        from hanziconv import HanziConv  # type: ignore

        return HanziConv.toSimplified(s)
    except Exception:
        return s


def _contains_cjk(text: str) -> bool:
    try:
        return any(_is_cjk_char(ch) for ch in str(text or ""))
    except Exception:
        return False


def _normalize_zh_text(text: str) -> str:
    s = str(text or "")
    # Always normalize to Simplified to reduce script variance
    try:
        s = _to_simplified_zh(s)
    except Exception:
        pass
    # Normalize Chinese punctuation to English equivalents
    for zh, en in _ZH_PUNCT_MAP.items():
        s = s.replace(zh, en)
    # Remove ALL punctuation for WER calculation to avoid spurious errors
    # Keep only letters, numbers, Chinese characters, and spaces
    s = re.sub(r'[^\w\s\u4e00-\u9fff]', '', s)
    # Collapse whitespace, remove surrounding spaces
    s = " ".join(s.split())
    # Remove spaces between CJK to get stable char sequence
    # e.g., "摩 洛" -> "摩洛"
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    return s.strip()


def _char_edit_distance(ref: str, hyp: str) -> int:
    # Standard Levenshtein distance at character level
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = temp
    return dp[m]


def _cer(ref: str, hyp: str) -> Optional[float]:
    if ref is None:
        return None
    ref_s = str(ref)
    hyp_s = str(hyp or "")
    if len(ref_s) == 0:
        return None
    dist = _char_edit_distance(ref_s, hyp_s)
    return dist / len(ref_s)


def _ensure_emotion_classifier_ready(classifier, logger) -> bool:
    mods = getattr(classifier, "mods", None)
    if mods is None:
        return False

    required = {"compute_features", "mean_var_norm", "embedding_model", "classifier"}
    if required.issubset(set(mods.keys())):
        return True

    def _try_get(name):
        try:
            return mods[name]
        except KeyError:
            return None

    wav2vec_module = _try_get("wav2vec2") or _try_get("wav2vec")
    pool_module = _try_get("avg_pool")
    head_module = _try_get("output_mlp")

    if wav2vec_module is None or pool_module is None or head_module is None:
        if logger is not None:
            logger.warning(
                "Emotion recognizer missing required SpeechBrain modules; disabling emotion metrics.",
            )
        return False

    if "compute_features" not in mods:
        mods["compute_features"] = _EmotionFeatureAdapter(wav2vec_module)
    if "mean_var_norm" not in mods:
        mods["mean_var_norm"] = _EmotionPassThrough()
    if "embedding_model" not in mods:
        mods["embedding_model"] = _EmotionEmbeddingAdapter(pool_module)
    if "classifier" not in mods:
        mods["classifier"] = _EmotionClassifierHead(head_module)

    return True


def _load_emotion_recognizer(device, logger):
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("SpeechBrain unavailable; skipping emotion metrics (%s).", exc)
        return None

    savedir = Path(to_absolute_path("checkpoints/emotion-recognition-wav2vec2-IEMOCAP"))
    savedir.mkdir(parents=True, exist_ok=True)
    resolved_device = _normalize_device(device)

    run_opts = {"device": str(resolved_device)}

    try:
        classifier = EncoderClassifier.from_hparams(
            source=_EMOTION_MODEL_ID,
            savedir=str(savedir),
            run_opts=run_opts,
        )
    except RemoteEntryNotFoundError as err:
        if logger is not None:
            logger.warning(
                "SpeechBrain emotion recognizer missing custom.py (%s); applying local stub.",
                err,
            )
        try:
            _ensure_speechbrain_artifacts(
                _EMOTION_MODEL_ID,
                savedir,
                (
                    "hyperparams.yaml",
                    "wav2vec2.ckpt",
                    "model.ckpt",
                    "label_encoder.txt",
                ),
                logger,
            )
        except Exception:
            return None

        _write_speechbrain_stub(
            savedir,
            logger,
            base_import="from speechbrain.inference.classifiers import EncoderClassifier",
            class_name="EncoderClassifier",
        )

        try:
            classifier = EncoderClassifier.from_hparams(
                source=str(savedir),
                savedir=str(savedir),
                run_opts=run_opts,
                pymodule_file="custom.py",
            )
        except Exception as inner_exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Failed to load emotion recognizer from local cache even after stub: %s",
                inner_exc,
            )
            return None
    except Exception as exc:  # pragma: no cover - download/runtime failure
        logger.warning("Failed to load emotion recognizer %s: %s", _EMOTION_MODEL_ID, exc)
        return None

    if not _ensure_emotion_classifier_ready(classifier, logger):
        return None

    try:
        classifier.to(resolved_device)
    except Exception:
        logger.debug("Unable to move emotion recognizer to %s; using default device", resolved_device)
    classifier._awm_device = resolved_device
    return classifier


def _predict_emotion_label(emotion_model, audio_path: Path, logger):
    if emotion_model is None:
        return None
    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception as exc:
        logger.warning("Failed to load %s for emotion prediction: %s", audio_path, exc)
        return None

    if waveform.numel() == 0:
        logger.warning("Loaded empty waveform for %s; skipping emotion prediction.", audio_path)
        return None

    if sr != _EMOTION_SR:
        waveform = torchaudio.functional.resample(waveform, sr, _EMOTION_SR)

    waveform = waveform.mean(dim=0, keepdim=True)
    emotion_device = getattr(emotion_model, "_awm_device", torch.device("cpu"))
    waveform = waveform.to(emotion_device)

    try:
        _, _, _, labels = emotion_model.classify_batch(waveform)
    except Exception as exc:
        logger.warning("Emotion recognizer failed on %s: %s", audio_path, exc)
        return None

    if not labels:
        return None

    label = labels[0]
    if isinstance(label, bytes):
        label = label.decode("utf-8", errors="ignore")
    return str(label)


def _resolve_generated_file(
    gt_path: Path,
    idx: int,
    metadata,
    available_files: dict,
    logger,
):
    if isinstance(metadata, dict):
        speaker_token = _sanitize_component(metadata.get("speaker_id", "0"))
    else:
        speaker_token = _sanitize_component("0")

    base_stub = _sanitize_component(gt_path.stem)
    speaker_id = _sanitize_component(gt_path.parent.name)

    candidate_names = [
        f"{base_stub}_cloned.wav",
        f"{speaker_token}_{speaker_id}_{idx}.wav",  # BERT-VITS2 pattern
        f"{speaker_token}_{base_stub}_{idx}.wav",  # GlowTTS / StyleTTS2 / Higgs
        f"{base_stub}_{speaker_token}_{idx}.wav",  # OZSpeech pattern
        f"{base_stub}_{idx}.wav",                  # Fallback: base with index
        f"{speaker_token}_{idx}.wav",               # Fallback: token with index
    ]

    for name in candidate_names:
        gen_path = available_files.pop(name, None)
        if gen_path is not None:
            return gen_path

    # Some generators append extra descriptors (e.g., language tags) to the
    # filename. Fall back to prefix matching before giving up so we still
    # evaluate if the core stub matches.
    prefixes = [name[:-4] for name in candidate_names if name.endswith(".wav")]
    for candidate_prefix in prefixes:
        for key in list(available_files.keys()):
            if key.startswith(candidate_prefix):
                return available_files.pop(key)

    logger.warning(
        "Could not locate generated audio for %s (tried %s)",
        gt_path,
        ", ".join(candidate_names),
    )
    return None


def _coerce_sva_decision(decision):
    if isinstance(decision, torch.Tensor):
        if decision.numel() == 1:
            try:
                decision = decision.item()
            except Exception:
                return None
        else:
            return None

    if isinstance(decision, bool):
        return decision

    if isinstance(decision, (int, float)):
        return bool(decision)

    try:
        token = str(decision).strip().lower()
    except Exception:
        return None

    if token in {"y", "yes", "true", "same", "positive", "match", "1"}:
        return True
    if token in {"n", "no", "false", "different", "negative", "mismatch", "0"}:
        return False
    return None


def _load_evaluation_components(device, logger):
    _ensure_hf_token_compat(logger)
    logger.info("Loading evaluation model (Whisper-multilingual, ECAPA-TDNN)...")
    # Use multilingual Whisper to support non-English transcripts (e.g., ZH)
    whisper_model = whisper.load_model("medium", device=device)
    sim_model = _load_speaker_recognition(device, logger)
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")

    speechmos_model = _load_speechmos_predictor(logger, device)
    dnsmos_model = _load_dnsmos_predictor(logger)
    emotion_pipeline = _load_emotion_recognizer(device, logger)

    return (
        whisper_model,
        sim_model,
        mcd_toolbox,
        speechmos_model,
        dnsmos_model,
        emotion_pipeline,
    )


def _evaluate_pairs(
    pairs: Sequence[Tuple[Path, Path, Union[dict, str]]],
    generated_audio_dir,
    device,
    logger,
    target_sr: int,
    synthesis_time_sec: Optional[float] = None,
    bootstrap_config=None,
    max_generated_audio_seconds: Optional[float] = None,
):
    if not pairs:
        raise RuntimeError("No evaluation pairs provided.")
        # return {"error": "No generated files found for evaluation."}

    (
        whisper_model,
        sim_model,
        mcd_toolbox,
        speechmos_model,
        dnsmos_model,
        emotion_pipeline,
    ) = _load_evaluation_components(device, logger)
    bootstrapper = bootstrap_utils.create_bootstrapper(bootstrap_config)

    gen_dir = Path(generated_audio_dir)
    synthesis_timings = _load_synthesis_timings(gen_dir)
    capped_audio_cache: dict[tuple[str, float], Path] = {}
    # Aggregate sums and independent counters so a failure in one metric
    # does not exclude the sample entirely.
    total_mcd = 0.0
    total_mcd_samples = 0
    total_wer = 0.0
    total_wer_samples = 0
    total_sim = 0.0
    total_sim_samples = 0
    total_sva_positive = 0
    total_sva_samples = 0
    total_emotion_matches = 0
    total_emotion_samples = 0
    total_audio_duration = 0.0
    speechmos_scores = []
    dnsmos_scores = {"ovrl": [], "sig": [], "bak": []}
    mcd_values = []
    wer_values = []
    sim_values = []
    sva_values = []
    emotion_values = []
    sample_metrics = []
    file_count = 0  # Number of processed (generated file existed) pairs

    logger.info("Calculating Generation Metrics (MCD, WER, SIM, SpeechMOS, DNSMOS, Emotion)...")
    for i, (gt_path, gen_file, metadata) in enumerate(tqdm(pairs)):
        gt_path = Path(gt_path)
        gen_file = Path(gen_file)
        if isinstance(metadata, dict):
            gt_text = metadata.get("text")
            speaker_value = metadata.get("speaker_id")
            lang_hint = metadata.get("language")
        else:
            gt_text = metadata
            speaker_value = None
            lang_hint = None

        if not gen_file.exists():
            logger.warning("Generated file %s not found; skipping.", gen_file)
            continue

        gen_file_for_metrics = _maybe_cap_generated_audio(
            gen_file,
            gen_dir,
            max_generated_audio_seconds,
            logger,
            capped_audio_cache,
        )

        # 1. MCD (robust – do not abort other metrics if it fails)
        mcd_value = None
        try:
            mcd_value = float(mcd_toolbox.calculate_mcd(str(gt_path), str(gen_file_for_metrics)))
        except Exception as exc:
            logger.warning("Failed to compute MCD for %s vs %s: %s", gt_path, gen_file_for_metrics, exc)
        else:
            total_mcd += mcd_value
            mcd_values.append(mcd_value)
            total_mcd_samples += 1

        # 2. WER (robust) – compute WER for all languages.
        transcription = None
        wer = None
        if gt_text is None:
            gt_text = ""
        try:
            # Encourage strict transcription (no translation, no context carry-over)
            transcribe_kwargs = {
                "task": "transcribe",
                "condition_on_previous_text": False,
                "without_timestamps": True,
            }
            # If language is explicitly provided in metadata and not set to "auto",
            # let it guide Whisper; otherwise, allow Whisper to auto-detect.
            if isinstance(lang_hint, str):
                lang_hint_norm = lang_hint.strip().lower()
                if lang_hint_norm and lang_hint_norm not in {"auto", "none", "null"}:
                    transcribe_kwargs["language"] = lang_hint_norm

            result = whisper_model.transcribe(str(gen_file_for_metrics), **transcribe_kwargs)
            transcription = result["text"]
            # Convert traditional Chinese to simplified Chinese immediately after transcription
            if _contains_cjk(transcription):
                transcription = _to_simplified_zh(transcription)
            detected_lang = result.get("language")
        except Exception as exc:
            logger.warning("Whisper failed for %s: %s", gen_file_for_metrics, exc)
        else:
            ref_text_proc = str(gt_text)
            hyp_text_proc = str(transcription)
            ref_is_chinese = _contains_cjk(ref_text_proc) or (detected_lang == "zh")
            
            # For Chinese (or texts containing CJK), segment into words using jieba then compute WER.
            if ref_is_chinese:
                try:
                    import jieba
                    ref_norm = _normalize_zh_text(ref_text_proc)
                    hyp_norm = _normalize_zh_text(hyp_text_proc)
                    ref_tokens = " ".join(jieba.lcut(ref_norm))
                    hyp_tokens = " ".join(jieba.lcut(hyp_norm))
                    wer = float(jiwer.wer(ref_tokens, hyp_tokens))
                except Exception as exc:
                    logger.warning("Failed to compute Chinese WER for %s: %s", gen_file, exc)
            else:
                try:
                    # Remove punctuation before WER calculation to avoid inflated error rates
                    import string
                    ref_no_punct = ref_text_proc.translate(str.maketrans('', '', string.punctuation)).lower()
                    hyp_no_punct = hyp_text_proc.translate(str.maketrans('', '', string.punctuation)).lower()
                    wer = float(jiwer.wer(ref_no_punct, hyp_no_punct))
                except Exception as exc:
                    logger.warning("Failed to compute WER for %s: %s", gen_file, exc)
        if wer is not None:
            total_wer += wer
            wer_values.append(wer)
            total_wer_samples += 1

        # 3. SIM & SVA (robust)
        sim_score = None
        sva_decision = None
        try:
            score, decision = sim_model.verify_files(str(gt_path), str(gen_file_for_metrics))
            sim_score = float(score)
            sva_decision = _coerce_sva_decision(decision)
        except Exception as exc:
            logger.warning("Speaker similarity failed for %s vs %s: %s", gt_path, gen_file_for_metrics, exc)
        if sim_score is not None:
            total_sim += sim_score
            sim_values.append(sim_score)
            total_sim_samples += 1

        if sva_decision is not None:
            total_sva_samples += 1
            if sva_decision:
                total_sva_positive += 1
            sva_values.append(1.0 if sva_decision else 0.0)
        else:
            logger.debug(
                "Speaker verification decision for %s vs %s is unsupported type %r; excluding from SVA.",
                gt_path,
                gen_file,
                sva_decision,
            )

        speechmos_score = None
        if speechmos_model is not None:
            try:
                gen_wav, gen_sr = torchaudio.load(str(gen_file_for_metrics))
            except Exception as exc:
                logger.warning(f"Failed to load {gen_file_for_metrics} for SpeechMOS calculation: {exc}")
            else:
                if gen_wav.numel() == 0:
                    logger.warning(f"Loaded empty audio tensor for {gen_file_for_metrics}; skipping SpeechMOS.")
                else:
                    if gen_sr != target_sr:
                        gen_wav = torchaudio.functional.resample(gen_wav, gen_sr, target_sr)
                    gen_wav = gen_wav.mean(dim=0)
                    speechmos_score = _predict_speechmos_mos(
                        speechmos_model,
                        gen_file,
                        gen_wav,
                        target_sr,
                        logger,
                    )
                    if speechmos_score is not None:
                        speechmos_score = float(speechmos_score)
                        speechmos_scores.append(speechmos_score)

        dnsmos_values = _predict_dnsmos_scores(dnsmos_model, gen_file_for_metrics, logger)
        if dnsmos_values is not None:
            for key in ("ovrl", "sig", "bak"):
                dnsmos_scores[key].append(dnsmos_values[key])

        duration_sec = None
        metadata_info = None
        backends = []
        try:
            from torchaudio.backend import sox_io_backend
            backends.append(("sox", sox_io_backend.info))
        except Exception:
            pass

        try:
            from torchaudio.backend import soundfile_backend
            backends.append(("soundfile", soundfile_backend.info))
        except Exception:
            pass
        
        try:
            backends.append(("torchaudio", torchaudio.info))
        except Exception:
            pass



        for name, fn in backends:
            try:
                metadata_info = fn(str(gen_file))
                break
            except Exception as exc:
                logger.warning(
                    "Failed to read audio metadata via %s backend for %s: %s",
                    name,
                    gen_file,
                    exc,
                )

        if metadata_info is not None:
            sample_rate = float(metadata_info.sample_rate or 0)
            if sample_rate > 0 and metadata_info.num_frames:
                duration_sec = metadata_info.num_frames / sample_rate
                total_audio_duration += duration_sec

        try:
            resolved_gen = str(gen_file.resolve())
        except Exception:
            resolved_gen = str(gen_file)
        synthesis_time = synthesis_timings.get(resolved_gen)

        ref_emotion = None
        gen_emotion = None
        emotion_match = None
        if emotion_pipeline is not None:
            ref_emotion = _predict_emotion_label(emotion_pipeline, gt_path, logger)
            gen_emotion = _predict_emotion_label(emotion_pipeline, gen_file_for_metrics, logger)
            if ref_emotion is not None and gen_emotion is not None:
                emotion_match = ref_emotion == gen_emotion
                total_emotion_samples += 1
                if emotion_match:
                    total_emotion_matches += 1
                emotion_values.append(1.0 if emotion_match else 0.0)

        # Convert traditional Chinese to simplified for display consistency
        display_transcription = transcription
        if transcription and _contains_cjk(transcription):
            try:
                display_transcription = _to_simplified_zh(transcription)
            except Exception:
                pass  # Keep original if conversion fails
        
        sample_metrics.append(
            {
                "ground_truth_path": str(gt_path),
                "generated_path": str(gen_file),
                "speaker_id": speaker_value,
                "ground_truth_text": gt_text,
                "predicted_text": display_transcription,
                "mcd": mcd_value,
                "wer": wer,
                "sim": sim_score,
                "speechmos_mos": speechmos_score,
                "dnsmos_ovrl": dnsmos_values["ovrl"] if dnsmos_values else None,
                "dnsmos_sig": dnsmos_values["sig"] if dnsmos_values else None,
                "dnsmos_bak": dnsmos_values["bak"] if dnsmos_values else None,
                "sva": sva_decision,
                "generated_duration_sec": duration_sec,
                "synthesis_time_sec": synthesis_time,
                "reference_emotion": ref_emotion,
                "generated_emotion": gen_emotion,
                "emotion_match": emotion_match,
            }
        )
        # Count this pair as processed for aggregate stats
        file_count += 1

    if file_count == 0:
        logger.error("No evaluation pairs yielded any successful metric; returning empty metrics set.")
        return {
            "evaluated_pairs": 0,
            "avg_mcd": None,
            "avg_wer": None,
            "avg_sim": None,
            "mcd_pairs": 0,
            "wer_pairs": 0,
            "sim_pairs": 0,
        }

    result = {
        "avg_mcd": (total_mcd / total_mcd_samples) if total_mcd_samples else None,
        "avg_wer": (total_wer / total_wer_samples) if total_wer_samples else None,
        "avg_sim": (total_sim / total_sim_samples) if total_sim_samples else None,
        "evaluated_pairs": file_count,
        "mcd_pairs": total_mcd_samples,
        "wer_pairs": total_wer_samples,
        "sim_pairs": total_sim_samples,
    }

    if total_audio_duration > 0:
        result["total_audio_duration_sec"] = total_audio_duration
        if synthesis_time_sec is not None:
            result["rtf"] = synthesis_time_sec / total_audio_duration
            result["synthesis_time_sec"] = synthesis_time_sec
        else:
            result["rtf"] = None
            result["synthesis_time_sec"] = None
    else:
        result["total_audio_duration_sec"] = 0.0
        result["rtf"] = None
        result["synthesis_time_sec"] = synthesis_time_sec

    if total_sva_samples > 0:
        result["avg_sva"] = total_sva_positive / total_sva_samples
        result["sva_pairs"] = total_sva_samples
    else:
        result["avg_sva"] = None
        result["sva_pairs"] = 0

    if speechmos_scores:
        result["avg_speechmos_mos"] = sum(speechmos_scores) / len(speechmos_scores)
        result["speechmos_pairs"] = len(speechmos_scores)
        result["min_speechmos_mos"] = min(speechmos_scores)
        result["max_speechmos_mos"] = max(speechmos_scores)
    else:
        result["avg_speechmos_mos"] = None
        result["speechmos_pairs"] = 0
        result["min_speechmos_mos"] = None
        result["max_speechmos_mos"] = None

    dnsmos_pairs = len(dnsmos_scores["ovrl"])
    if dnsmos_pairs:
        result["avg_dnsmos_ovrl"] = sum(dnsmos_scores["ovrl"]) / dnsmos_pairs
        result["avg_dnsmos_sig"] = sum(dnsmos_scores["sig"]) / dnsmos_pairs
        result["avg_dnsmos_bak"] = sum(dnsmos_scores["bak"]) / dnsmos_pairs
        result["dnsmos_pairs"] = dnsmos_pairs
        result["min_dnsmos_ovrl"] = min(dnsmos_scores["ovrl"])
        result["max_dnsmos_ovrl"] = max(dnsmos_scores["ovrl"])
    else:
        result["avg_dnsmos_ovrl"] = None
        result["avg_dnsmos_sig"] = None
        result["avg_dnsmos_bak"] = None
        result["dnsmos_pairs"] = 0
        result["min_dnsmos_ovrl"] = None
        result["max_dnsmos_ovrl"] = None

    if total_emotion_samples > 0:
        result["emotion_match_rate"] = total_emotion_matches / total_emotion_samples
        result["emotion_pairs"] = total_emotion_samples
    else:
        result["emotion_match_rate"] = None
        result["emotion_pairs"] = 0

    if sample_metrics:
        sample_metrics_path = gen_dir.parent / "generation_sample_metrics.csv"
        sample_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "ground_truth_path",
            "generated_path",
            "speaker_id",
            "ground_truth_text",
            "predicted_text",
            "mcd",
            "wer",
            "sim",
            "speechmos_mos",
            "dnsmos_ovrl",
            "dnsmos_sig",
            "dnsmos_bak",
            "sva",
            "generated_duration_sec",
            "synthesis_time_sec",
            "reference_emotion",
            "generated_emotion",
            "emotion_match",
        ]
        with open(sample_metrics_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_metrics)
        logger.info("Saved sample-wise generation metrics to %s", sample_metrics_path)
        result["sample_metrics_csv"] = str(sample_metrics_path)
    else:
        result["sample_metrics_csv"] = None

    if bootstrapper is not None:
        bootstrapper.maybe_add_interval(result, "avg_mcd", mcd_values)
        bootstrapper.maybe_add_interval(result, "avg_wer", wer_values)
        bootstrapper.maybe_add_interval(result, "avg_sim", sim_values)
        bootstrapper.maybe_add_interval(result, "avg_sva", sva_values)
        bootstrapper.maybe_add_interval(result, "avg_speechmos_mos", speechmos_scores)
        for key in ("ovrl", "sig", "bak"):
            bootstrapper.maybe_add_interval(result, f"avg_dnsmos_{key}", dnsmos_scores[key])
        bootstrapper.maybe_add_interval(result, "emotion_match_rate", emotion_values)

    return result


def evaluate_pairs(
    pairs: Iterable[Tuple[Path, Path, Union[dict, str]]],
    generated_audio_dir,
    device,
    logger,
    target_sr: int = 24000,
    synthesis_time_sec: Optional[float] = None,
    bootstrap_config=None,
    max_generated_audio_seconds: Optional[float] = None,
):
    """Evaluate generation metrics for explicit (ground truth, generated) pairs."""
    pairs_list: List[Tuple[Path, Path, Union[dict, str]]] = list(pairs)
    return _evaluate_pairs(
        pairs_list,
        generated_audio_dir,
        device,
        logger,
        target_sr,
        synthesis_time_sec,
        bootstrap_config,
        max_generated_audio_seconds,
    )


def evaluate(
    gt_files_map,
    generated_audio_dir,
    device,
    logger,
    target_sr: int = 24000,
    synthesis_time_sec: Optional[float] = None,
    bootstrap_config=None,
    max_generated_audio_seconds: Optional[float] = None,
):
    """
    Evaluates the quality of the synthesized audio against ground truth
    by discovering generated files inside ``generated_audio_dir``.
    """
    gen_dir = Path(generated_audio_dir)
    available_files = {path.name: path for path in gen_dir.rglob("*.wav")}
    gt_paths = list(gt_files_map.keys())

    pairs: List[Tuple[Path, Path, Union[dict, str]]] = []
    for i, gt_path_str in enumerate(gt_paths):
        gt_path = Path(gt_path_str)
        metadata = gt_files_map[gt_path_str]
        gen_file = _resolve_generated_file(gt_path, i, metadata, available_files, logger)
        if gen_file is None:
            continue
        pairs.append((gt_path, gen_file, metadata))

    return _evaluate_pairs(
        pairs,
        generated_audio_dir,
        device,
        logger,
        target_sr,
        synthesis_time_sec,
        bootstrap_config,
        max_generated_audio_seconds,
    )
def _ensure_hf_token_compat(logger=None):
    try:
        import huggingface_hub
    except ImportError:  # pragma: no cover - optional dependency
        return

    if getattr(_ensure_hf_token_compat, "_patched", False):
        return

    hf_download = getattr(huggingface_hub, "hf_hub_download", None)
    if hf_download is None:
        return

    signature = inspect.signature(hf_download)
    if "use_auth_token" in signature.parameters:
        return

    def _compat(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None:
            if isinstance(use_auth_token, str):
                kwargs.setdefault("token", use_auth_token)
            elif use_auth_token is True:
                kwargs.setdefault("token", True)
            elif use_auth_token is False:
                kwargs.setdefault("token", None)
        return hf_download(*args, **kwargs)

    setattr(huggingface_hub, "hf_hub_download", _compat)
    _ensure_hf_token_compat._patched = True
    if logger is not None:
        logger.info("Patched huggingface_hub.hf_hub_download for use_auth_token compatibility.")


def _write_speechbrain_stub(
    savedir: Path,
    logger,
    *,
    base_import: str,
    class_name: str,
) -> Path:
    """Create a fallback SpeechBrain custom.py that re-exports the base class."""

    savedir.mkdir(parents=True, exist_ok=True)
    stub_path = savedir / "custom.py"
    if not stub_path.exists():
        stub_code = f"{base_import} as _SBBase\n\n\nclass {class_name}(_SBBase):\n    pass\n"
        stub_path.write_text(stub_code, encoding="utf-8")
        if logger is not None:
            logger.info("Created SpeechBrain custom.py stub at %s", stub_path)
    return stub_path


def _ensure_speechbrain_artifacts(
    model_id: str,
    savedir: Path,
    filenames: Sequence[str],
    logger,
) -> None:
    """Ensure required SpeechBrain files are cached locally for offline use."""

    try:
        from speechbrain.utils.fetching import fetch
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("SpeechBrain fetching utilities unavailable.") from exc

    savedir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        try:
            fetch(
                filename=filename,
                source=model_id,
                savedir=str(savedir),
            )
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Failed to cache %s from %s for SpeechBrain models: %s",
                    filename,
                    model_id,
                    exc,
                )
            raise


def _load_speaker_recognition(device, logger):
    from speechbrain.inference.speaker import SpeakerRecognition

    savedir = Path(to_absolute_path("checkpoints/spkrec-ecapa-voxceleb"))
    savedir.mkdir(parents=True, exist_ok=True)

    try:
        return SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": device},
        )
    except RemoteEntryNotFoundError as err:
        if logger is not None:
            logger.warning(
                "SpeechBrain repo missing custom.py (%s); falling back to local stub.",
                err,
            )
        _write_speechbrain_stub(
            savedir,
            logger,
            base_import="from speechbrain.inference.speaker import SpeakerRecognition",
            class_name="SpeakerRecognition",
        )
        try:
            return SpeakerRecognition.from_hparams(
                source=str(savedir),
                savedir=str(savedir),
                run_opts={"device": device},
                pymodule_file="custom.py",
            )
        except Exception as inner_err:  # pragma: no cover - defensive fallback
            raise RuntimeError(
                "Failed to load SpeechBrain speaker verification model even after applying stub. "
                "Ensure checkpoints are cached under 'checkpoints/spkrec-ecapa-voxceleb'."
            ) from inner_err
