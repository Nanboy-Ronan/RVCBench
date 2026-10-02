import csv
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

import jiwer
import numpy as np
import torch
import torchaudio
import whisper
from hydra.utils import to_absolute_path
from speechbrain.inference.speaker import SpeakerRecognition
from torch_stoi import NegSTOILoss
from tqdm import tqdm
import torch.nn.functional as F

from . import bootstrap as bootstrap_utils
from pymcd.mcd import Calculate_MCD

# UTMOS is optional; we lazily load it if available
try:
    import torch.hub as _torch_hub
except ImportError:  # pragma: no cover - torch is already present but keep safe guard
    _torch_hub = None


def calculate_snr(original, protected):
    noise = original - protected
    signal_power = np.sum(original**2)
    noise_power = np.sum(noise**2)
    if noise_power == 0:
        return float("inf")
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def _is_cjk_char(ch: str) -> bool:
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def _contains_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    try:
        return any(_is_cjk_char(ch) for ch in str(text or ""))
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


def _transcribe_audio(audio_path: Path, whisper_model, cache: dict, logger):
    path_str = str(audio_path)
    if path_str in cache:
        return cache[path_str]

    try:
        with torch.no_grad():
            result = whisper_model.transcribe(
                path_str,
                language="en",
                fp16=whisper_model.device.type != "cpu",
            )
        text = result.get("text", "").strip()
        # Convert traditional Chinese to simplified Chinese immediately after transcription
        if _contains_cjk(text):
            text = _to_simplified_zh(text)
    except Exception as exc:
        logger.warning(f"Failed to transcribe {path_str} for WER calculation: {exc}")
        text = ""

    cache[path_str] = text
    return text


def _load_and_align(orig_path: Path, prot_path: Path, target_sr: int, logger):
    try:
        original_wav, sr_orig = torchaudio.load(orig_path)
        protected_wav, sr_prot = torchaudio.load(prot_path)
    except Exception as e:
        logger.warning(f"Could not load {orig_path} or {prot_path}. Error: {e}")
        return None, None

    if sr_orig != target_sr:
        original_wav = torchaudio.functional.resample(original_wav, sr_orig, target_sr)
    if sr_prot != target_sr:
        protected_wav = torchaudio.functional.resample(
            protected_wav, sr_prot, target_sr
        )

    original_wav = original_wav.mean(dim=0)
    protected_wav = protected_wav.mean(dim=0)

    min_len = min(len(original_wav), len(protected_wav))
    if min_len <= 0:
        return None, None
    return original_wav[:min_len], protected_wav[:min_len]


def _build_protected_indexes(protected_dir: Path, logger=None):
    """
    Scan the protected directory (recursing into subfolders) and build:
      - stem_map: {original stem -> protected file Path}. Includes both
        `<stem>_(spec|safespeech)_protected.wav` and plain stems so we prefer
        exact filename matches.
      - idx_map:  {last two numeric segments 'A_B' -> [Path, ...]} for filenames containing ..._A_B.wav
      - order_list: fallback list ordered by index (matches 0_SPEC_i / 0_SafeSpeech_i, etc.)
    """
    stem_map = {}
    idx_map = {}
    order_list = []

    re_new = re.compile(
        r"^(?P<stem>.+)_(?:spec|safespeech)_protected\.wav$", re.IGNORECASE
    )
    re_idx = re.compile(r".*_(?P<a>\d+?)_(?P<b>\d+?)\.wav$", re.IGNORECASE)
    re_order = re.compile(r"^0_(?:SPEC|SafeSpeech)_(?P<i>\d+)\.wav$", re.IGNORECASE)

    for p in sorted(protected_dir.rglob("*.wav")):
        name = p.name

        m_new = re_new.match(name)
        if m_new:
            stem_map.setdefault(m_new.group("stem"), p)

        # Always index by the plain stem to allow direct filename matches (e.g., 1089_XXXX.wav).
        stem = p.stem
        stem_map.setdefault(stem, p)

        m_idx = re_idx.match(name)
        if m_idx:
            key = f"{m_idx.group('a')}_{m_idx.group('b')}"
            paths = idx_map.get(key)
            if paths is None:
                idx_map[key] = [p]
            else:
                paths.append(p)
                if logger is not None:
                    logger.debug("Multiple protected files share suffix key %s: %s", key, paths)

        m_ord = re_order.match(name)
        if m_ord:
            order_list.append((int(m_ord.group("i")), p))

    # Fallback ordering: sort by the extracted index
    order_list.sort(key=lambda x: x[0])
    order_list = [p for _, p in order_list]

    return stem_map, idx_map, order_list


def _load_speechmos_predictor(logger, device):
    """Load the SpeechMOS UTMOS predictor used for perceptual MOS."""

    if _torch_hub is None:
        logger.warning("Torch Hub unavailable; skipping SpeechMOS (UTMOS) metric.")
        return None

    repo = "tarepan/SpeechMOS"
    entry_point = "utmos22_strong"

    try:
        predictor = _torch_hub.load(
            repo,
            entry_point,
            trust_repo=True,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Unable to load SpeechMOS predictor from repo %s: %s",
            repo,
            exc,
        )
        return None

    try:
        predictor = predictor.to(device if hasattr(device, "type") else "cpu")
    except Exception:
        # Some implementations expose a plain callable without .to(); fall back to CPU.
        predictor = predictor.cpu() if hasattr(predictor, "cpu") else predictor

    predictor = predictor.eval()

    logger.info("Loaded SpeechMOS predictor '%s' from %s.", entry_point, repo)
    return predictor


def _infer_predictor_device(predictor):
    """Best-effort detection of the device a predictor expects tensors on."""

    if predictor is None:
        return None

    params = getattr(predictor, "parameters", None)
    if callable(params):  # torch.nn.Module exposes parameters() iterator
        try:
            first_param = next(params())
        except (StopIteration, TypeError):
            first_param = None
        else:
            if hasattr(first_param, "device"):
                return first_param.device

    dev_attr = getattr(predictor, "device", None)
    if isinstance(dev_attr, torch.device):
        return dev_attr
    if isinstance(dev_attr, str):
        try:
            return torch.device(dev_attr)
        except (RuntimeError, ValueError):
            return None

    return None


def _predict_speechmos_mos(
    predictor, prot_path: Path, waveform: torch.Tensor, sample_rate: int, logger
):
    """Run the UTMOS predictor following the SpeechMOS calling convention.

    SpeechMOS expects a batch ``(B, T)`` tensor and sampling rate. We keep a
    few defensive fallbacks for other possible interfaces, but prioritise the
    SpeechMOS-style call.
    """

    # Ensure mono batch tensor on CPU for resampling inside SpeechMOS.
    wav_batch = waveform.detach().to(torch.float32).unsqueeze(0)

    target_device = _infer_predictor_device(predictor)
    if target_device is not None and target_device.type != "cpu":
        try:
            wav_batch = wav_batch.to(target_device)
        except RuntimeError as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to move waveform batch to %s for UTMOS evaluation: %s",
                target_device,
                exc,
            )
            wav_batch = wav_batch.cpu()
    else:
        wav_batch = wav_batch.cpu()

    # Primary path: SpeechMOS call signature (batch_tensor, sample_rate)
    call_targets = [
        (predictor, (wav_batch, sample_rate), {}),
    ]

    # Explicit method variants in case predictor overrides __call__ differently
    for attr in ("predict", "infer", "forward"):
        fn = getattr(predictor, attr, None)
        if fn is not None:
            call_targets.append((fn, (wav_batch, sample_rate), {}))

    # Fallbacks: allow tensor-only or path-based signatures for compatibility
    for attr in ("predict", "__call__", "infer", "evaluate"):
        fn = getattr(predictor, attr, None)
        if fn is not None:
            call_targets.append((fn, (str(prot_path),), {}))

    call_targets.append((predictor, (wav_batch,), {}))

    for fn, args, kwargs in call_targets:
        try:
            with torch.no_grad():
                score = fn(*args, **kwargs)
        except TypeError:
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "UTMOS call %s%r failed: %s", getattr(fn, "__name__", fn), args, exc
            )
            continue

        if isinstance(score, torch.Tensor):
            if score.numel() == 0:
                continue
            return float(score.detach().cpu().view(-1)[0].item())

        try:
            return float(score)
        except (TypeError, ValueError):
            logger.debug("UTMOS returned unsupported type %s", type(score))
            continue

    return None


_DNSMOS_PRIMARY_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
_DNSMOS_P808_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx"
_DNSMOS_PERSONALIZED_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"


def _download_file(url: str, destination: Path, logger) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as sink:
            shutil.copyfileobj(response, sink)
    except Exception as exc:  # pragma: no cover - network failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    tmp_path.replace(destination)
    if logger is not None:
        logger.info("Downloaded %s to %s", url, destination)


class _DNSMOSPredictor:
    SAMPLING_RATE = 16000
    INPUT_LENGTH = 9.01

    def __init__(
        self,
        primary_model_path: Path,
        p808_model_path: Path,
        *,
        personalized: bool = False,
    ):
        import librosa
        import numpy as np
        import onnxruntime as ort
        import soundfile as sf

        self.librosa = librosa
        self.np = np
        self.sf = sf
        self.ort = ort
        self.personalized = personalized
        self.onnx_sess = ort.InferenceSession(str(primary_model_path))
        self.p808_sess = ort.InferenceSession(str(p808_model_path))

    def _audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160):
        mel_spec = self.librosa.feature.melspectrogram(
            y=audio,
            sr=self.SAMPLING_RATE,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel_spec = (self.librosa.power_to_db(mel_spec, ref=self.np.max) + 40) / 40
        return mel_spec.T

    def _polyfit(self, sig, bak, ovr):
        np = self.np
        if self.personalized:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def __call__(self, audio_path: Path):
        audio, sr = self.sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self._score_from_array(audio, sr)

    def _score_from_array(self, audio, sample_rate: int):
        np = self.np
        librosa = self.librosa

        if sample_rate != self.SAMPLING_RATE:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=self.SAMPLING_RATE
            )

        len_samples = int(self.INPUT_LENGTH * self.SAMPLING_RATE)
        if len(audio) == 0:
            return None
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        hop_len_samples = self.SAMPLING_RATE
        num_hops = (
            int(np.floor(len(audio) / self.SAMPLING_RATE) - self.INPUT_LENGTH) + 1
        )
        if num_hops <= 0:
            num_hops = 1

        sig_vals = []
        bak_vals = []
        ovr_vals = []
        sig_raw = []
        bak_raw = []
        ovr_raw = []
        p808_vals = []

        for idx in range(num_hops):
            start = int(idx * hop_len_samples)
            end = int((idx + self.INPUT_LENGTH) * hop_len_samples)
            audio_seg = audio[start:end]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg, dtype=np.float32)[np.newaxis, :]
            mel_input = self._audio_melspec(audio_seg[:-160]).astype(np.float32)[
                np.newaxis, :, :
            ]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(
                None, {"input_1": input_features}
            )[0][0]
            mos_sig, mos_bak, mos_ovr = self._polyfit(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw
            )
            p808_mos = self.p808_sess.run(None, {"input_1": mel_input})[0][0][0]

            sig_vals.append(mos_sig)
            bak_vals.append(mos_bak)
            ovr_vals.append(mos_ovr)
            sig_raw.append(mos_sig_raw)
            bak_raw.append(mos_bak_raw)
            ovr_raw.append(mos_ovr_raw)
            p808_vals.append(p808_mos)

        if not ovr_vals:
            return None

        return {
            "ovrl": float(np.mean(ovr_vals)),
            "sig": float(np.mean(sig_vals)),
            "bak": float(np.mean(bak_vals)),
            "ovrl_raw": float(np.mean(ovr_raw)),
            "sig_raw": float(np.mean(sig_raw)),
            "bak_raw": float(np.mean(bak_raw)),
            "p808_mos": float(np.mean(p808_vals)),
        }


def _load_dnsmos_predictor(logger, *, personalized: bool = False):
    model_dir = Path(to_absolute_path("checkpoints/dnsmos"))
    primary_name = "pdnsmos_sig_bak_ovr.onnx" if personalized else "sig_bak_ovr.onnx"
    primary_url = _DNSMOS_PERSONALIZED_URL if personalized else _DNSMOS_PRIMARY_URL
    primary_path = model_dir / primary_name
    p808_path = model_dir / "model_v8.onnx"

    try:
        _download_file(primary_url, primary_path, logger)
        _download_file(_DNSMOS_P808_URL, p808_path, logger)
    except Exception as exc:
        if logger is not None:
            logger.warning("Unable to download DNSMOS models: %s", exc)
        return None

    try:
        predictor = _DNSMOSPredictor(primary_path, p808_path, personalized=personalized)
    except Exception as exc:
        if logger is not None:
            logger.warning("Unable to initialize DNSMOS predictor: %s", exc)
        return None

    if logger is not None:
        logger.info("Loaded DNSMOS predictor (personalized=%s).", personalized)
    return predictor


def _predict_dnsmos_scores(predictor, audio_path: Path, logger):
    if predictor is None:
        return None
    try:
        return predictor(audio_path)
    except Exception as exc:
        if logger is not None:
            logger.warning("DNSMOS failed on %s: %s", audio_path, exc)
        return None


def make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        # If it's a single value (scalar), convert to float/int
        if obj.numel() == 1:
            return obj.item()
        # If it's a list/matrix, convert to list
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    else:
        return obj


import string


def preprocess_text(text):
    # 1. 将所有标点符号替换为空格
    # re.escape 确保 string.punctuation 中的特殊字符被正确转义
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # 2. 将连续的多个空格合并为一个，并转为小写
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def evaluate(
    original_files,
    protected_audio_dir,
    logger,
    target_sr: int = 24000,
    whisper_model_name: str = "base.en",
    transcript_map: Optional[dict] = None,
    bootstrap_config=None,
    sample_metrics_path: Optional[Path] = None,
):
    """
    Compute SNR, STOI, WER, SIM, and Naturalness (UTMOS) between original and
    protected audios.
    Priority for matching:
      1) <stem>_(spec|safespeech)_protected.wav
      2) * * _<last2parts>.wav  (e.g.,  *_*_000062_000000.wav)
      3) Next: 0_SPEC_i / 0_SafeSpeech_i
    """
    protected_dir = Path(protected_audio_dir)
    stoi_loss_func = NegSTOILoss(sample_rate=target_sr)
    bootstrapper = bootstrap_utils.create_bootstrapper(bootstrap_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    whisper_model = None
    sim_model = None
    transcription_cache = {}
    wer_pairs = 0
    total_wer = 0.0
    compute_wer = transcript_map is not None and len(transcript_map) > 0

    if compute_wer:
        try:
            logger.info(
                f"Loading Whisper model '{whisper_model_name}' on {device} for WER calculation..."
            )
            whisper_model = whisper.load_model(whisper_model_name, device=device)
        except Exception as exc:
            logger.warning(
                f"Unable to load Whisper model '{whisper_model_name}'. Skipping WER evaluation. Error: {exc}"
            )
            whisper_model = None
            compute_wer = False
    else:
        logger.info(
            "Skipping WER calculation because no train transcripts were provided."
        )

    # Speaker Similarity model
    try:
        logger.info(
            f"Loading ECAPA-TDNN speaker encoder on {device} for SIM calculation..."
        )
        sim_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="checkpoints/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
    except Exception as exc:
        logger.warning(
            f"Unable to load ECAPA-TDNN speaker encoder. Skipping SIM evaluation. Error: {exc}"
        )
        sim_model = None

    stem_map, idx_map, order_list = _build_protected_indexes(protected_dir, logger=logger)

    speechmos_model = _load_speechmos_predictor(logger, device)
    dnsmos_model = _load_dnsmos_predictor(logger)
    try:
        mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Unable to initialize MCD calculator. Skipping MCD metric. Error: %s", exc
        )
        mcd_toolbox = None

    total_snr = 0.0
    snr_scores = []
    stoi_scores = []
    wer_scores = []
    total_sim = 0.0
    sim_scores = []
    speechmos_scores = []
    dnsmos_scores = {"ovrl": [], "sig": [], "bak": []}
    mcd_scores = []
    sample_metrics = []
    matched = 0
    missing = []
    sim_pairs = 0
    mcd_pairs = 0

    logger.info(
        "Calculating Fidelity Metrics (SNR, STOI, WER, SIM, SpeechMOS, DNSMOS, MCD)..."
    )
    for i, original_path_str in enumerate(tqdm(original_files)):
        orig_path = Path(original_path_str)
        stem = orig_path.stem

        # 1) First try an exact stem + suffix match
        prot_path = stem_map.get(stem)

        # 2) Then use the last two index segments
        if prot_path is None:
            parts = stem.split("_")
            if len(parts) >= 2:
                key = f"{parts[-2]}_{parts[-1]}"
                candidates = idx_map.get(key)
                if candidates:
                    if len(candidates) > 1:
                        # Prefer an exact filename match if multiple share the same numeric suffix.
                        prot_path = next((p for p in candidates if p.name == orig_path.name), candidates[0])
                        logger.warning(
                            "Multiple protected files match suffix %s; using %s for %s",
                            key,
                            prot_path,
                            orig_path.name,
                        )
                    else:
                        prot_path = candidates[0]

        # 3) Finally fall back to the ordered list
        if prot_path is None and i < len(order_list):
            prot_path = order_list[i]

        if prot_path is None:
            missing.append(stem)
            continue

        assert orig_path.name == prot_path.name
        original_wav, protected_wav = _load_and_align(orig_path, prot_path, target_sr, logger)
        if original_wav is None:
            continue

        sample_entry = {
            "ground_truth_path": str(orig_path),
            "protected_path": str(prot_path),
            "ground_truth_text": None,
            "predicted_text": None,
            "snr_db": None,
            "stoi": None,
            "wer": None,
            "sim": None,
            "speechmos_mos": None,
            "dnsmos_ovrl": None,
            "dnsmos_sig": None,
            "dnsmos_bak": None,
            "mcd": None,
            "protected_duration_sec": None,
        }

        try:
            sample_entry["protected_duration_sec"] = float(
                protected_wav.shape[-1] / float(target_sr)
            )
        except Exception:
            sample_entry["protected_duration_sec"] = None

        if transcript_map is not None:
            gt_text = transcript_map.get(original_path_str)
            if gt_text is None:
                gt_text = transcript_map.get(str(orig_path))
            sample_entry["ground_truth_text"] = gt_text
        else:
            gt_text = None

        if mcd_toolbox is not None:
            try:
                mcd_val = float(
                    mcd_toolbox.calculate_mcd(str(orig_path), str(prot_path))
                )
            except Exception as exc:
                logger.warning(
                    "Failed to compute MCD for %s vs %s: %s",
                    orig_path.name,
                    prot_path.name,
                    exc,
                )
            else:
                mcd_scores.append(mcd_val)
                mcd_pairs += 1
                sample_entry["mcd"] = mcd_val

        snr_val = float(calculate_snr(original_wav.numpy(), protected_wav.numpy()))
        total_snr += snr_val
        snr_scores.append(snr_val)
        sample_entry["snr_db"] = snr_val

        # 1. 确保是 2D [Batch, Time]
        if original_wav.dim() == 1:
            orig = original_wav.unsqueeze(0)
            prot = protected_wav.unsqueeze(0)
        else:
            orig = original_wav
            prot = protected_wav

        current_len = orig.shape[-1]

        # 2. 如果长度不够，进行 Padding
        if current_len < target_sr:
            pad_amount = target_sr - current_len
            # 在最后一个维度右侧补零
            orig = F.pad(orig, (0, pad_amount), mode="constant", value=0)
            prot = F.pad(prot, (0, pad_amount), mode="constant", value=0)
        stoi_val = -stoi_loss_func(orig, prot).item()
        # stoi_val = -stoi_loss_func(original_wav.unsqueeze(0), protected_wav.unsqueeze(0)).item()
        stoi_scores.append(stoi_val)
        sample_entry["stoi"] = stoi_val

        if compute_wer and whisper_model is not None:
            if not gt_text:
                logger.warning(
                    f"Missing transcript for {orig_path}. Skipping WER for this pair."
                )
            else:
                prot_text = _transcribe_audio(
                    prot_path, whisper_model, transcription_cache, logger
                )
                try:
                    # Remove punctuation before WER calculation to avoid inflated error rates
                    # import string

                    # gt_no_punct = gt_text.translate(
                    #     str.maketrans("", "", string.punctuation)
                    # ).lower()
                    # prot_no_punct = prot_text.translate(
                    #     str.maketrans("", "", string.punctuation)
                    # ).lower()
                    gt_no_punct = preprocess_text(gt_text)
                    prot_no_punct = preprocess_text(prot_text)
                    wer_val = jiwer.wer(gt_no_punct, prot_no_punct)
                except Exception as exc:
                    logger.warning(
                        f"Failed to compute WER for {orig_path.name} vs {prot_path.name}: {exc}"
                    )
                else:
                    total_wer += wer_val
                    wer_pairs += 1
                    wer_scores.append(wer_val)
                    sample_entry["wer"] = wer_val
                    sample_entry["ground_truth_text"] = gt_text
                    sample_entry["predicted_text"] = prot_text

        if sim_model is not None:
            try:
                sim_score, _ = sim_model.verify_files(str(orig_path), str(prot_path))
                if isinstance(sim_score, torch.Tensor):
                    sim_score = sim_score.squeeze().item()
                sim_score = float(sim_score)
            except Exception as exc:
                logger.warning(
                    "Failed to compute SIM for %s vs %s: %s",
                    orig_path.name,
                    prot_path.name,
                    exc,
                )
            else:
                total_sim += sim_score
                sim_scores.append(sim_score)
                sim_pairs += 1
                sample_entry["sim"] = sim_score

        speechmos_val = None
        if speechmos_model is not None:
            speechmos_val = _predict_speechmos_mos(
                speechmos_model,
                prot_path,
                protected_wav,
                target_sr,
                logger,
            )
            if speechmos_val is not None:
                speechmos_scores.append(speechmos_val)
                sample_entry["speechmos_mos"] = speechmos_val

        dnsmos_val = _predict_dnsmos_scores(dnsmos_model, prot_path, logger)
        if dnsmos_val is not None:
            for key in ("ovrl", "sig", "bak"):
                dnsmos_scores[key].append(dnsmos_val[key])
            sample_entry["dnsmos_ovrl"] = dnsmos_val.get("ovrl")
            sample_entry["dnsmos_sig"] = dnsmos_val.get("sig")
            sample_entry["dnsmos_bak"] = dnsmos_val.get("bak")

        matched += 1
        sample_metrics.append(sample_entry)

    if matched == 0:
        return {
            "error": "No matching protected files found for fidelity evaluation.",
            "num_pairs": 0,
            "num_missing": len(missing),
        }

    result = {
        "avg_snr_db": total_snr / matched,
        "num_pairs": matched,
        "num_missing": len(missing),
    }

    if stoi_scores:
        result["avg_stoi"] = sum(stoi_scores) / len(stoi_scores)
        result["stoi_pairs"] = len(stoi_scores)
        result["min_stoi"] = min(stoi_scores)
        result["max_stoi"] = max(stoi_scores)
    else:
        result["avg_stoi"] = None
        result["stoi_pairs"] = 0
        result["min_stoi"] = None
        result["max_stoi"] = None

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

    if mcd_scores:
        result["avg_mcd"] = sum(mcd_scores) / len(mcd_scores)
        result["mcd_pairs"] = len(mcd_scores)
        result["min_mcd"] = min(mcd_scores)
        result["max_mcd"] = max(mcd_scores)
    else:
        result["avg_mcd"] = None
        result["mcd_pairs"] = 0
        result["min_mcd"] = None
        result["max_mcd"] = None

    if sim_model is not None and sim_pairs > 0:
        result["avg_sim"] = total_sim / sim_pairs
        result["sim_pairs"] = sim_pairs
    else:
        result["avg_sim"] = None
        result["sim_pairs"] = 0

    if compute_wer and whisper_model is not None and wer_pairs > 0:
        result["avg_wer"] = total_wer / wer_pairs
        result["wer_pairs"] = wer_pairs
    else:
        result["avg_wer"] = None
        result["wer_pairs"] = 0

    if sample_metrics:
        csv_path = (
            Path(sample_metrics_path)
            if sample_metrics_path is not None
            else protected_dir.parent / "fidelity_sample_metrics.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "ground_truth_path",
            "protected_path",
            "ground_truth_text",
            "predicted_text",
            "snr_db",
            "stoi",
            "wer",
            "sim",
            "speechmos_mos",
            "dnsmos_ovrl",
            "dnsmos_sig",
            "dnsmos_bak",
            "mcd",
            "protected_duration_sec",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_metrics)
        logger.info("Saved sample-wise fidelity metrics to %s", csv_path)
        result["sample_metrics_csv"] = str(csv_path)
    else:
        result["sample_metrics_csv"] = None

    if bootstrapper is not None:
        bootstrapper.maybe_add_interval(result, "avg_snr_db", snr_scores)
        bootstrapper.maybe_add_interval(result, "avg_stoi", stoi_scores)
        bootstrapper.maybe_add_interval(result, "avg_wer", wer_scores)
        bootstrapper.maybe_add_interval(result, "avg_sim", sim_scores)
        bootstrapper.maybe_add_interval(result, "avg_speechmos_mos", speechmos_scores)
        bootstrapper.maybe_add_interval(result, "avg_mcd", mcd_scores)
        for key in ("ovrl", "sig", "bak"):
            bootstrapper.maybe_add_interval(
                result, f"avg_dnsmos_{key}", dnsmos_scores[key]
            )

    return result
