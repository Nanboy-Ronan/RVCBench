"""StyleTTS2 synthesizer used by the off-the-shelf adversary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class StyleTTS2SynthesizerConfig:
    code_path: Path
    config_path: Path
    checkpoint_path: Path
    alpha: float = 0.3
    beta: float = 0.7
    diffusion_steps: int = 5
    embedding_scale: float = 1.0
    sample_rate: int = 24000
    tail_trim: int = 50


class StyleTTS2Synthesizer(BaseModel):
    """Handles StyleTTS2 model loading, style extraction, and inference."""

    def __init__(
        self,
        config: StyleTTS2SynthesizerConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.checkpoint_path),
            device=device,
            logger=logger,
        )
        self.config = config

        self._imports_loaded = False
        self._style_cache: Dict[Path, torch.Tensor] = {}

        self._model = None
        self._model_params = None
        self._sampler = None
        self._text_cleaner = None
        self._phonemizer = None
        self._length_to_mask = None
        self._word_tokenize = None
        self._mel_transform = None
        self._mean = None
        self._std = None

        self._validate_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def synthesize(self, text: str, reference_path: Path) -> np.ndarray:
        """Generate an utterance conditioned on the provided reference audio."""
        self.ensure_model()
        ref_features = self._compute_style(reference_path)
        wav = self._run_diffusion(text, ref_features)
        if wav.size == 0:
            return wav
        if self.config.tail_trim > 0 and wav.shape[-1] > self.config.tail_trim:
            wav = wav[..., :-self.config.tail_trim]
        return wav.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"StyleTTS2 code path not found: {self.config.code_path}")
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"StyleTTS2 config not found: {self.config.config_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"StyleTTS2 checkpoint not found: {self.config.checkpoint_path}")

    def load_model(self) -> None:
        if self._imports_loaded:
            return

        import os
        import sys
        import yaml
        from munch import Munch

        code_path = self.config.code_path
        if str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))

        # StyleTTS2 checkpoints rely on pickled state dicts; Torch 2.6+ defaults to weights_only=True.
        if not getattr(torch.load, "_styletts2_patched", False):
            _orig_torch_load = torch.load

            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_torch_load(*args, **kwargs)

            _patched_load._styletts2_patched = True  # type: ignore[attr-defined]
            torch.load = _patched_load  # type: ignore[assignment]

        # Some installations of monotonic_align miss mask_from_lens; add a safe fallback
        try:
            import monotonic_align as ma

            if not hasattr(ma, "mask_from_lens"):
                def mask_from_lens(attn, input_lengths, mel_length):
                    lengths = input_lengths
                    if isinstance(input_lengths, torch.Tensor):
                        lengths = input_lengths.detach().cpu().tolist()
                    targets = mel_length
                    if not isinstance(targets, (list, tuple)):
                        targets = [targets] * len(lengths)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.detach().cpu().tolist()
                    mask = torch.ones_like(attn, dtype=torch.bool)
                    for idx, (src_len, tgt_len) in enumerate(zip(lengths, targets)):
                        t_len = min(int(tgt_len), attn.shape[1])
                        s_len = min(int(src_len), attn.shape[2])
                        mask[idx, :t_len, :s_len] = False
                    return mask

                ma.mask_from_lens = mask_from_lens
                sys.modules["monotonic_align"] = ma
        except Exception:
            pass

        try:
            import librosa
            import torchaudio
            from phonemizer.backend import EspeakBackend
        except ImportError as exc:
            raise ImportError(
                "Missing dependency for StyleTTS2. Install yaml, munch, librosa, torchaudio, and phonemizer."
            ) from exc

        try:
            from nltk.tokenize import word_tokenize as nltk_word_tokenize
        except ImportError:
            def nltk_word_tokenize(text: str) -> List[str]:
                return text.split()

        def _word_tokenize(text: str) -> List[str]:
            try:
                return nltk_word_tokenize(text)
            except LookupError:
                return text.split()

        try:
            from models import build_model, load_ASR_models, load_F0_models
            from modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule  # type: ignore
        except ModuleNotFoundError:
            from models import build_model, load_ASR_models, load_F0_models
            from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule  # type: ignore

        from utils import recursive_munch, length_to_mask
        from text_utils import TextCleaner
        from Utils.PLBERT.util import load_plbert

        prev_cwd = Path.cwd()
        os.chdir(code_path)
        try:
            with open(self.config.config_path, "r", encoding="utf-8") as handle:
                raw_config = yaml.safe_load(handle)

            def _resolve(path_value):
                if not path_value:
                    return None
                path_obj = Path(path_value)
                if not path_obj.is_absolute():
                    path_obj = (code_path / path_obj).resolve()
                return path_obj

            asr_config = _resolve(raw_config.get("ASR_config"))
            asr_path = _resolve(raw_config.get("ASR_path"))
            f0_path = _resolve(raw_config.get("F0_path"))
            plbert_dir = _resolve(raw_config.get("PLBERT_dir"))
            
            if not asr_config or not asr_config.exists():
                raise FileNotFoundError("StyleTTS2 ASR config file not found. Update the adversary config paths.")
            if not asr_path or not asr_path.exists():
                raise FileNotFoundError("StyleTTS2 ASR checkpoint not found. Update the adversary config paths.")
            if not f0_path or not f0_path.exists():
                raise FileNotFoundError("StyleTTS2 F0 checkpoint not found. Update the adversary config paths.")
            if not plbert_dir or not plbert_dir.exists():
                raise FileNotFoundError("StyleTTS2 PL-BERT directory not found. Update the adversary config paths.")

            text_aligner = load_ASR_models(str(asr_path), str(asr_config))
            pitch_extractor = load_F0_models(str(f0_path))
            plbert = load_plbert(str(plbert_dir))

            model_params = recursive_munch(raw_config["model_params"])
            model = build_model(model_params, text_aligner, pitch_extractor, plbert)
            params_whole = torch.load(self.config.checkpoint_path, map_location="cpu")
            params = params_whole["net"]
            for key in model:
                if key in params:
                    try:
                        model[key].load_state_dict(params[key])
                    except RuntimeError:
                        from collections import OrderedDict

                        state_dict = params[key]
                        new_state_dict = OrderedDict()
                        for k, value in state_dict.items():
                            new_key = k[7:] if k.startswith("module.") else k
                            new_state_dict[new_key] = value
                        model[key].load_state_dict(new_state_dict, strict=False)
            for key in model:
                model[key].eval()
                model[key].to(self.device)

            sampler = DiffusionSampler(
                model.diffusion.diffusion,
                sampler=ADPM2Sampler(),
                sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                clamp=False,
            )
        finally:
            os.chdir(prev_cwd)

        self._model = model
        self._model_params = model_params
        self._sampler = sampler
        self._text_cleaner = TextCleaner()
        self._phonemizer = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
        self._length_to_mask = length_to_mask
        self._word_tokenize = _word_tokenize
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_mels=80,
            n_fft=2048,
            win_length=1200,
            hop_length=300,
        )
        self._mean = -4
        self._std = 4

        self._imports_loaded = True

    def _compute_style(self, path: Path) -> torch.Tensor:
        cached = self._style_cache.get(path)
        if cached is not None:
            return cached

        import librosa

        wave, sr = librosa.load(str(path), sr=None)
        if sr != self.config.sample_rate:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.config.sample_rate)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        if len(audio) == 0:
            audio = wave

        wave_tensor = torch.from_numpy(audio).float()
        mel_tensor = self._mel_transform(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self._mean) / self._std
        mel_tensor = mel_tensor.to(self.device)

        # Style encoder down-samples the time axis 4x (factor 16) before a 5x5 conv.
        # Ensure the input has at least 5 frames after downsampling => 5 * 16 = 80 frames pre-downsample.
        min_frames = 80
        if mel_tensor.shape[-1] < min_frames:
            pad_frames = min_frames - mel_tensor.shape[-1]
            mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_frames), mode="replicate")

        assert self._model is not None
        with torch.no_grad():
            ref_s = self._model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self._model.predictor_encoder(mel_tensor.unsqueeze(1))
            style = torch.cat([ref_s, ref_p], dim=1)

        self._style_cache[path] = style
        return style

    def _phoneme_tokens(self, text: str) -> torch.Tensor:
        assert self._phonemizer is not None
        assert self._word_tokenize is not None
        assert self._text_cleaner is not None

        text = text.strip()
        phonemes = self._phonemizer.phonemize([text])[0]
        tokens = self._word_tokenize(phonemes)
        normalized = " ".join(tokens)
        token_ids = self._text_cleaner(normalized)
        token_ids.insert(0, 0)
        if len(token_ids) == 0:
            token_ids = [0]
        return torch.LongTensor(token_ids).to(self.device).unsqueeze(0)

    def _run_diffusion(self, text: str, ref_features: torch.Tensor) -> np.ndarray:
        assert self._model is not None
        assert self._model_params is not None
        assert self._sampler is not None
        assert self._length_to_mask is not None
        
        tokens = self._phoneme_tokens(text)
        if tokens.size(1) > 512:
            tokens = tokens[:, :512]
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = self._length_to_mask(input_lengths).to(self.device)

        with torch.no_grad():
            model = self._model
            sampler = self._sampler

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            noise = torch.randn((1, 256), device=self.device).unsqueeze(1)
            s_pred = sampler(
                noise=noise,
                embedding=bert_dur,
                embedding_scale=self.config.embedding_scale,
                features=ref_features,
                num_steps=self.config.diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = self.config.alpha * ref + (1.0 - self.config.alpha) * ref_features[:, :128]
            s = self.config.beta * s + (1.0 - self.config.beta) * ref_features[:, 128:]

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)

            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            total_frames = int(pred_dur.sum().item())
            alignment = torch.zeros((tokens.shape[-1], total_frames), device=self.device)
            cursor = 0
            for i in range(alignment.size(0)):
                span = max(int(pred_dur[i].item()), 1)
                end = min(cursor + span, total_frames)
                alignment[i, cursor:end] = 1.0
                cursor = end
            alignment = alignment.unsqueeze(0)

            en = torch.matmul(d.transpose(-1, -2), alignment)
            if self._model_params.decoder.type == "hifigan":
                shifted = torch.zeros_like(en)
                shifted[:, :, 0] = en[:, :, 0]
                shifted[:, :, 1:] = en[:, :, :-1]
                en = shifted

            f0_pred, n_pred = model.predictor.F0Ntrain(en, s)

            asr = torch.matmul(t_en, alignment)
            if self._model_params.decoder.type == "hifigan":
                shifted = torch.zeros_like(asr)
                shifted[:, :, 0] = asr[:, :, 0]
                shifted[:, :, 1:] = asr[:, :, :-1]
                asr = shifted

            waveform = model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))

        wav = waveform.squeeze().detach().cpu().numpy()
        return wav
