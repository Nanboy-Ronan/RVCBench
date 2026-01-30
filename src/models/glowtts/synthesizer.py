"""Glow-TTS synthesizer wrapper used by off-the-shelf adversary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class GlowTTSSynthesizerConfig:
    code_path: Path
    config_path: Path
    checkpoint_path: Path
    noise_scale: float = 0.667
    length_scale: float = 1.0
    vocoder_type: str = "waveglow"
    waveglow_sigma: float = 0.666
    waveglow_checkpoint: Optional[Path] = None
    waveglow_code_path: Optional[Path] = None
    griffinlim_iters: int = 60
    sample_rate: int = 22050
    speaker_map: Optional[Dict[str, int]] = None


class GlowTTSSynthesizer(BaseModel):
    """Encapsulates Glow-TTS model and vocoder loading/inference."""

    def __init__(
        self,
        config: GlowTTSSynthesizerConfig,
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
        self._commons = None
        self._text_to_sequence = None
        self._symbols: Optional[List[str]] = None
        self._text_cleaners: List[str] = []
        self._cmu_dict = None
        self._model = None
        self._add_blank = False
        self._n_speakers = 0
        self._has_speaker_embedding = False
        self._waveglow = None
        self._griffinlim_stft = None

        self._speaker_cache: Dict[str, int] = {}
        self._speaker_mapping: Dict[str, int] = config.speaker_map or {}

        self._validate_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def synthesize(self, text: str, speaker_label: str) -> np.ndarray:
        """Generate speech for the provided text and speaker label."""
        self.ensure_model()
        speaker_id = self._resolve_speaker_id(speaker_label)
        sequence, lengths = self._prepare_text(text)
        wav = self._synthesize(sequence, lengths, speaker_id)
        return wav.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"Glow-TTS code path not found: {self.config.code_path}")
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"Glow-TTS config not found: {self.config.config_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"Glow-TTS checkpoint not found: {self.config.checkpoint_path}")
        if (
            self.config.vocoder_type.lower() == "waveglow"
            and self.config.waveglow_checkpoint is not None
            and not self.config.waveglow_checkpoint.exists()
        ):
            raise FileNotFoundError(f"WaveGlow checkpoint not found: {self.config.waveglow_checkpoint}")

    def load_model(self) -> None:
        if self._imports_loaded:
            return

        import json
        import sys

        code_path = str(self.config.code_path)
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        try:
            import numpy as _np  # noqa: F401
            import librosa  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Glow-TTS synthesizer requires numpy and librosa to be installed."
            ) from exc

        try:
            import utils as glow_utils
            import models
            import commons
            from text import cmudict, text_to_sequence
            from text.symbols import symbols
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Glow-TTS Python modules could not be imported. Ensure the repository is available."
            ) from exc

        self._commons = commons
        self._text_to_sequence = text_to_sequence
        self._symbols = symbols

        hps = glow_utils.get_hparams_from_file(str(self.config.config_path))
        self._add_blank = getattr(hps.data, "add_blank", False)
        self._text_cleaners = list(getattr(hps.data, "text_cleaners", ["english_cleaners"]))

        cmu_path = getattr(hps.data, "cmudict_path", None)
        if cmu_path:
            cmu_path = Path(cmu_path)
            if not cmu_path.is_absolute():
                cmu_path = (self.config.code_path / cmu_path).resolve()
            if not cmu_path.exists():
                raise FileNotFoundError(f"CMU dict not found at {cmu_path}")
            self._cmu_dict = cmudict.CMUDict(str(cmu_path))
        else:
            self._cmu_dict = None

        n_vocab = len(self._symbols) + (1 if self._add_blank else 0)
        model_kwargs = vars(hps.model)
        model = models.FlowGenerator(
            n_vocab,
            out_channels=hps.data.n_mel_channels,
            **model_kwargs,
        ).to(self.device)
        model, _, _, _ = glow_utils.load_checkpoint(str(self.config.checkpoint_path), model)
        model.decoder.store_inverse()
        model.eval()
        self._model = model

        self._n_speakers = getattr(model, "n_speakers", 0)
        self._has_speaker_embedding = hasattr(model, "emb_g") and model.emb_g is not None
        if self._has_speaker_embedding:
            self._n_speakers = model.emb_g.num_embeddings
            self.logger.info("[GlowTTS] Loaded multi-speaker model with %d speakers.", self._n_speakers)
        else:
            self.logger.info("[GlowTTS] Loaded single-speaker model.")

        vocoder_type = self.config.vocoder_type.lower()
        if vocoder_type == "waveglow":
            self._load_waveglow(glow_utils)
        elif vocoder_type == "griffinlim":
            self._load_griffinlim(hps)
        else:
            raise ValueError(f"Unsupported vocoder '{self.config.vocoder_type}'.")

        self._imports_loaded = True

    def _load_waveglow(self, glow_utils) -> None:
        import sys
        import torch

        waveglow_checkpoint = self.config.waveglow_checkpoint
        if waveglow_checkpoint is None:
            raise ValueError(
                "Glow-TTS waveglow vocoder selected but 'waveglow_checkpoint' not provided."
            )
        waveglow_path = waveglow_checkpoint.resolve()
        waveglow_code_dir = (
            self.config.waveglow_code_path.resolve()
            if self.config.waveglow_code_path is not None
            else (self.config.code_path / "waveglow").resolve()
        )
        if not waveglow_code_dir.exists():
            raise FileNotFoundError(
                "WaveGlow code directory not found. Set 'waveglow_code_path' appropriately."
            )
        waveglow_code = str(waveglow_code_dir)
        if waveglow_code not in sys.path:
            sys.path.insert(0, waveglow_code)

        try:
            import waveglow
        except ModuleNotFoundError as exc:
            raise ImportError("Failed to import WaveGlow modules.") from exc

        waveglow_obj = torch.load(str(waveglow_path), map_location=self.device)
        if isinstance(waveglow_obj, dict) and "model" in waveglow_obj:
            waveglow_model = waveglow_obj["model"]
        else:
            waveglow_model = waveglow_obj
        waveglow.remove_weightnorm(waveglow_model)
        waveglow_model.eval()
        for param in waveglow_model.parameters():
            param.requires_grad = False
        self._waveglow = waveglow_model.to(self.device)
        self.logger.info("[GlowTTS] WaveGlow vocoder loaded from %s", waveglow_path)

    def _load_griffinlim(self, hps) -> None:
        import torchaudio

        n_fft = hps.data.filter_length
        hop_length = hps.data.hop_length
        win_length = hps.data.win_length
        self._griffinlim_stft = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_iter=self.config.griffinlim_iters,
            power=1.5,
        ).to(self.device)
        self.logger.warning(
            "[GlowTTS] Using Griffin-Lim vocoder fallback; quality may be reduced."
        )

    def _resolve_speaker_id(self, speaker_label: str) -> Optional[int]:
        if speaker_label in self._speaker_mapping:
            return int(self._speaker_mapping[speaker_label])

        try:
            sid = int(speaker_label)
            if self._n_speakers > 0 and sid >= self._n_speakers:
                self.logger.debug(
                    "[GlowTTS] Speaker id %s out of range (%d). Applying modulo mapping.",
                    speaker_label,
                    self._n_speakers,
                )
                sid = sid % self._n_speakers
            return sid
        except (TypeError, ValueError):
            pass

        if not self._has_speaker_embedding:
            return None

        sid = self._speaker_cache.get(speaker_label)
        if sid is None:
            sid = len(self._speaker_cache) % max(self._n_speakers, 1)
            self._speaker_cache[speaker_label] = sid
            self.logger.warning(
                "[GlowTTS] Speaker '%s' not in mapping; assigning id %d.",
                speaker_label,
                sid,
            )
        return sid

    def _prepare_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._text_to_sequence is not None
        assert self._symbols is not None
        assert self._commons is not None

        phoneme_seq = self._text_to_sequence(text.strip(), self._text_cleaners, self._cmu_dict)
        if self._add_blank:
            phoneme_seq = self._commons.intersperse(phoneme_seq, len(self._symbols))
        if not phoneme_seq:
            phoneme_seq = [0]
        sequence = torch.LongTensor(phoneme_seq).unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([sequence.shape[1]]).to(self.device)
        return sequence, lengths

    def _synthesize(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        speaker_id: Optional[int],
    ) -> np.ndarray:
        import torch

        assert self._model is not None
        speaker_tensor = None
        if self._has_speaker_embedding and speaker_id is not None:
            speaker_tensor = torch.LongTensor([speaker_id]).to(self.device)

        with torch.no_grad():
            (mel, *_), *_ = self._model(
                sequence,
                lengths,
                g=speaker_tensor,
                gen=True,
                noise_scale=self.config.noise_scale,
                length_scale=self.config.length_scale,
            )
            mel = mel.float()

            if self.config.vocoder_type.lower() == "waveglow":
                audio = self._waveglow.infer(mel, sigma=self.config.waveglow_sigma)
                wav = audio.squeeze().cpu().numpy()
            else:
                import torchaudio

                assert self._griffinlim_stft is not None
                mel_cpu = mel.squeeze().cpu()
                inv_mel_scale = torchaudio.transforms.InverseMelScale(
                    n_stft=self._griffinlim_stft.n_fft // 2 + 1,
                    n_mels=mel_cpu.shape[0],
                    sample_rate=self.config.sample_rate,
                )
                linear_spec = inv_mel_scale(mel_cpu)
                waveform = self._griffinlim_stft(linear_spec)
                wav = waveform.squeeze().cpu().numpy()

        return wav
