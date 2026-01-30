"""Integrated OZSpeech synthesis pipeline used by the off-the-shelf adversary."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from collections import defaultdict
import importlib
import sys

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig

from src.models.model import BaseModel


SR = 16000 # TODO: add this into argument make sure load follows the model and save follows our config


@dataclass
class _ManifestEntry:
    target_stub: str
    prompt_name: str
    text: str


class OzSpeechSynthesizer(BaseModel):
    """Thin wrapper that mirrors OZSpeech's standalone ``synthesize.py`` script."""

    def __init__(
        self,
        cfg_path: Path,
        checkpoint_path: Path,
        device: str,
        temperature: float,
        logger,
        code_path: Optional[Path] = None,
        codec_encoder_path: Optional[Path] = None,
        codec_decoder_path: Optional[Path] = None,
    ) -> None:
        super().__init__(
            model_name_or_path=str(checkpoint_path),
            device=device,
            logger=logger,
        )
        self._device_string = str(self.device)
        self.cfg_path = Path(cfg_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.temperature = float(temperature)
        self.code_path = Path(code_path) if code_path is not None else None
        self.codec_encoder_path = codec_encoder_path
        self.codec_decoder_path = codec_decoder_path

        self._model = None
        self._codec_encoder = None
        self._codec_decoder = None
        self._zact_class = None
        self._facodec_encoder_class = None
        self._facodec_decoder_class = None

        self._validate_paths()

    def synthesize(self, manifest_path: Path, prompt_dir: Path, output_dir: Path) -> List[Path]:
        entries = list(self._load_manifest(manifest_path))
        if not entries:
            return []

        self.ensure_model()
        assert self._model is not None  # guarded by _ensure_resources
        assert self._codec_encoder is not None
        assert self._codec_decoder is not None

        synth_dir = output_dir / "synth"
        prior_dir = output_dir / "prior"
        synth_dir.mkdir(parents=True, exist_ok=True)
        prior_dir.mkdir(parents=True, exist_ok=True)

        generated: List[Path] = []
        infer_times: List[float] = []

        self.logger.info("[OZSpeech] Synthesizing %d prompts (temp=%.3f)...", len(entries), self.temperature)

        with torch.inference_mode():
            for entry in entries:
                prompt_path = prompt_dir / entry.prompt_name
                if not prompt_path.exists():
                    raise FileNotFoundError(
                        f"Prompt audio '{entry.prompt_name}' not found under {prompt_dir}"
                    )

                output = self._model.synthesize(
                    text=entry.text,
                    acoustic_prompt=str(prompt_path),
                    codec_encoder=self._codec_encoder,
                    codec_decoder=self._codec_decoder,
                    temperature=self.temperature,
                )

                synth_wav = output.get("synth_wav")
                if synth_wav is None:
                    raise RuntimeError("OZSpeech model returned no 'synth_wav' audio")
                synth_wav = self._to_numpy(synth_wav)

                synth_path = synth_dir / f"{entry.target_stub}.wav"
                sf.write(str(synth_path), synth_wav, SR)
                generated.append(synth_path)

                prior_wav = output.get("prior_wav")
                if prior_wav is not None:
                    prior_wav = self._to_numpy(prior_wav)
                    prior_path = prior_dir / f"{entry.target_stub}.wav"
                    sf.write(str(prior_path), prior_wav, SR)

                infer_time = output.get("time")
                if isinstance(infer_time, (float, int)):
                    infer_times.append(float(infer_time))

        if infer_times:
            avg_time = sum(infer_times) / len(infer_times)
            self.logger.info(
                "[OZSpeech] Inference time: total=%.2fs avg=%.2fs (n=%d)",
                sum(infer_times),
                avg_time,
                len(infer_times),
            )

        return generated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"OZSpeech config not found: {self.cfg_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"OZSpeech checkpoint not found: {self.checkpoint_path}")
        if self.code_path is not None and not self.code_path.exists():
            raise FileNotFoundError(f"OZSpeech code path not found: {self.code_path}")
        if self.codec_encoder_path is not None and not self.codec_encoder_path.exists():
            raise FileNotFoundError(f"FACodec encoder weights not found: {self.codec_encoder_path}")
        if self.codec_decoder_path is not None and not self.codec_decoder_path.exists():
            raise FileNotFoundError(f"FACodec decoder weights not found: {self.codec_decoder_path}")

    def load_model(self) -> None:
        if self._model is not None and self._codec_encoder is not None and self._codec_decoder is not None:
            return

        self._ensure_dependencies()

        cfg = OmegaConf.load(str(self.cfg_path))
        self._codec_encoder, self._codec_decoder = self._load_codec_models()

        assert self._zact_class is not None
        self.logger.info("[OZSpeech] Loading ZACT model (device=%s)...", self._device_string)

        try:
            from torch.serialization import add_safe_globals

            add_safe_globals(
                [
                    list,
                    int,
                    float,
                    bool,
                    str,
                    dict,
                    tuple,
                    set,
                    ListConfig,
                    DictConfig,
                    ContainerMetadata,
                    Metadata,
                    AnyNode,
                    Any,
                    defaultdict,
                ]
            )
        except Exception:
            pass

        self._model = self._zact_class.from_pretrained(
            cfg=cfg,
            ckpt_path=str(self.checkpoint_path),
            device=self._device_string,
            training_mode=False,
        )

    def _ensure_dependencies(self) -> None:
        if self.code_path is not None and str(self.code_path) not in sys.path:
            sys.path.insert(0, str(self.code_path))

        if self._zact_class is not None and self._facodec_encoder_class is not None and self._facodec_decoder_class is not None:
            return

        try:
            zact_module = importlib.import_module("zact")
            facodec_module = importlib.import_module("zact.models.facodec")
        except ModuleNotFoundError as exc:
            hint = (
                "Ensure OZSpeech's code_path is correctly configured and contains the 'zact' package."
            )
            raise ModuleNotFoundError(f"Could not import OZSpeech dependencies: {hint}") from exc

        self._zact_class = getattr(zact_module, "ZACT")
        self._facodec_encoder_class = getattr(facodec_module, "FACodecEncoder")
        self._facodec_decoder_class = getattr(facodec_module, "FACodecDecoder")

    def _load_manifest(self, manifest_path: Path) -> Iterable[_ManifestEntry]:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    raise ValueError(
                        f"Malformed OZSpeech manifest at {manifest_path}:{line_no}: '{line}'"
                    )
                target_stub = parts[0].strip()
                prompt_name = parts[1].strip()
                text = parts[2].strip() if len(parts) > 2 else ""
                if not target_stub:
                    target_stub = f"sample_{line_no:04d}"
                if not prompt_name:
                    raise ValueError(
                        f"Manifest entry missing prompt filename at {manifest_path}:{line_no}"
                    )
                yield _ManifestEntry(target_stub=target_stub, prompt_name=prompt_name, text=text)

    def _load_codec_models(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        if self._codec_encoder is not None and self._codec_decoder is not None:
            return self._codec_encoder, self._codec_decoder

        if self._facodec_encoder_class is None or self._facodec_decoder_class is None:
            raise RuntimeError("FACodec classes not loaded; call _ensure_dependencies first")

        encoder_weights = (
            str(self.codec_encoder_path)
            if self.codec_encoder_path is not None
            else hf_hub_download(
                repo_id="amphion/naturalspeech3_facodec",
                filename="ns3_facodec_encoder.bin",
            )
        )
        decoder_weights = (
            str(self.codec_decoder_path)
            if self.codec_decoder_path is not None
            else hf_hub_download(
                repo_id="amphion/naturalspeech3_facodec",
                filename="ns3_facodec_decoder.bin",
            )
        )

        self.logger.info("[OZSpeech] Loading FACodec weights...")
        codec_encoder = self._facodec_encoder_class(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        codec_decoder = self._facodec_decoder_class(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )

        state_kwargs = {"map_location": "cpu"}
        codec_encoder.load_state_dict(torch.load(encoder_weights, **state_kwargs))
        codec_decoder.load_state_dict(torch.load(decoder_weights, **state_kwargs))

        codec_encoder.eval()
        codec_decoder.eval()

        return codec_encoder, codec_decoder

    def _to_numpy(self, audio) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32, copy=False)
        if torch.is_tensor(audio):
            return audio.detach().cpu().float().numpy()
        if isinstance(audio, list):
            return np.asarray(audio, dtype=np.float32)
        raise TypeError(f"Unsupported audio type from OZSpeech model: {type(audio)!r}")
