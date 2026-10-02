"""MOSS-TTS generator wrapper (OpenMOSS-Team/MOSS-TTS-v1.5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from rvcbench.models.model import BaseModel


@dataclass
class MossTTSGeneratorConfig:
    """Configuration required to run the MOSS-TTS generator."""

    checkpoint: str = "OpenMOSS-Team/MOSS-TTS-v1.5"
    attn_implementation: str = "sdpa"
    max_new_tokens: int = 4096
    language: Optional[str] = None
    codec_path: Optional[str] = None


class MossTTSGenerator(BaseModel):
    """Thin wrapper around the upstream `transformers` MOSS-TTS integration."""

    def __init__(
        self,
        config: MossTTSGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.checkpoint),
            device=device,
            logger=logger,
        )
        self.config = config
        self._processor = None
        self._sample_rate: Optional[int] = None

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if self._processor is not None and self.model is not None:
            return

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'transformers'. Install it with `pip install transformers`."
            ) from exc

        # AutoModel/AutoProcessor with trust_remote_code always place the
        # model on torch.device("cuda") (device index 0) unless the current
        # default CUDA device is changed first, so `device: cuda:N` in the
        # benchmark config would otherwise be silently ignored.
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            # Per the upstream model card: the cuDNN SDPA backend is broken
            # for this model; disable it and keep flash/mem-efficient/math
            # SDPA as fallbacks.
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(
            self.config.checkpoint,
            trust_remote_code=True,
            **({'codec_path': self.config.codec_path} if self.config.codec_path else {}),
        )
        self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(device)

        self.model = AutoModel.from_pretrained(
            self.config.checkpoint,
            trust_remote_code=True,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

        self._sample_rate = int(self._processor.model_config.sampling_rate)
        self.logger.info("[MossTTS] Loaded model from %s", self.config.checkpoint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio,
        prompt_text: Optional[str],
        sample_index: int,
        language: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate an utterance conditioned on a prompt clip and text."""

        del sample_index, prompt_text  # MOSS-TTS conditions on reference audio only.

        self.ensure_model()
        assert self._processor is not None
        device = torch.device(self.device)

        message = self._processor.build_user_message(
            text=text,
            reference=[str(prompt_audio)],
            language=language or self.config.language,
        )
        # processor() expects a batch of *conversations*, each conversation
        # being a list of messages; a single-turn, batch-of-one call is
        # [[message]], not [message].
        batch = self._processor([[message]], mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(self.config.max_new_tokens),
            )

        messages = self._processor.decode(outputs)
        audio = messages[0].audio_codes_list[0]

        if isinstance(audio, torch.Tensor):
            wav_np = audio.detach().float().cpu().numpy()
        else:
            wav_np = np.asarray(audio, dtype=np.float32)

        wav_np = np.atleast_1d(wav_np).astype(np.float32).flatten()
        sample_rate = self._sample_rate
        assert sample_rate is not None
        return wav_np, sample_rate
