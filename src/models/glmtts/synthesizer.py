"""GLM-TTS zero-shot synthesis wrapper."""
from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class GLMTTSSynthesizerConfig:
    code_path: Path
    sample_rate: int = 24000
    use_cache: bool = True
    use_phoneme: bool = False
    sample_method: str = "ras"
    seed: Optional[int] = None
    ckpt_dir: Optional[Path] = None
    frontend_dir: Optional[Path] = None


class GLMTTSSynthesizer(BaseModel):
    """Thin wrapper around GLM-TTS inference utilities."""

    def __init__(
        self,
        config: GLMTTSSynthesizerConfig,
        device: torch.device,
        logger,
    ) -> None:
        materialized = GLMTTSSynthesizerConfig(
            code_path=Path(config.code_path),
            sample_rate=int(config.sample_rate),
            use_cache=bool(config.use_cache),
            use_phoneme=bool(config.use_phoneme),
            sample_method=str(config.sample_method),
            seed=config.seed,
            ckpt_dir=Path(config.ckpt_dir) if config.ckpt_dir else None,
            frontend_dir=Path(config.frontend_dir) if config.frontend_dir else None,
        )
        if materialized.ckpt_dir is None:
            materialized.ckpt_dir = materialized.code_path / "ckpt"
        if materialized.frontend_dir is None:
            materialized.frontend_dir = materialized.code_path / "frontend"

        super().__init__(
            model_name_or_path=str(materialized.code_path),
            device=device,
            logger=logger,
        )
        self.config = materialized

        self._imports_loaded = False
        self._glmtts_module = None
        self._generate_long = None
        self._get_special_token_ids = None

        self._frontend = None
        self._text_frontend = None
        self._speech_tokenizer = None
        self._llm = None
        self._flow = None

        self._validate_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        text: str,
        prompt_audio: Path,
        prompt_text: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        with self._in_code_path():
            self.ensure_model()
            assert self._frontend is not None
            assert self._text_frontend is not None
            assert self._llm is not None
            assert self._flow is not None
            assert self._generate_long is not None

            prompt_path = Path(prompt_audio)
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt audio not found: {prompt_path}")

            prompt_text_value = (prompt_text or "").strip()
            synth_text_value = (text or "").strip()

            if synth_text_value:
                synth_text_value = self._text_frontend.text_normalize(synth_text_value)
            if prompt_text_value:
                prompt_text_value = self._text_frontend.text_normalize(prompt_text_value)
                prompt_text_value = f"{prompt_text_value} "

            prompt_text_token = self._frontend._extract_text_token(prompt_text_value)
            prompt_speech_token = self._frontend._extract_speech_token([str(prompt_path)])
            speech_feat = self._frontend._extract_speech_feat(
                str(prompt_path),
                sample_rate=int(self.config.sample_rate),
            )
            embedding = self._frontend._extract_spk_embedding(str(prompt_path))

            cache_speech_token_list = [prompt_speech_token.squeeze().tolist()]
            flow_prompt_token = torch.tensor(
                cache_speech_token_list, dtype=torch.int32
            ).to(self.device)

            cache = {
                "cache_text": [prompt_text_value],
                "cache_text_token": [prompt_text_token],
                "cache_speech_token": cache_speech_token_list,
                "use_cache": bool(self.config.use_cache),
            }

            seed_value = seed if seed is not None else self.config.seed or 0
            tts_speech, _, _, _ = self._generate_long(
                frontend=self._frontend,
                text_frontend=self._text_frontend,
                llm=self._llm,
                flow=self._flow,
                text_info=["", synth_text_value],
                cache=cache,
                embedding=embedding,
                seed=seed_value,
                sample_method=str(self.config.sample_method),
                flow_prompt_token=flow_prompt_token,
                speech_feat=speech_feat,
                device=torch.device(self.device),
                use_phoneme=bool(self.config.use_phoneme),
            )

            waveform = tts_speech.squeeze().detach().cpu().numpy().astype(np.float32)
            return waveform, int(self.config.sample_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"GLM-TTS code_path not found: {self.config.code_path}")
        if not self.config.ckpt_dir.exists():
            raise FileNotFoundError(f"GLM-TTS checkpoint dir not found: {self.config.ckpt_dir}")
        if not self.config.frontend_dir.exists():
            raise FileNotFoundError(f"GLM-TTS frontend dir not found: {self.config.frontend_dir}")

        required_dirs = (
            "speech_tokenizer",
            "vq32k-phoneme-tokenizer",
            "llm",
            "flow",
        )
        missing = [name for name in required_dirs if not (self.config.ckpt_dir / name).exists()]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(
                f"GLM-TTS checkpoints missing required subdirectories: {missing_list}"
            )

        flow_ckpt = self.config.ckpt_dir / "flow" / "flow.pt"
        flow_cfg = self.config.ckpt_dir / "flow" / "config.yaml"
        if not flow_ckpt.exists() or not flow_cfg.exists():
            raise FileNotFoundError(
                "GLM-TTS flow checkpoint/config missing; expected "
                f"{flow_ckpt} and {flow_cfg}."
            )

        campplus_path = self.config.frontend_dir / "campplus.onnx"
        if not campplus_path.exists():
            raise FileNotFoundError(
                f"GLM-TTS frontend asset missing: {campplus_path}"
            )

    def _ensure_imports(self) -> None:
        if self._imports_loaded:
            return

        code_path = str(self.config.code_path)
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        self._glmtts_module = importlib.import_module("glmtts_inference")
        self._generate_long = getattr(self._glmtts_module, "generate_long", None)
        self._get_special_token_ids = getattr(self._glmtts_module, "get_special_token_ids", None)

        if self._generate_long is None or self._get_special_token_ids is None:
            raise ImportError("glmtts_inference missing required helpers for synthesis.")

        self._imports_loaded = True

    @contextmanager
    def _in_code_path(self):
        current_dir = Path.cwd()
        target_dir = self.config.code_path
        try:
            if current_dir != target_dir:
                os.chdir(target_dir)
            yield
        finally:
            if Path.cwd() != current_dir:
                os.chdir(current_dir)

    def load_model(self) -> None:
        if self._llm is not None:
            return

        self._ensure_imports()

        from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
        from llm.glmtts import GLMTTS
        from transformers import AutoTokenizer, LlamaForCausalLM
        from utils import tts_model_util, yaml_util
        from utils.audio import mel_spectrogram

        device = torch.device(self.device)
        speech_tokenizer_path = self.config.ckpt_dir / "speech_tokenizer"
        speech_model, speech_feature_extractor = yaml_util.load_speech_tokenizer(
            str(speech_tokenizer_path)
        )
        self._speech_tokenizer = SpeechTokenizer(speech_model, speech_feature_extractor)

        if int(self.config.sample_rate) == 32000:
            feat_extractor = partial(
                mel_spectrogram,
                sampling_rate=32000,
                hop_size=640,
                n_fft=2560,
                num_mels=80,
                win_size=2560,
                fmin=0,
                fmax=8000,
                center=False,
            )
        elif int(self.config.sample_rate) == 24000:
            feat_extractor = partial(
                mel_spectrogram,
                sampling_rate=24000,
                hop_size=480,
                n_fft=1920,
                num_mels=80,
                win_size=1920,
                fmin=0,
                fmax=8000,
                center=False,
            )
        else:
            raise ValueError(f"Unsupported sample_rate: {self.config.sample_rate}")

        glm_tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.ckpt_dir / "vq32k-phoneme-tokenizer"),
            trust_remote_code=True,
        )
        tokenize_fn = lambda text: glm_tokenizer.encode(text)

        self._frontend = TTSFrontEnd(
            tokenize_fn,
            self._speech_tokenizer,
            feat_extractor,
            str(self.config.frontend_dir / "campplus.onnx"),
            str(self.config.frontend_dir / "spk2info.pt"),
            device,
        )
        self._text_frontend = TextFrontEnd(bool(self.config.use_phoneme))

        llama_path = self.config.ckpt_dir / "llm"
        glmtts_configs_dir = self.config.code_path / "configs"
        lora_adapter_config = glmtts_configs_dir / "lora_adapter_configV3.1.json"
        spk_prompt_dict_path = glmtts_configs_dir / "spk_prompt_dict.yaml"
        hift_ckpt = self.config.ckpt_dir / "hift" / "hift.pt"
        if hift_ckpt.exists():
            os.environ["GLMTTS_HIFT_CKPT"] = str(hift_ckpt)
        self._llm = GLMTTS(
            llama_cfg_path=str(llama_path / "config.json"),
            mode="PRETRAIN",
            lora_adapter_config=str(lora_adapter_config) if lora_adapter_config.exists() else None,
            spk_prompt_dict_path=str(spk_prompt_dict_path) if spk_prompt_dict_path.exists() else None,
        )
        self._llm.llama = LlamaForCausalLM.from_pretrained(
            str(llama_path),
            torch_dtype=torch.float32,
        ).to(device)
        self._llm.llama_embedding = self._llm.llama.model.embed_tokens

        special_token_ids = self._get_special_token_ids(self._frontend.tokenize_fn)
        self._llm.set_runtime_vars(special_token_ids=special_token_ids)
        self._llm.eval()

        flow_ckpt = self.config.ckpt_dir / "flow" / "flow.pt"
        flow_config = self.config.ckpt_dir / "flow" / "config.yaml"
        flow = yaml_util.load_flow_model(str(flow_ckpt), str(flow_config), device)
        self._flow = tts_model_util.Token2Wav(
            flow,
            sample_rate=int(self.config.sample_rate),
            device=str(device),
        )

        self.model = self._llm
