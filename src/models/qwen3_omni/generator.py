"""Qwen3-Omni generator wrapper."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch

from src.models.model import BaseModel


DEFAULT_SYSTEM_PROMPT = (
    "You are a virtual voice assistant with no gender or age.\n"
    "You are communicating with the user.\n"
    "In user messages, 'I/me/my/we/our' refer to the user and 'you/your' refer to the assistant.\n"
    "In your replies, address the user as 'you/your' and yourself as 'I/me/my'; never mirror the user's pronouns.\n"
    "Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\n"
    "Interact with users using short (no more than 50 words), brief, straightforward language, maintaining a natural tone.\n"
    "Never use formal phrasing, mechanical expressions, bullet points, or overly structured language.\n"
    "Your output must consist only of the spoken content you want the user to hear.\n"
    "Do not include any descriptions of actions, emotions, sounds, or voice changes.\n"
    "Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions.\n"
    "You must answer users' audio or text questions and should communicate in the same language as the user unless they request otherwise.\n"
    "When uncertain, use brief questions to guide the user to continue the conversation.\n"
    "Keep replies concise and conversational, as if talking face-to-face."
)


@dataclass
class Qwen3OmniGeneratorConfig:
    """Configuration options for running the Qwen3-Omni generator."""

    checkpoint_path: str
    torch_dtype: Optional[str] = "auto"
    device_map: Optional[str] = "auto"
    use_flash_attn2: bool = False
    attn_implementation: Optional[str] = None
    return_audio: bool = True
    use_audio_in_video: bool = True
    thinker_max_new_tokens: int = 2048
    thinker_temperature: float = 0.7
    thinker_top_p: float = 0.8
    thinker_top_k: int = 20
    thinker_do_sample: bool = True
    speaker: Optional[str] = None
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    seed: Optional[int] = None
    sample_rate: Optional[int] = None
    clean_up_tokenization_spaces: bool = False


class Qwen3OmniGenerator(BaseModel):
    """Thin wrapper around the Transformers Qwen3-Omni inference APIs."""

    def __init__(
        self,
        config: Qwen3OmniGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        materialised_config = replace(config)
        super().__init__(
            model_name_or_path=str(materialised_config.checkpoint_path),
            device=device,
            logger=logger,
        )
        self.config = materialised_config

        self._imports_ready = False
        self._model_cls = None
        self._processor_cls = None
        self._process_mm_info: Optional[Callable[..., Tuple[Any, Any, Any]]] = None

        self._model = None
        self._processor = None
        self._sample_rate = self.config.sample_rate
        self.last_generated_text: Optional[str] = None

        self._validate_checkpoint_hint()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, messages: List[dict[str, Any]], sample_index: int = 0) -> Tuple[np.ndarray, int]:
        """Generate speech conditioned on a multi-modal conversation."""

        self.ensure_model()
        assert self._model is not None and self._processor is not None

        self._apply_seed(sample_index)

        messages = list(messages)
        if self.config.system_prompt:
            has_system = any((msg or {}).get("role") == "system" for msg in messages)
            if not has_system:
                system_message = {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": str(self.config.system_prompt),
                        }
                    ],
                }
                messages = [system_message, *messages]

        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if self.config.return_audio:
            if self._process_mm_info is None:
                raise RuntimeError(
                    "qwen-omni-utils is required to supply audio prompts. Install it via "
                    "`pip install qwen-omni-utils` and ensure it is importable."
                )
            audios, images, videos = self._process_mm_info(messages, use_audio_in_video=self.config.use_audio_in_video)
        else:
            audios, images, videos = None, None, None

        processor_kwargs = {
            "text": text,
            "audio": audios,
            "images": images,
            "videos": videos,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.config.use_audio_in_video:
            processor_kwargs["use_audio_in_video"] = True

        inputs = self._processor(**processor_kwargs)

        if hasattr(inputs, "keys"):
            for key in list(inputs.keys()):
                value = inputs[key]
                if hasattr(value, "shape"):
                    self.logger.debug("[Qwen3-Omni] Processor output %s shape %s", key, tuple(value.shape))
        else:
            self.logger.debug("[Qwen3-Omni] Processor returned %s", type(inputs).__name__)

        model_device = getattr(self._model, "device", self.device)
        inputs = inputs.to(model_device)

        target_dtype = getattr(self._model, "dtype", None)
        if target_dtype is not None and hasattr(inputs, "to"):
            try:
                inputs = inputs.to(target_dtype)
            except TypeError:
                # Some BatchFeature instances do not accept dtype; fall back silently.
                pass

        generation_kwargs = dict(
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=int(self.config.thinker_max_new_tokens),
            thinker_do_sample=bool(self.config.thinker_do_sample),
            thinker_temperature=float(self.config.thinker_temperature),
            thinker_top_p=float(self.config.thinker_top_p),
            thinker_top_k=int(self.config.thinker_top_k),
            use_audio_in_video=bool(self.config.use_audio_in_video),
        )
        if self.config.speaker:
            generation_kwargs["speaker"] = str(self.config.speaker)
        if self.config.return_audio:
            generation_kwargs["return_audio"] = True

        text_ids, audio = self._model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        if hasattr(text_ids, "sequences"):
            generated_sequences = text_ids.sequences[:, input_length:]
        else:
            generated_sequences = text_ids[:, input_length:]

        decoded = self._processor.batch_decode(
            generated_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=bool(self.config.clean_up_tokenization_spaces),
        )
        self.last_generated_text = decoded[0] if decoded else ""

        if not self.config.return_audio:
            raise RuntimeError("Qwen3-Omni configured with return_audio=False; audio output is unavailable.")
        if audio is None:
            raise RuntimeError("Qwen3-Omni did not return any audio waveform for the request.")

        audio_tensor = torch.as_tensor(audio).reshape(-1).detach().cpu().float()
        wav = audio_tensor.numpy()
        sample_rate = self._determine_sample_rate()
        return wav, sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_checkpoint_hint(self) -> None:
        raw = str(self.config.checkpoint_path).strip()
        if not raw:
            raise ValueError("Qwen3-Omni checkpoint_path must be provided.")
        path = Path(raw).expanduser()
        if path.exists():
            return
        # Keep behaviour consistent with Transformers by allowing remote identifiers.

    def _determine_sample_rate(self) -> int:
        if self._sample_rate is not None:
            return int(self._sample_rate)
        if self._processor is not None:
            extractor = getattr(self._processor, "feature_extractor", None)
            if extractor is not None:
                sr = getattr(extractor, "sampling_rate", None)
                if sr is not None:
                    self._sample_rate = int(sr)
                    return int(sr)
            audio_config = getattr(self._processor, "audio_config", None)
            if isinstance(audio_config, dict):
                sr = audio_config.get("sampling_rate")
                if sr is not None:
                    self._sample_rate = int(sr)
                    return int(sr)
        self._sample_rate = 24000
        return 24000

    def _apply_seed(self, sample_index: int) -> None:
        if self.config.seed is None:
            return
        base_seed = int(self.config.seed)
        seed = base_seed + int(sample_index)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _ensure_imports(self) -> None:
        if self._imports_ready:
            return

        try:
            self._model_cls = self._safe_import("transformers", "Qwen3OmniMoeForConditionalGeneration")
            self._processor_cls = self._safe_import("transformers", "Qwen3OmniMoeProcessor")
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "transformers does not provide Qwen3-Omni components. Install a version that includes "
                "Qwen3OmniMoe support (e.g., 'pip install git+https://github.com/huggingface/transformers')."
            ) from exc

        try:
            module = importlib.import_module("qwen_omni_utils")
            self._process_mm_info = getattr(module, "process_mm_info")
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'qwen-omni-utils'. Install it via `pip install qwen-omni-utils`."
            ) from exc

        modeling_module = importlib.import_module(
            "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"
        )
        if not getattr(modeling_module, "_awm_rotary_patch", False):
            original_apply_rotary = modeling_module.apply_rotary_pos_emb

            def _patched_apply_rotary_pos_emb(q, k, cos, sin, *, position_ids=None, unsqueeze_dim=1):
                cos_last = cos.shape[-1]
                target = q.shape[-1]
                if cos_last != target:
                    min_dim = min(cos_last, target)
                    if cos_last != min_dim:
                        cos = cos[..., :min_dim]
                        sin = sin[..., :min_dim]
                    if target != min_dim:
                        q = q[..., :min_dim]
                        k = k[..., :min_dim]
                    import logging

                    logging.getLogger("Qwen3-Omni").debug(
                        "Adjusted RoPE dims from q=%s cos=%s to %d",
                        tuple(q.shape),
                        tuple(cos.shape),
                        min_dim,
                    )
                return original_apply_rotary(
                    q,
                    k,
                    cos,
                    sin,
                    position_ids=position_ids,
                    unsqueeze_dim=unsqueeze_dim,
                )

            modeling_module.apply_rotary_pos_emb = _patched_apply_rotary_pos_emb
            modeling_module._awm_rotary_patch = True

        self._imports_ready = True

    def _safe_import(self, module_name: str, symbol: str):
        module = importlib.import_module(module_name)
        value = getattr(module, symbol, None)
        if value is None:
            raise AttributeError(f"Module '{module_name}' does not define '{symbol}'.")
        return value

    def _resolve_checkpoint(self) -> str:
        candidate = Path(str(self.config.checkpoint_path)).expanduser()
        if not candidate.exists():
            return str(self.config.checkpoint_path)

        if self._looks_like_checkpoint_dir(candidate):
            return str(candidate.resolve())

        nested_matches = [
            path for path in candidate.iterdir() if path.is_dir() and self._looks_like_checkpoint_dir(path)
        ]
        if len(nested_matches) == 1:
            resolved = nested_matches[0].resolve()
            self.logger.debug(
                "Resolved Qwen3-Omni checkpoint directory to nested path '%s'", resolved
            )
            return str(resolved)

        return str(candidate.resolve())

    def _looks_like_checkpoint_dir(self, path: Path) -> bool:
        if not path.is_dir():
            return False

        required_files = {"config.json", "generation_config.json", "tokenizer_config.json"}
        existing = {item.name for item in path.iterdir()}
        if not required_files.issubset(existing):
            return False

        if any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
            return True

        if any(path.glob("model-*-of-*.safetensors")):
            return True

        return False

    def load_model(self) -> None:
        self._ensure_imports()
        assert self._model_cls is not None and self._processor_cls is not None

        checkpoint = self._resolve_checkpoint()

        model_kwargs = {}
        if self.config.device_map is not None:
            if self._accelerate_available():
                model_kwargs["device_map"] = self.config.device_map
            else:
                self.logger.warning(
                    "accelerate is not available; ignoring device_map=%s",
                    self.config.device_map,
                )
        if self.config.torch_dtype is not None:
            dtype_value = self._resolve_dtype(self.config.torch_dtype)
            if dtype_value is not None:
                model_kwargs["torch_dtype"] = dtype_value
        if self.config.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.logger.info("[Qwen3-Omni] Loading model from %s", checkpoint)

        config = self._model_cls.config_class.from_pretrained(checkpoint)
        self._repair_nested_configs(config)

        self._model = self._model_cls.from_pretrained(checkpoint, config=config, **model_kwargs)
        self._processor = self._processor_cls.from_pretrained(checkpoint)

        try:
            talker = getattr(self._model, "talker", None)
            if talker is not None and hasattr(talker, "model"):
                rotary = getattr(talker.model, "rotary_emb", None)
                head_dim = None
                if getattr(talker.model, "layers", None):
                    head_dim = getattr(talker.model.layers[0].self_attn, "head_dim", None)
                if rotary is not None and hasattr(rotary, "inv_freq"):
                    self.logger.debug(
                        "[Qwen3-Omni] Talker rotary inv_freq shape %s head_dim %s",
                        tuple(rotary.inv_freq.shape),
                        head_dim,
                    )
            code2wav = getattr(self._model, "code2wav", None)
            if code2wav is not None and hasattr(code2wav, "pre_transformer"):
                rotary = getattr(code2wav.pre_transformer, "rotary_emb", None)
                if rotary is not None and hasattr(rotary, "inv_freq"):
                    self.logger.debug(
                        "[Qwen3-Omni] Code2Wav rotary inv_freq shape %s",
                        tuple(rotary.inv_freq.shape),
                    )
        except Exception as exc:
            self.logger.debug("[Qwen3-Omni] Failed rotary diagnostics: %s", exc)

        self.logger.info("[Qwen3-Omni] Model ready on %s", getattr(self.device, "type", self.device))

    def _accelerate_available(self) -> bool:
        try:
            from transformers.utils import is_accelerate_available
        except ImportError:
            return False
        return bool(is_accelerate_available())

    def _resolve_dtype(self, value: str) -> Optional[torch.dtype]:
        raw = str(value).strip().lower()
        if raw in {"auto", "none", ""}:
            return None
        if raw in {"float32", "fp32"}:
            return torch.float32
        if raw in {"float16", "fp16", "half"}:
            return torch.float16
        if raw in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if raw in {"float", "single"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype '{value}' for Qwen3-Omni generator.")

    def _repair_nested_configs(self, config) -> None:
        """Apply compatibility fixes for incomplete checkpoint configs."""

        code2wav = getattr(config, "code2wav_config", None)
        if code2wav is not None and not hasattr(code2wav, "rope_parameters"):
            rope_theta = getattr(code2wav, "rope_theta", 10000)
            code2wav.rope_parameters = {
                "rope_type": "default",
                "rope_theta": float(rope_theta),
            }
            self.logger.debug("Injected default rope parameters into code2wav config")

        talker = getattr(config, "talker_config", None)
        if talker is not None:
            text_conf = getattr(talker, "text_config", None)
            if text_conf is not None:
                shared_value = getattr(text_conf, "shared_expert_intermediate_size", None)
                if shared_value is None:
                    fallback = getattr(text_conf, "intermediate_size", None)
                    if fallback is not None:
                        setattr(text_conf, "shared_expert_intermediate_size", fallback)
                        self.logger.debug(
                            "Injected shared_expert_intermediate_size=%s into talker text config", fallback
                        )
