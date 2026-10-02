"""MOSS-TTSD generator wrapper for zero-shot cloning."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from rvcbench.models.model import BaseModel


@dataclass
class MossTTSDGeneratorConfig:
    """Configuration required to run the MOSS-TTSD generator."""

    code_path: Path
    model_path: str
    spt_config_path: Path
    spt_checkpoint_path: Path
    system_prompt: str = (
        "You are a speech synthesizer that clones natural, realistic, and human-like audio from reference text."
    )
    torch_dtype: Union[str, torch.dtype] = torch.bfloat16
    attn_implementation: str = "flash_attention_2"
    use_normalize: bool = True
    silence_duration: float = 0.0
    seed: Optional[int] = None
    use_prompt_transcript: bool = False
    reference_asr_model: str = 'base.en'


class MossTTSDGenerator(BaseModel):
    """Thin wrapper around the upstream MOSS-TTSD inference utilities."""

    def __init__(
        self,
        config: MossTTSDGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.model_path),
            device=device,
            logger=logger,
        )
        self.config = config

        self._torch_dtype = self._resolve_dtype(config.torch_dtype)
        self._generation_utils = None
        self._tokenizer = None
        self._model = None
        self._spt = None

        self._validate_paths()
        self.whisper_model = None
        self._checkpoint_native_class = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio: Optional[Path],
        prompt_text: str,
        sample_index: int,
    ) -> Tuple[np.ndarray, int]:
        """Generate audio conditioned on dialogue text and a reference clip."""

        self.ensure_model()
        self._set_seed(sample_index)

        if self.config.use_prompt_transcript:
            original_text = (prompt_text or '').strip()
            if not original_text:
                raise ValueError('MOSS-TTSD requires reference text when use_prompt_transcript=True')
        else:
            assert self.whisper_model is not None
            with torch.no_grad():
                result = self.whisper_model.transcribe(
                    str(prompt_audio),
                    language="en",
                    fp16=self.whisper_model.device.type != "cpu",
                )
            original_text = result.get("text", "").strip()

        item = {"text": f"[S1]{text} "}
        item["prompt_audio_speaker1"] = str(prompt_audio)
        item["prompt_audio_speaker2"] = str(prompt_audio)
        item["prompt_text_speaker1"] = original_text
        item["prompt_text_speaker2"] = original_text

        process_batch = getattr(self._generation_utils, "process_batch")
        generation_model = self._model
        if self._checkpoint_native_class:
            # Legacy batch processing slices off the prompt itself; request the
            # complete sequence from the checkpoint's newer generation API.
            generation_model = _FullSequenceModel(self._model)
        actual_texts, audio_results = process_batch(
            batch_items=[item],
            tokenizer=self._tokenizer,
            model=generation_model,
            spt=self._spt,
            device=self.device,
            system_prompt=self.config.system_prompt,
            start_idx=0,
            use_normalize=bool(self.config.use_normalize),
            silence_duration=float(self.config.silence_duration),
        )

        self.logger.debug(
            "[MOSS-TTSD] Actual text info: %s",
            actual_texts[0] if actual_texts else "<missing>",
        )

        if not audio_results or audio_results[0] is None:
            raise RuntimeError("MOSS-TTSD failed to generate audio for the provided prompt.")

        audio_data = audio_results[0]["audio_data"]
        sample_rate = int(audio_results[0]["sample_rate"])

        if isinstance(audio_data, torch.Tensor):
            wav = audio_data.detach().cpu().numpy()
        else:
            wav = np.asarray(audio_data)

        wav = np.atleast_1d(wav).astype(np.float32)
        if wav.ndim > 1:
            wav = wav.reshape(-1)
        return wav, sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"MOSS-TTSD code path not found: {self.config.code_path}")
        if not self.config.spt_config_path.exists():
            raise FileNotFoundError(
                f"MOSS-TTSD tokenizer config path not found: {self.config.spt_config_path}"
            )
        if not self.config.spt_checkpoint_path.exists():
            raise FileNotFoundError(
                f"MOSS-TTSD tokenizer checkpoint not found: {self.config.spt_checkpoint_path}"
            )

    def _ensure_imports(self) -> None:
        if self._generation_utils is not None:
            return

        import sys

        code_path_str = str(self.config.code_path)
        if code_path_str not in sys.path:
            sys.path.insert(0, code_path_str)

        xy_path = self.config.code_path / "XY_Tokenizer"
        xy_path_str = str(xy_path)
        if xy_path.exists() and xy_path_str not in sys.path:
            sys.path.insert(0, xy_path_str)

        try:
            import generation_utils  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive
            raise ModuleNotFoundError(
                "Failed to import MOSS-TTSD generation utilities. Ensure 'code_path' points to the "
                "MOSS-TTSD repository."
            ) from exc

        self._generation_utils = generation_utils

    def load_model(self) -> None:
        if self._model is not None and self._spt is not None:
            self._load_reference_asr()
            return

        self._ensure_imports()

        metadata = Path(self.config.model_path) / 'config.json'
        declared = json.loads(metadata.read_text()) if metadata.is_file() else {}
        self._checkpoint_native_class = bool(declared.get('auto_map', {}).get('AutoModel'))
        if self._checkpoint_native_class:
            from transformers import AutoTokenizer, AutoModel, AutoConfig
            _check_native_runtime()
            from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            native_config = AutoConfig.from_pretrained(self.config.model_path, trust_remote_code=True,
                                                       pad_token_id=tokenizer.pad_token_id)
            # Transformers 5 removes generation-only fields during config loading,
            # but this checkpoint implementation still reads this embedding index.
            native_config.pad_token_id = tokenizer.pad_token_id
            model, loading = AutoModel.from_pretrained(
                self.config.model_path, trust_remote_code=True, torch_dtype=self._torch_dtype,
                config=native_config, attn_implementation=self.config.attn_implementation, output_loading_info=True)
            _restore_declared_ties(model, loading)
            _validate_loaded_parameters(model, loading)
            spt = XY_Tokenizer.load_from_checkpoint(config_path=str(self.config.spt_config_path),
                                                   ckpt_path=str(self.config.spt_checkpoint_path))
        else:
            load_model_fn = getattr(self._generation_utils, "load_model")
            tokenizer, model, spt = load_model_fn(
                self.config.model_path,
                str(self.config.spt_config_path),
                str(self.config.spt_checkpoint_path),
                torch_dtype=self._torch_dtype,
                attn_implementation=self.config.attn_implementation,
            )

        self._tokenizer = tokenizer
        self._model = model.to(self.device)
        self._spt = spt.to(self.device)
        self._model.eval()
        self._spt.eval()
        self._load_reference_asr()

    def _load_reference_asr(self):
        if not self.config.use_prompt_transcript and self.whisper_model is None:
            import whisper
            self.whisper_model = whisper.load_model(self.config.reference_asr_model, device=self.device)

    def close(self):
        self._model = self._spt = self._tokenizer = self.whisper_model = None
        self._model_ready = False


    def _resolve_dtype(self, dtype: Union[str, torch.dtype]) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        lookup = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        key = str(dtype).lower()
        if key not in lookup:
            raise ValueError(f"Unsupported torch dtype for MOSS-TTSD: {dtype}")
        return lookup[key]

    def _set_seed(self, sample_index: int) -> None:
        if self.config.seed is None:
            return
        seed = int(self.config.seed) + int(sample_index)
        try:
            from accelerate.utils import set_seed

            set_seed(seed)
        except Exception:  # pragma: no cover - best effort fallback
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


class _FullSequenceModel:
    def __init__(self, model):
        self.model = model

    def generate(self, *args, **kwargs):
        kwargs['output_only'] = False
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _validate_loaded_parameters(model, loading):
    ties = getattr(model, '_tied_weights_keys', {})
    missing = []
    for name in loading.get('missing_keys', []):
        source = ties.get(name) if isinstance(ties, dict) else None
        if source and model.get_parameter(name) is model.get_parameter(source):
            continue
        missing.append(name)
    if missing or loading.get('mismatched_keys'):
        raise RuntimeError(f'MOSS-TTSD checkpoint parameters were not loaded: {missing}; '
                           f'mismatches={loading.get("mismatched_keys", [])}')


def _restore_declared_ties(model, loading):
    """Honor explicit checkpoint aliases on older Transformers loaders."""
    ties = getattr(model, '_tied_weights_keys', {})
    if not isinstance(ties, dict):
        return
    missing = set(loading.get('missing_keys', []))
    for target, source in ties.items():
        if target not in missing or source in missing:
            continue
        original, loaded = model.get_parameter(target), model.get_parameter(source)
        if original.shape != loaded.shape:
            raise RuntimeError(f'MOSS-TTSD declared weight alias has incompatible shapes: {target} -> {source}')
        parent, name = target.rsplit('.', 1)
        setattr(model.get_submodule(parent), name, loaded)


def _check_native_runtime():
    """Check the APIs used by the configured native MOSS-TTSD checkpoint."""
    import inspect
    from transformers import PreTrainedModel
    from transformers.generation.utils import GenerationMixin
    tie = inspect.signature(PreTrainedModel.tie_weights).parameters
    cache = getattr(GenerationMixin, '_get_initial_cache_position', None)
    if not {'missing_keys', 'recompute_mapping'} <= tie.keys() or cache is None:
        raise RuntimeError('Native MOSS-TTSD checkpoint requires compatible Transformers tie_weights '
                           'and generation cache APIs; the validated runtime uses transformers==5.0.0 '
                           '(see envs/moss-ttsd.yml).')
