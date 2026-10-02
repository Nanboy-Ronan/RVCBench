"""ZONOS2 generator wrapper (Zyphra/ZONOS2)."""
from __future__ import annotations

import contextlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from rvcbench.models.model import BaseModel


@dataclass
class Zonos2GeneratorConfig:
    """Configuration required to run the ZONOS2 generator."""

    checkpoint: str = "Zyphra/ZONOS2"
    seed: Optional[int] = 42
    native_seed_policy: str = 'source_index'
    clean_speaker_background: bool = False
    accurate_mode: bool = True
    max_tokens: Optional[int] = None
    code_path: Optional[str] = None
    memory_ratio: float = 0.9
    max_running_req: int = 256
    distributed_port: Optional[int] = None
    vocoder_path: Optional[str] = None
    speaker_file_path: Optional[str] = None


class Zonos2Generator(BaseModel):
    """Thin wrapper around the upstream `zonos2` offline `TTSLLM` API.

    The single-rank engine binds logical cuda:0. Physical GPU placement is
    selected with CUDA_VISIBLE_DEVICES before starting this dedicated process.
    """

    def __init__(
        self,
        config: Zonos2GeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.checkpoint),
            device=device,
            logger=logger,
        )
        self.config = config
        self._tts = None
        self._sampling_params_cls = None
        self._vocoder_module = None
        self._owned_dac = None
        self._owns_group = False
        self.last_native_seed = None

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if self._tts is not None:
            return

        device = torch.device(self.device)
        if device.type != 'cuda' or device.index not in (None, 0):
            raise ValueError('ZONOS2 requires logical cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES')
        if torch.distributed.is_initialized():
            raise RuntimeError('ZONOS2 requires its own process group; use a dedicated process')
        if not 0 < self.config.memory_ratio <= 1 or self.config.max_running_req < 1:
            raise ValueError('ZONOS2 memory_ratio must be in (0, 1] and max_running_req must be positive')
        if self.config.native_seed_policy not in ('source_index', 'legacy_fixed'):
            raise ValueError('ZONOS2 native_seed_policy must be source_index or legacy_fixed')
        if self.config.code_path:
            root = Path(self.config.code_path).expanduser().resolve()
            import_root = root / 'python' if (root / 'python').is_dir() else root
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))

        try:
            from zonos2.message import TTSSamplingParams
            from zonos2.tts import TTSLLM
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'zonos2'. Clone Zyphra/ZONOS2 and run `uv sync`."
            ) from exc

        module = importlib.import_module('zonos2.tts.llm')
        if self.config.code_path and not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('ZONOS2 runtime was imported from outside configured code_path')
        self._vocoder_module = importlib.import_module('zonos2.tokenizer.vocoder')
        if self._vocoder_module._dac_model is not None:
            raise RuntimeError('ZONOS2 requires an empty DAC cache in its dedicated process')

        self._sampling_params_cls = TTSSamplingParams
        self._owns_group = True  # Preflight confirmed no existing group.
        try:
            with self._runtime_context(), self._scheduler_config(module):
                self._tts = TTSLLM(model_path=self.config.checkpoint,
                                   memory_ratio=float(self.config.memory_ratio),
                                   max_running_req=int(self.config.max_running_req))
                if self.config.vocoder_path:
                    import dac
                    self._owned_dac = dac.DAC.load(self.config.vocoder_path).eval().to(device)
                    self._vocoder_module._dac_model = self._owned_dac
                if self.config.speaker_file_path:
                    from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding
                    class LocalSpeakerEmbedding(Qwen3SpeakerEmbedding):
                        MODEL_NAME = str(self.config.speaker_file_path)
                    self._tts._speaker_embedder = LocalSpeakerEmbedding(device='cpu')
        except BaseException:
            try:
                self.close()
            except Exception:
                self.logger.exception('ZONOS2 cleanup failed during initialization')
            raise
        self.model = getattr(self._tts, "model", None)
        self.logger.info("[ZONOS2] Loaded model from %s", self.config.checkpoint)
        self.logger.info('[ZONOS2] Engine device=%s; native seed policy=%s',
                         getattr(getattr(self._tts, 'engine', None), 'device', None), self.config.native_seed_policy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio,
        prompt_text: Optional[str],
        sample_index: int,
        language: str = "en_us",
    ) -> Tuple[np.ndarray, int]:
        """Generate an utterance conditioned on a prompt clip and text."""

        del prompt_text  # ZONOS2 conditions on reference audio only.

        self.ensure_model()
        assert self._tts is not None
        assert self._sampling_params_cls is not None

        offset = int(sample_index) if self.config.native_seed_policy == 'source_index' else 0
        seed = None if self.config.seed is None else int(self.config.seed) + offset
        self.last_native_seed = seed
        sampling_params = self._sampling_params_cls(seed=seed)
        with self._runtime_context():
            speaker_embedding = self._tts.embed_speaker_file(str(prompt_audio))
            try:
                result = self._tts.generate_one(
                    text,
                    sampling_params,
                    language=language,
                    speaker_embedding=speaker_embedding,
                    clean_speaker_background=self.config.clean_speaker_background,
                    accurate_mode=self.config.accurate_mode,
                    max_tokens=self.config.max_tokens,
                )
            finally:
                self._owned_dac = self._vocoder_module._dac_model

        # result["audio"] is raw float32 PCM bytes (see TTSLLM.save_audio).
        audio_bytes = result["audio"]
        wav_np = np.frombuffer(audio_bytes, dtype=np.float32).copy()
        wav_np = np.atleast_1d(wav_np).astype(np.float32).flatten()
        if not wav_np.size or not np.isfinite(wav_np).all():
            raise RuntimeError('ZONOS2 returned empty or nonfinite audio')

        sample_rate = int(result.get("sample_rate", 44100))
        return wav_np, sample_rate

    @contextlib.contextmanager
    def _runtime_context(self):
        with torch.cuda.device(self.device):
            stream = getattr(self._tts, 'stream', None) if self._tts is not None else torch.cuda.current_stream()
            with torch.cuda.stream(stream):
                yield

    @contextlib.contextmanager
    def _scheduler_config(self, module):
        port = self.config.distributed_port
        if port is None:
            yield
            return
        if not 1 <= int(port) <= 65535:
            raise ValueError('ZONOS2 distributed_port must be in [1, 65535]')
        original = module.SchedulerConfig
        class LocalSchedulerConfig(original):
            @property
            def distributed_addr(self):
                return f'tcp://127.0.0.1:{int(port)}'
        module.SchedulerConfig = LocalSchedulerConfig
        try:
            yield
        finally:
            module.SchedulerConfig = original

    def close(self):
        try:
            if self._tts is not None:
                with self._runtime_context():
                    self._tts.shutdown()
        finally:
            self._tts = self.model = None
            if self._vocoder_module is not None and self._vocoder_module._dac_model is self._owned_dac:
                self._vocoder_module._dac_model = None
            self._owned_dac = self._vocoder_module = self._sampling_params_cls = None
            self._model_ready = False
            if self._owns_group and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            self._owns_group = False
            self.last_native_seed = None
            self.logger.info('[ZONOS2] Closed; process_group_initialized=%s; owned DAC cache released',
                             torch.distributed.is_initialized())
