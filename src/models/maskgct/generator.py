"""MaskGCT generator wrapper backed by a persistent subprocess worker."""

from __future__ import annotations

import atexit
import json
import math
import uuid
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

from src.models.model import BaseModel
from src.models.worker_protocol import WorkerResponseReader


@dataclass
class MaskGCTGeneratorConfig:
    code_path: Path
    config_path: Path
    runtime_python: Optional[Path] = None
    worker_script_path: Optional[Path] = None
    repo_id: str = "amphion/MaskGCT"
    checkpoint_dir: Optional[Path] = None
    semantic_model_path: Optional[Path] = None
    espeak_runtime: Optional[Dict[str, Any]] = None
    startup_timeout_sec: float = 600.0
    request_timeout_sec: float = 900.0
    verbose: bool = False
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


class MaskGCTGenerator(BaseModel):
    """Runs Amphion MaskGCT in an isolated Python environment."""

    def __init__(self, config: MaskGCTGeneratorConfig, device: torch.device, logger) -> None:
        materialised = replace(
            config,
            espeak_runtime=self._to_jsonable(config.espeak_runtime),
            code_path=Path(config.code_path).expanduser().resolve(),
            config_path=Path(config.config_path).expanduser().resolve(),
            semantic_model_path=Path(config.semantic_model_path).expanduser().absolute() if config.semantic_model_path is not None else None,
            checkpoint_dir=Path(config.checkpoint_dir).expanduser().absolute() if config.checkpoint_dir is not None else None,
            runtime_python=(
                Path(config.runtime_python).expanduser().absolute()
                if config.runtime_python is not None
                else None
            ),
            worker_script_path=(
                Path(config.worker_script_path).expanduser().resolve()
                if config.worker_script_path is not None
                else None
            ),
        )
        super().__init__(
            model_name_or_path=str(materialised.code_path),
            device=device,
            logger=logger,
        )
        self.config = materialised
        self._process: Optional[subprocess.Popen[str]] = None
        self._stderr_handle = None
        self._stderr_path: Optional[Path] = None
        self.sample_rate: Optional[int] = None
        self.parameter_count: Optional[int] = None
        self.espeak_runtime = None
        self.last_native_seed = self.last_native_requested_seed = None
        self._response_reader = WorkerResponseReader('MaskGCT')
        for name in ('startup_timeout_sec', 'request_timeout_sec'):
            value = float(getattr(self.config, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'MaskGCT {name} must be finite and positive')

        self._validate_paths()
        atexit.register(self.close)

    def load_model(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        from .assets import checkpoint_files, semantic_files
        if self.config.checkpoint_dir is None:
            raise ValueError('MaskGCT requires an explicit checkpoint_dir; resolve model assets before preparing the worker')
        checkpoint_files(self.config.checkpoint_dir)
        if self.config.semantic_model_path is None:
            raise ValueError("MaskGCT requires an explicit semantic_model_path; resolve model assets before preparing the worker")
        semantic_files(self.config.semantic_model_path)
        runtime_python = str(self.config.runtime_python or Path(sys.executable))
        worker_script = str(self.config.worker_script_path)
        generation_json = json.dumps(self._to_jsonable(self.config.generation_kwargs or {}))

        command = [
            runtime_python,
            worker_script,
            "--code-path",
            str(self.config.code_path),
            "--config-path",
            str(self.config.config_path),
            "--device",
            self._device_string(),
            "--checkpoint-dir",
            str(self.config.checkpoint_dir),
            "--semantic-model-path",
            str(self.config.semantic_model_path),
            "--generation-kwargs-json",
            generation_json,
        ]
        if self.config.espeak_runtime is not None:
            command.extend(['--expected-espeak-json', json.dumps(self.config.espeak_runtime)])
        if self.config.verbose:
            command.append("--verbose")

        self._response_reader.clear()
        atexit.register(self.close)
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._open_stderr_log(), env=self._build_worker_env(),
                text=True, bufsize=1,
            )
            message = self._read_message(expect_event="ready", timeout_sec=self.config.startup_timeout_sec)
            if message.get("ok") is not True:
                raise RuntimeError(f"MaskGCT worker failed to start: {message.get('error', 'unknown error')}")
            reported = message.get('espeak_runtime')
            if self.config.espeak_runtime is not None and json.dumps(reported, sort_keys=True) != json.dumps(self.config.espeak_runtime, sort_keys=True):
                raise RuntimeError('MaskGCT worker eSpeak resources differ from the resolved asset fingerprints')
            self.espeak_runtime = reported
            reported_params = message.get("parameter_count")
            if isinstance(reported_params, int) and reported_params > 0:
                self.parameter_count = reported_params
        except BaseException:
            self.close()
            raise

    def generate(
        self,
        *,
        prompt_speech_path: Path,
        prompt_text: str,
        target_text: str,
        output_path: Path,
        seed: Optional[int] = None,
        prompt_language: str = "en",
        target_language: str = "en",
        target_len: Optional[float] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ) -> Path:
        self.ensure_model()
        request = {
            "action": "generate",
            "request_id": uuid.uuid4().hex,
            "seed": seed,
            "prompt_speech_path": str(Path(prompt_speech_path).expanduser().resolve()),
            "prompt_text": str(prompt_text or "").strip(),
            "target_text": str(target_text or "").strip(),
            "output_path": str(Path(output_path).expanduser().resolve()),
            "prompt_language": str(prompt_language or "en").strip(),
            "target_language": str(target_language or prompt_language or "en").strip(),
            "target_len": target_len,
            "generation_overrides": generation_overrides or {},
        }
        self.last_native_seed = self.last_native_requested_seed = None
        if not request['target_text']:
            raise ValueError('MaskGCT target text cannot be empty')
        if not Path(request['prompt_speech_path']).is_file():
            raise FileNotFoundError(f"Missing MaskGCT reference: {request['prompt_speech_path']}")
        if seed is not None and type(seed) is not int:
            raise ValueError('MaskGCT seed must be an integer or None')
        dest = Path(request['output_path'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        try:
            self._send_message(request)
            response = self._read_message(expect_event="response", timeout_sec=self.config.request_timeout_sec)
            if response.get('request_id') != request['request_id']:
                raise RuntimeError('MaskGCT worker response does not match the request')
            if response.get('ok') is not True:
                raise RuntimeError(response.get('error', 'MaskGCT worker returned an unknown error'))
            if 'seed' not in response or type(response['seed']) is not type(seed) or response['seed'] != seed:
                raise RuntimeError('MaskGCT worker did not acknowledge the requested seed')
            result_path = Path(str(response.get('output_path') or '')).resolve()
            if result_path != dest:
                raise RuntimeError('MaskGCT worker returned an unexpected output path')
            if not result_path.is_file():
                raise FileNotFoundError(f'MaskGCT worker reported success but no audio was written: {result_path}')
            audio, rate = sf.read(str(result_path), dtype='float32', always_2d=False)
            if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or rate <= 0:
                raise RuntimeError('MaskGCT worker output must be nonempty finite mono audio')
            self.sample_rate = int(rate)
            self.last_native_requested_seed = self.last_native_seed = seed
            return result_path
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        atexit.unregister(self.close)
        process = self._process
        self._process = None
        self._model_ready = False
        self.model = None
        self.sample_rate = self.parameter_count = None
        self.espeak_runtime = None
        self.last_native_seed = self.last_native_requested_seed = None
        self._response_reader = WorkerResponseReader('MaskGCT')
        if process is None:
            self._close_stderr_log()
            return
        try:
            if process.poll() is None and process.stdin is not None:
                self._send_message({"action": "close"}, process=process)
        except Exception:
            pass
        try:
            process.communicate(timeout=5)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                pass
        self._close_stderr_log()

    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"MaskGCT code_path not found: {self.config.code_path}")
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"MaskGCT config_path not found: {self.config.config_path}")
        if self.config.runtime_python is not None and not self.config.runtime_python.exists():
            raise FileNotFoundError(f"MaskGCT runtime_python not found: {self.config.runtime_python}")
        if self.config.worker_script_path is None:
            raise FileNotFoundError("MaskGCT worker_script_path was not provided.")
        if not self.config.worker_script_path.exists():
            raise FileNotFoundError(f"MaskGCT worker script not found: {self.config.worker_script_path}")

    def _device_string(self) -> str:
        device = self.device
        if isinstance(device, torch.device):
            if device.index is None:
                return device.type
            return f"{device.type}:{device.index}"
        return str(device)

    def _open_stderr_log(self):
        self._close_stderr_log()
        handle = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="maskgct_worker_",
            suffix=".log",
            delete=False,
        )
        self._stderr_handle = handle
        self._stderr_path = Path(handle.name)
        return handle

    def _build_worker_env(self) -> Dict[str, str]:
        return build_worker_env(self.config.runtime_python or Path(sys.executable))

    def _close_stderr_log(self) -> None:
        handle = self._stderr_handle
        self._stderr_handle = None
        if handle is None:
            return
        try:
            handle.close()
        except Exception:
            pass

    def _read_stderr_tail(self, max_chars: int = 4000) -> str:
        if self._stderr_path is None or not self._stderr_path.exists():
            return ""
        try:
            text = self._stderr_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        return text[-max_chars:].strip()

    def _send_message(self, payload: Dict[str, Any], process: Optional[subprocess.Popen[str]] = None) -> None:
        proc = process or self._process
        if proc is None or proc.stdin is None:
            raise RuntimeError("MaskGCT worker is not running.")
        proc.stdin.write(json.dumps(self._to_jsonable(payload)) + "\n")
        proc.stdin.flush()

    def _to_jsonable(self, value):
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        if hasattr(value, "items") and not isinstance(value, dict):
            try:
                return {str(key): self._to_jsonable(item) for key, item in value.items()}
            except Exception:
                pass
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
            module_name = value.__class__.__module__
            if module_name.startswith("omegaconf"):
                try:
                    return [self._to_jsonable(item) for item in value]
                except Exception:
                    pass
        return value

    def _read_message(self, *, expect_event, timeout_sec):
        return self._response_reader.read(self._process, expect_event=expect_event,
            timeout_sec=timeout_sec, logger=self.logger, stderr_tail=self._read_stderr_tail)


def build_worker_env(runtime_python) -> Dict[str, str]:
    """Use the same native-library search path for asset probes and workers."""
    env = dict(os.environ)

    runtime_root = Path(runtime_python).resolve().parent.parent
    lib_paths = []

    candidate_paths = [
        runtime_root / "lib",
        runtime_root / "lib64",
        runtime_root / "lib" / "python3.10" / "site-packages" / "onnxruntime" / "capi",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cublas" / "lib",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cuda_runtime" / "lib",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cudnn" / "lib",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cufft" / "lib",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "curand" / "lib",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cusolver" / "lib",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cusparse" / "lib",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "cuda_nvrtc" / "lib",
        runtime_root / "lib" / "python3.10" / "site-packages" / "nvidia" / "nvjitlink" / "lib",
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
    ]

    seen = set()
    for path in candidate_paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        token = str(resolved)
        if token in seen or not resolved.exists():
            continue
        seen.add(token)
        lib_paths.append(token)

    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        for token in existing.split(":"):
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                lib_paths.append(token)

    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)
    return env
