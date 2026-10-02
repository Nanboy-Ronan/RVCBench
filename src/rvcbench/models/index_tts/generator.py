"""IndexTTS generator wrapper backed by a persistent subprocess worker."""

from __future__ import annotations

import atexit
import json
import math
import uuid
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

from rvcbench.models.model import BaseModel
from rvcbench.models.worker_protocol import WorkerResponseReader


@dataclass
class IndexTTSGeneratorConfig:
    code_path: Path
    model_dir: Path
    config_path: Path
    runtime_python: Optional[Path] = None
    worker_script_path: Optional[Path] = None
    use_fp16: bool = False
    use_cuda_kernel: bool = False
    use_deepspeed: bool = False
    use_accel: bool = False
    use_torch_compile: bool = False
    interval_silence: int = 200
    max_text_tokens_per_segment: int = 120
    startup_timeout_sec: float = 600.0
    request_timeout_sec: float = 900.0
    output_wait_timeout_sec: float = 5.0
    output_wait_poll_interval_sec: float = 0.1
    verbose: bool = False
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


class IndexTTSGenerator(BaseModel):
    """Runs IndexTTS in an isolated Python environment via a long-lived worker."""

    def __init__(self, config: IndexTTSGeneratorConfig, device: torch.device, logger) -> None:
        materialised = replace(
            config,
            code_path=Path(config.code_path).expanduser().resolve(),
            model_dir=Path(config.model_dir).expanduser().resolve(),
            config_path=Path(config.config_path).expanduser().resolve(),
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
            model_name_or_path=str(materialised.model_dir),
            device=device,
            logger=logger,
        )
        self.config = materialised
        self._process: Optional[subprocess.Popen[str]] = None
        self._stderr_handle = None
        self._stderr_path: Optional[Path] = None
        self.sample_rate: Optional[int] = None
        self.last_output_wait_sec: float = 0.0
        self.parameter_count: Optional[int] = None
        self.last_native_seed = None
        self.last_native_requested_seed = None
        self._response_reader = WorkerResponseReader('IndexTTS')
        for name in ('startup_timeout_sec', 'request_timeout_sec'):
            value = float(getattr(self.config, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'IndexTTS {name} must be finite and positive')

        self._validate_paths()
        atexit.register(self.close)

    def load_model(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        device_arg = self._device_string()
        runtime_python = str(self.config.runtime_python or Path(sys.executable))
        worker_script = str(self.config.worker_script_path)
        generation_json = json.dumps(self.config.generation_kwargs or {})

        command = [
            runtime_python,
            worker_script,
            "--code-path",
            str(self.config.code_path),
            "--model-dir",
            str(self.config.model_dir),
            "--config-path",
            str(self.config.config_path),
            "--device",
            device_arg,
            "--interval-silence",
            str(int(self.config.interval_silence)),
            "--max-text-tokens-per-segment",
            str(int(self.config.max_text_tokens_per_segment)),
            "--generation-kwargs-json",
            generation_json,
        ]
        if self.config.use_fp16:
            command.append("--use-fp16")
        if self.config.use_cuda_kernel:
            command.append("--use-cuda-kernel")
        if self.config.use_deepspeed:
            command.append("--use-deepspeed")
        if self.config.use_accel:
            command.append("--use-accel")
        if self.config.use_torch_compile:
            command.append("--use-torch-compile")
        if self.config.verbose:
            command.append("--verbose")

        self._response_reader.clear()
        atexit.register(self.close)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._open_stderr_log(),
                text=True,
                bufsize=1,
            )
            message = self._read_message(expect_event="ready", timeout_sec=self.config.startup_timeout_sec)
            if message.get("ok") is not True:
                raise RuntimeError(f"IndexTTS worker failed to start: {message.get('error', 'unknown error')}")
            reported_params = message.get("parameter_count")
            if isinstance(reported_params, int) and reported_params > 0:
                self.parameter_count = reported_params
        except BaseException:
            self.close()
            raise

    def generate(
        self,
        *,
        ref_audio: Path,
        text: str,
        output_path: Path,
        seed: Optional[int] = None,
        emo_audio_prompt: Optional[Path] = None,
        emo_alpha: float = 1.0,
    ) -> Path:
        self.ensure_model()
        self.last_output_wait_sec = 0.0
        if not text or not str(text).strip():
            raise ValueError("IndexTTS generation text cannot be empty.")

        request = {
            "action": "generate",
            "request_id": uuid.uuid4().hex,
            "seed": seed,
            "ref_audio": str(Path(ref_audio).expanduser().resolve()),
            "text": str(text).strip(),
            "output_path": str(Path(output_path).expanduser().resolve()),
            "emo_audio_prompt": (
                str(Path(emo_audio_prompt).expanduser().resolve())
                if emo_audio_prompt is not None
                else None
            ),
            "emo_alpha": float(emo_alpha),
        }
        self.last_native_seed = self.last_native_requested_seed = None
        if not Path(request['ref_audio']).is_file():
            raise FileNotFoundError(f"Missing IndexTTS reference: {request['ref_audio']}")
        if seed is not None and type(seed) is not int:
            raise ValueError('IndexTTS seed must be an integer or None')
        dest = Path(request['output_path'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        try:
            self._send_message(request)
            response = self._read_message(expect_event="response", timeout_sec=self.config.request_timeout_sec)
            if response.get("request_id") != request["request_id"]:
                raise RuntimeError('IndexTTS worker response does not match the request')
            if response.get("ok") is not True:
                raise RuntimeError(response.get("error", "IndexTTS worker returned an unknown error."))
            if 'seed' not in response or type(response['seed']) is not type(seed) or response['seed'] != seed:
                raise RuntimeError('IndexTTS worker did not acknowledge the requested seed')
            result_path = Path(str(response.get("output_path") or '')).resolve()
            if result_path != Path(request['output_path']):
                raise RuntimeError('IndexTTS worker returned an unexpected output path')
            output_ready, output_wait_sec = self._wait_for_output_file(result_path)
            self.last_output_wait_sec = output_wait_sec
            if not output_ready:
                raise FileNotFoundError(f"IndexTTS worker reported success but no audio was written: {result_path}")
            audio, sample_rate = sf.read(str(result_path), dtype="float32", always_2d=False)
            if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or sample_rate <= 0:
                raise RuntimeError('IndexTTS worker output must be nonempty finite mono audio')
            self.sample_rate = int(sample_rate)
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
        self.last_native_seed = self.last_native_requested_seed = None
        self._response_reader = WorkerResponseReader('IndexTTS')
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
            raise FileNotFoundError(f"IndexTTS code_path not found: {self.config.code_path}")
        if not self.config.model_dir.exists():
            raise FileNotFoundError(f"IndexTTS model_dir not found: {self.config.model_dir}")
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"IndexTTS config_path not found: {self.config.config_path}")
        if self.config.runtime_python is not None and not self.config.runtime_python.exists():
            raise FileNotFoundError(f"IndexTTS runtime_python not found: {self.config.runtime_python}")
        if self.config.worker_script_path is None:
            raise FileNotFoundError("IndexTTS worker_script_path was not provided.")
        if not self.config.worker_script_path.exists():
            raise FileNotFoundError(f"IndexTTS worker script not found: {self.config.worker_script_path}")

    def _open_stderr_log(self):
        self._close_stderr_log()
        handle = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="indextts_worker_",
            suffix=".log",
            delete=False,
        )
        self._stderr_handle = handle
        self._stderr_path = Path(handle.name)
        return handle

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

    def _wait_for_output_file(self, result_path: Path) -> tuple:
        timeout = max(0.0, float(self.config.output_wait_timeout_sec))
        poll_interval = max(0.01, float(self.config.output_wait_poll_interval_sec))
        start = time.monotonic()
        deadline = start + timeout

        while True:
            try:
                if result_path.exists() and result_path.stat().st_size > 0:
                    return True, max(0.0, time.monotonic() - start)
            except FileNotFoundError:
                pass
            except OSError:
                pass

            if time.monotonic() >= deadline:
                return False, max(0.0, time.monotonic() - start)
            time.sleep(poll_interval)

    def _device_string(self) -> str:
        device = self.device
        if isinstance(device, torch.device):
            if device.index is None:
                return device.type
            return f"{device.type}:{device.index}"
        return str(device)

    def _send_message(self, payload: Dict[str, Any], process: Optional[subprocess.Popen[str]] = None) -> None:
        proc = process or self._process
        if proc is None or proc.stdin is None:
            raise RuntimeError("IndexTTS worker is not running.")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def _read_message(self, *, expect_event, timeout_sec):
        return self._response_reader.read(self._process, expect_event=expect_event,
            timeout_sec=timeout_sec, logger=self.logger, stderr_tail=self._read_stderr_tail)
