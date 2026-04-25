"""IndexTTS generator wrapper backed by a persistent subprocess worker."""

from __future__ import annotations

import atexit
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
import torch

from src.models.model import BaseModel


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
                Path(config.runtime_python).expanduser().resolve()
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

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._open_stderr_log(),
            text=True,
            bufsize=1,
        )
        message = self._read_message(expect_event="ready")
        if not message.get("ok"):
            self.close()
            raise RuntimeError(f"IndexTTS worker failed to start: {message.get('error', 'unknown error')}")
        reported_params = message.get("parameter_count")
        if isinstance(reported_params, int) and reported_params > 0:
            self.parameter_count = reported_params

    def generate(
        self,
        *,
        ref_audio: Path,
        text: str,
        output_path: Path,
        emo_audio_prompt: Optional[Path] = None,
        emo_alpha: float = 1.0,
    ) -> Path:
        self.ensure_model()
        self.last_output_wait_sec = 0.0
        if not text or not str(text).strip():
            raise ValueError("IndexTTS generation text cannot be empty.")

        request = {
            "action": "generate",
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
        self._send_message(request)
        response = self._read_message()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "IndexTTS worker returned an unknown error."))

        result_path = Path(str(response.get("output_path") or request["output_path"])).resolve()
        output_ready, output_wait_sec = self._wait_for_output_file(result_path)
        self.last_output_wait_sec = output_wait_sec
        if not output_ready:
            stderr_tail = self._read_stderr_tail()
            detail = f"IndexTTS worker reported success but no audio was written: {result_path}"
            if stderr_tail:
                detail += f"\nWorker stderr tail:\n{stderr_tail}"
            raise FileNotFoundError(detail)

        if self.sample_rate is None:
            try:
                _audio, sample_rate = sf.read(str(result_path), dtype="float32", always_2d=False)
                self.sample_rate = int(sample_rate)
            except Exception:
                self.sample_rate = 24000
        return result_path

    def close(self) -> None:
        process = self._process
        self._process = None
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

    def _read_message(
        self,
        *,
        expect_event: Optional[str] = None,
        process: Optional[subprocess.Popen[str]] = None,
    ) -> Dict[str, Any]:
        proc = process or self._process
        if proc is None or proc.stdout is None:
            raise RuntimeError("IndexTTS worker is not running.")

        while True:
            line = proc.stdout.readline()
            if line:
                payload = line.strip()
                if not payload:
                    continue
                try:
                    message = json.loads(payload)
                except json.JSONDecodeError:
                    if self.logger:
                        self.logger.debug("[IndexTTS] Ignoring non-JSON worker stdout: %s", payload[:500])
                    continue
                event = message.get("event")
                if event == "startup_error":
                    return message
                if expect_event is None or event == expect_event:
                    return message
                continue

            returncode = proc.poll()
            stderr_text = self._read_stderr_tail()
            if returncode is not None:
                raise RuntimeError(
                    f"IndexTTS worker exited unexpectedly with code {returncode}. {stderr_text}".strip()
                )
