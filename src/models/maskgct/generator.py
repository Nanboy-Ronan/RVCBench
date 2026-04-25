"""MaskGCT generator wrapper backed by a persistent subprocess worker."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
import torch

from src.models.model import BaseModel


@dataclass
class MaskGCTGeneratorConfig:
    code_path: Path
    config_path: Path
    runtime_python: Optional[Path] = None
    worker_script_path: Optional[Path] = None
    repo_id: str = "amphion/MaskGCT"
    verbose: bool = False
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


class MaskGCTGenerator(BaseModel):
    """Runs Amphion MaskGCT in an isolated Python environment."""

    def __init__(self, config: MaskGCTGeneratorConfig, device: torch.device, logger) -> None:
        materialised = replace(
            config,
            code_path=Path(config.code_path).expanduser().resolve(),
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

        self._validate_paths()
        atexit.register(self.close)

    def load_model(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

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
            "--repo-id",
            str(self.config.repo_id),
            "--generation-kwargs-json",
            generation_json,
        ]
        if self.config.verbose:
            command.append("--verbose")

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._open_stderr_log(),
            env=self._build_worker_env(),
            text=True,
            bufsize=1,
        )
        message = self._read_message(expect_event="ready")
        if not message.get("ok"):
            self.close()
            raise RuntimeError(f"MaskGCT worker failed to start: {message.get('error', 'unknown error')}")
        reported_params = message.get("parameter_count")
        if isinstance(reported_params, int) and reported_params > 0:
            self.parameter_count = reported_params

    def generate(
        self,
        *,
        prompt_speech_path: Path,
        prompt_text: str,
        target_text: str,
        output_path: Path,
        prompt_language: str = "en",
        target_language: str = "en",
        target_len: Optional[float] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ) -> Path:
        self.ensure_model()
        request = {
            "action": "generate",
            "prompt_speech_path": str(Path(prompt_speech_path).expanduser().resolve()),
            "prompt_text": str(prompt_text or "").strip(),
            "target_text": str(target_text or "").strip(),
            "output_path": str(Path(output_path).expanduser().resolve()),
            "prompt_language": str(prompt_language or "en").strip(),
            "target_language": str(target_language or prompt_language or "en").strip(),
            "target_len": target_len,
            "generation_overrides": generation_overrides or {},
        }
        self._send_message(request)
        response = self._read_message()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "MaskGCT worker returned an unknown error."))

        result_path = Path(str(response.get("output_path") or request["output_path"])).resolve()
        if not result_path.exists():
            raise FileNotFoundError(f"MaskGCT worker reported success but no audio was written: {result_path}")

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
        env = dict(os.environ)

        runtime_python = self.config.runtime_python or Path(sys.executable)
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

    def _read_message(
        self,
        *,
        expect_event: Optional[str] = None,
        process: Optional[subprocess.Popen[str]] = None,
    ) -> Dict[str, Any]:
        proc = process or self._process
        if proc is None or proc.stdout is None:
            raise RuntimeError("MaskGCT worker is not running.")

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
                        self.logger.debug("[MaskGCT] Ignoring non-JSON worker stdout: %s", payload[:500])
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
                    f"MaskGCT worker exited unexpectedly with code {returncode}. {stderr_text}".strip()
                )
