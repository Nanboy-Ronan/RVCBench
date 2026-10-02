"""ZipVoice inference with retained native models or an explicit CLI runtime."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from src.models.model import BaseModel
from .assets import model_files, vocoder_files


@dataclass
class ZipVoiceGeneratorConfig:
    code_path: Path
    model_name: str = "zipvoice"
    model_dir: Optional[Path] = None
    checkpoint_name: str = "model.pt"
    vocoder_path: Optional[Path] = None
    tokenizer: str = "emilia"
    lang: str = "en-us"
    guidance_scale: Optional[float] = None
    num_step: Optional[int] = None
    feat_scale: float = 0.1
    speed: float = 1.0
    t_shift: float = 0.5
    target_rms: float = 0.1
    raw_evaluation: bool = False
    max_duration: float = 100.0
    remove_long_sil: bool = False
    trt_engine_path: Optional[Path] = None
    num_thread: int = 1
    seed: Optional[int] = None
    runtime_python: Optional[Path] = None
    worker_script_path: Optional[Path] = None
    execution_backend: str = 'auto'
    request_timeout_sec: float = 900.0


class ZipVoiceGenerator(BaseModel):
    """Retains native models; a configured foreign interpreter selects the CLI."""

    def __init__(self, config: ZipVoiceGeneratorConfig, device: torch.device, logger) -> None:
        materialised = replace(
            config,
            code_path=Path(config.code_path).expanduser().resolve(),
            model_dir=Path(config.model_dir).expanduser().resolve() if config.model_dir is not None else None,
            vocoder_path=Path(config.vocoder_path).expanduser().resolve() if config.vocoder_path is not None else None,
            trt_engine_path=(
                Path(config.trt_engine_path).expanduser().resolve()
                if config.trt_engine_path is not None
                else None
            ),
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
            model_name_or_path=str(materialised.model_dir or materialised.model_name),
            device=device,
            logger=logger,
        )
        self.config = materialised
        if materialised.tokenizer not in ('emilia', 'libritts', 'espeak', 'simple'):
            raise ValueError(f'Unsupported ZipVoice tokenizer: {materialised.tokenizer}')
        self.execution_backend = materialised.execution_backend
        if self.execution_backend == 'auto':
            self.execution_backend = 'cli' if materialised.runtime_python else 'native'
        if self.execution_backend not in ('native', 'cli'):
            raise ValueError('ZipVoice execution_backend must be auto, native or cli')
        if (self.execution_backend == 'native' and materialised.runtime_python and
                materialised.runtime_python.absolute() != Path(sys.executable).absolute()):
            raise ValueError('Native ZipVoice must run in the current interpreter; use execution_backend=cli for runtime_python')
        timeout = materialised.request_timeout_sec
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("ZipVoice request_timeout_sec must be finite and positive")
        self._previous_num_threads = None
        self.sample_rate: int = 24000
        self.vocoder = None
        self._validate_paths()

    def load_model(self) -> None:
        if self.config.model_dir is not None:
            model_files(self.config.model_dir, self.config.checkpoint_name)
        if self.config.vocoder_path is not None:
            vocoder_files(self.config.vocoder_path)
        if self.execution_backend == 'cli':
            result = subprocess.run([str(self.config.runtime_python or sys.executable), '-c',
                                     'import zipvoice.bin.infer_zipvoice'], cwd=self.config.code_path,
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or 'ZipVoice runtime preflight failed')
            return
        if isinstance(self.model, torch.nn.Module):
            return

        code_path = str(self.config.code_path)
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        infer_module = importlib.import_module("zipvoice.bin.infer_zipvoice")
        self._generate_sentence = infer_module.generate_sentence
        self._previous_num_threads = torch.get_num_threads()
        torch.set_num_threads(self.config.num_thread)
        checkpoint_module = importlib.import_module("zipvoice.utils.checkpoint")

        hf_hub_download = getattr(infer_module, "hf_hub_download")
        model_dir_map = getattr(infer_module, "MODEL_DIR")
        get_vocoder = getattr(infer_module, "get_vocoder")
        hgrepo = getattr(infer_module, "HUGGINGFACE_REPO")
        ZipVoice = getattr(infer_module, "ZipVoice")
        ZipVoiceDistill = getattr(infer_module, "ZipVoiceDistill")
        EmiliaTokenizer = getattr(infer_module, "EmiliaTokenizer")
        LibriTTSTokenizer = getattr(infer_module, "LibriTTSTokenizer")
        EspeakTokenizer = getattr(infer_module, "EspeakTokenizer")
        SimpleTokenizer = getattr(infer_module, "SimpleTokenizer")
        VocosFbank = getattr(infer_module, "VocosFbank")
        load_checkpoint = getattr(checkpoint_module, "load_checkpoint")
        safetensors_torch = importlib.import_module("safetensors.torch")

        if self.config.model_dir is not None:
            model_dir = self.config.model_dir
            model_ckpt = model_dir / self.config.checkpoint_name
            model_config = model_dir / "model.json"
            token_file = model_dir / "tokens.txt"
        else:
            model_ckpt = Path(
                hf_hub_download(hgrepo, filename=f"{model_dir_map[self.config.model_name]}/{self.config.checkpoint_name}")
            )
            model_config = Path(
                hf_hub_download(hgrepo, filename=f"{model_dir_map[self.config.model_name]}/model.json")
            )
            token_file = Path(
                hf_hub_download(hgrepo, filename=f"{model_dir_map[self.config.model_name]}/tokens.txt")
            )

        if self.config.tokenizer == "emilia":
            tokenizer = EmiliaTokenizer(token_file=str(token_file))
        elif self.config.tokenizer == "libritts":
            tokenizer = LibriTTSTokenizer(token_file=str(token_file))
        elif self.config.tokenizer == "espeak":
            tokenizer = EspeakTokenizer(token_file=str(token_file), lang=self.config.lang)
        else:
            tokenizer = SimpleTokenizer(token_file=str(token_file))

        tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
        with model_config.open("r", encoding="utf-8") as handle:
            model_config_data = json.load(handle)

        model_cls = ZipVoice if self.config.model_name == "zipvoice" else ZipVoiceDistill
        model = model_cls(**model_config_data["model"], **tokenizer_config)

        if str(model_ckpt).endswith(".safetensors"):
            safetensors_torch.load_model(model, str(model_ckpt))
        elif str(model_ckpt).endswith(".pt"):
            load_checkpoint(filename=str(model_ckpt), model=model, strict=True)
        else:
            raise NotImplementedError(f"Unsupported ZipVoice checkpoint format: {model_ckpt}")

        model = model.to(self.device)
        model.eval()
        if self.config.trt_engine_path:
            infer_module.load_trt(model, str(self.config.trt_engine_path))
        self.model = model

        self.vocoder = get_vocoder(str(self.config.vocoder_path) if self.config.vocoder_path else None)
        self.vocoder = self.vocoder.to(self.device).eval()

        feature_extractor = VocosFbank()
        self.sample_rate = int(model_config_data["feature"]["sampling_rate"])
        self._feature_extractor = feature_extractor
        self._tokenizer = tokenizer

    def close(self) -> None:
        self.model = self.vocoder = None
        self._tokenizer = self._feature_extractor = None
        self._model_ready = False
        if self._previous_num_threads is not None:
            torch.set_num_threads(self._previous_num_threads)
            self._previous_num_threads = None

    def generate(
        self,
        *,
        text: str,
        prompt_wav: Path,
        prompt_text: str,
        lang: Optional[str] = None,
        sample_index: int = 0,
    ) -> Tuple[np.ndarray, int]:
        self.ensure_model()

        prompt_path = Path(prompt_wav).expanduser().resolve()
        if not prompt_path.exists():
            raise FileNotFoundError(f"ZipVoice prompt audio not found: {prompt_path}")

        if self.config.raw_evaluation:
            raise NotImplementedError(
                "ZipVoice raw_evaluation is not supported in the benchmark wrapper yet."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = Path(handle.name)

        if self.execution_backend == 'native':
            try:
                if self.config.seed is not None:
                    from src.utils.seeding import configure_seeds
                    configure_seeds(int(self.config.seed) + int(sample_index), logger=None)
                if self.config.tokenizer == 'espeak' and lang and lang != self.config.lang:
                    raise ValueError('Native ZipVoice espeak language must match its prepared tokenizer')
                defaults = (16, 1.0) if self.config.model_name == 'zipvoice' else (8, 3.0)
                with torch.inference_mode():
                    self._generate_sentence(
                        save_path=str(output_path), prompt_text=prompt_text, prompt_wav=str(prompt_path),
                        text=text, model=self.model, vocoder=self.vocoder, tokenizer=self._tokenizer,
                        feature_extractor=self._feature_extractor, device=self.device,
                        num_step=self.config.num_step if self.config.num_step is not None else defaults[0],
                        guidance_scale=self.config.guidance_scale if self.config.guidance_scale is not None else defaults[1],
                        feat_scale=self.config.feat_scale, speed=self.config.speed, t_shift=self.config.t_shift,
                        target_rms=self.config.target_rms, sampling_rate=self.sample_rate,
                        max_duration=self.config.max_duration, remove_long_sil=self.config.remove_long_sil)
                return self._read_output(output_path)
            finally:
                output_path.unlink(missing_ok=True)

        command = [
            str(self.config.runtime_python or Path(sys.executable)),
            "-m",
            "zipvoice.bin.infer_zipvoice",
            "--model-name",
            str(self.config.model_name),
            "--prompt-wav",
            str(prompt_path),
            "--prompt-text",
            str(prompt_text),
            "--text",
            str(text),
            "--res-wav-path",
            str(output_path),
            "--checkpoint-name",
            str(self.config.checkpoint_name),
            "--tokenizer",
            str(self.config.tokenizer),
            "--lang",
            str(lang or self.config.lang),
            "--feat-scale",
            str(float(self.config.feat_scale)),
            "--speed",
            str(float(self.config.speed)),
            "--t-shift",
            str(float(self.config.t_shift)),
            "--target-rms",
            str(float(self.config.target_rms)),
            "--max-duration",
            str(float(self.config.max_duration)),
            "--num-thread",
            str(int(self.config.num_thread)),
        ]
        if self.config.model_dir is not None:
            command.extend(["--model-dir", str(self.config.model_dir)])
        if self.config.vocoder_path is not None:
            command.extend(["--vocoder-path", str(self.config.vocoder_path)])
        if self.config.trt_engine_path is not None:
            command.extend(["--trt-engine-path", str(self.config.trt_engine_path)])
        if self.config.guidance_scale is not None:
            command.extend(["--guidance-scale", str(float(self.config.guidance_scale))])
        if self.config.num_step is not None:
            command.extend(["--num-step", str(int(self.config.num_step))])
        if self.config.seed is not None:
            command.extend(["--seed", str(int(self.config.seed) + int(sample_index))])
        if self.config.remove_long_sil:
            command.append("--remove-long-sil")

        try:
            result = subprocess.run(
                command,
                cwd=str(self.config.code_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=self.config.request_timeout_sec,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "unknown ZipVoice CLI error"
                raise RuntimeError(detail)

            return self._read_output(output_path, clip=True)
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _read_output(path, *, clip=False):
        waveform, sample_rate = sf.read(str(path), dtype='float32', always_2d=False)
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("ZipVoice output must be nonempty finite mono audio")
        return np.clip(audio, -1.0, 1.0) if clip else audio, int(sample_rate)

    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"ZipVoice code_path not found: {self.config.code_path}")
        if self.config.model_dir is not None and not self.config.model_dir.exists():
            raise FileNotFoundError(f"ZipVoice model_dir not found: {self.config.model_dir}")
        if self.config.vocoder_path is not None and not self.config.vocoder_path.exists():
            raise FileNotFoundError(f"ZipVoice vocoder_path not found: {self.config.vocoder_path}")
        if self.config.trt_engine_path is not None and not self.config.trt_engine_path.exists():
            raise FileNotFoundError(f"ZipVoice trt_engine_path not found: {self.config.trt_engine_path}")
        if self.config.runtime_python is not None and not self.config.runtime_python.exists():
            raise FileNotFoundError(f"ZipVoice runtime_python not found: {self.config.runtime_python}")
