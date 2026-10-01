#!/usr/bin/env python3
"""Persistent JSONL worker process for IndexTTS2 inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _resolve_device(requested: str | None) -> str:
    if requested:
        return str(requested)
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _maybe_count_parameters(model: Any) -> int | None:
    try:
        if model is None or not hasattr(model, "parameters"):
            return None
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interval-silence", type=int, default=200)
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--generation-kwargs-json", default="{}")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--use-cuda-kernel", action="store_true")
    parser.add_argument("--use-deepspeed", action="store_true")
    parser.add_argument("--use-accel", action="store_true")
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    code_path = Path(args.code_path).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()

    if not code_path.exists():
        _emit({"event": "startup_error", "ok": False, "error": f"IndexTTS code path not found: {code_path}"})
        return 1
    if not model_dir.exists():
        _emit({"event": "startup_error", "ok": False, "error": f"IndexTTS model dir not found: {model_dir}"})
        return 1
    if not config_path.exists():
        _emit({"event": "startup_error", "ok": False, "error": f"IndexTTS config path not found: {config_path}"})
        return 1

    try:
        generation_kwargs = json.loads(args.generation_kwargs_json or "{}")
        if not isinstance(generation_kwargs, dict):
            raise ValueError("generation kwargs must decode to a JSON object")
    except Exception as exc:
        _emit({"event": "startup_error", "ok": False, "error": f"Invalid generation kwargs JSON: {exc}"})
        return 1

    try:
        if str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))
        os.chdir(str(code_path))

        from indextts.infer_v2 import IndexTTS2

        device = _resolve_device(args.device)
        tts = IndexTTS2(
            cfg_path=str(config_path),
            model_dir=str(model_dir),
            use_fp16=bool(args.use_fp16),
            device=device,
            use_cuda_kernel=bool(args.use_cuda_kernel),
            use_deepspeed=bool(args.use_deepspeed),
            use_accel=bool(args.use_accel),
            use_torch_compile=bool(args.use_torch_compile),
        )

        parameter_count = _maybe_count_parameters(getattr(tts, "gpt", None))
        _emit(
            {
                "event": "ready",
                "ok": True,
                "device": device,
                "parameter_count": parameter_count,
            }
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"event": "startup_error", "ok": False, "error": str(exc)})
        return 1

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception as exc:
            _emit({"event": "response", "ok": False, "error": f"Invalid JSON request: {exc}"})
            continue

        action = request.get("action")
        if action == "close":
            _emit({"event": "response", "ok": True})
            return 0

        if action != "generate":
            _emit({"event": "response", "ok": False, "error": f"Unsupported action: {action}"})
            continue

        try:
            ref_audio = Path(str(request["ref_audio"])).expanduser().resolve()
            text = str(request["text"]).strip()
            output_path = Path(str(request["output_path"])).expanduser().resolve()
            emo_audio_prompt_value = request.get("emo_audio_prompt")
            emo_audio_prompt = (
                Path(str(emo_audio_prompt_value)).expanduser().resolve()
                if emo_audio_prompt_value not in (None, "")
                else None
            )
            emo_alpha = float(request.get("emo_alpha", 1.0))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                output_path.unlink()

            tts.infer(
                spk_audio_prompt=str(ref_audio),
                text=text,
                output_path=str(output_path),
                emo_audio_prompt=str(emo_audio_prompt) if emo_audio_prompt is not None else None,
                emo_alpha=emo_alpha,
                interval_silence=int(args.interval_silence),
                verbose=bool(args.verbose),
                max_text_tokens_per_segment=int(args.max_text_tokens_per_segment),
                **generation_kwargs,
            )

            _emit(
                {
                    "event": "response",
                    "ok": True,
                    "output_path": str(output_path),
                }
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _emit({"event": "response", "ok": False, "error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
