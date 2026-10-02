#!/usr/bin/env python3
"""Persistent JSONL worker process for Amphion MaskGCT inference."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout, contextmanager
import json
import hashlib
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


@contextmanager
def _semantic_snapshot_loaders(model_class, processor_class, directory):
    """Bind upstream's hardcoded semantic Hub loads inside this isolated worker."""
    from unittest.mock import patch
    def local_loader(loader):
        def load(repo, *args, **kwargs):
            if str(repo) != 'facebook/w2v-bert-2.0':
                raise ValueError(f'Unexpected MaskGCT semantic model reference: {repo}')
            kwargs['local_files_only'] = True
            return loader(str(directory), *args, **kwargs)
        return load
    model_loader = local_loader(model_class.from_pretrained)
    processor_loader = local_loader(processor_class.from_pretrained)
    with patch.object(model_class, 'from_pretrained', side_effect=model_loader), \
         patch.object(processor_class, 'from_pretrained', side_effect=processor_loader):
        yield


def _espeak_runtime():
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    wrapper = EspeakWrapper()
    library, data = Path(wrapper.library_path), Path(wrapper.data_path)
    files = {str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(data.rglob('*')) if path.is_file()}
    if not files:
        raise FileNotFoundError('MaskGCT eSpeak data directory is empty')
    return {'version': list(wrapper.version), 'library_path': str(library),
            'data_path': str(data), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
            'data_sha256': hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'data_file_count': len(files)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--semantic-model-path", required=True)
    parser.add_argument("--generation-kwargs-json", default="{}")
    parser.add_argument("--expected-espeak-json", default="null")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    code_path = Path(args.code_path).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().absolute()
    semantic_dir = Path(args.semantic_model_path).expanduser().absolute()

    if not code_path.exists():
        _emit({"event": "startup_error", "ok": False, "error": f"MaskGCT code path not found: {code_path}"})
        return 1
    if not config_path.exists():
        _emit({"event": "startup_error", "ok": False, "error": f"MaskGCT config path not found: {config_path}"})
        return 1

    try:
        expected_espeak = json.loads(args.expected_espeak_json)
        if expected_espeak is not None and not isinstance(expected_espeak, dict):
            raise ValueError("Expected eSpeak resources must be a JSON object")
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

        with redirect_stdout(sys.stderr):
            import torch
            import soundfile as sf
            import safetensors.torch as safetensors_torch
            from models.tts.maskgct.maskgct_utils import (
                MaskGCT_Inference_Pipeline, build_semantic_model, build_semantic_codec,
                build_acoustic_codec, build_t2s_model, build_s2a_model, load_config,
            )

            import models.tts.maskgct.maskgct_utils as native_utils
            espeak_runtime = _espeak_runtime()
            if expected_espeak is not None and json.dumps(espeak_runtime, sort_keys=True) != json.dumps(expected_espeak, sort_keys=True):
                raise RuntimeError('MaskGCT eSpeak resources changed after asset resolution')
            device = torch.device(_resolve_device(args.device))
            cfg = load_config(str(config_path))
            with _semantic_snapshot_loaders(native_utils.Wav2Vec2BertModel,
                                            native_utils.SeamlessM4TFeatureExtractor, semantic_dir):
                semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
            semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
            codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
            t2s_model = build_t2s_model(cfg.model.t2s_model, device)
            s2a_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
            s2a_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)
            weights = [
                (semantic_codec, 'semantic_codec/model.safetensors'),
                (codec_encoder, 'acoustic_codec/model.safetensors'),
                (codec_decoder, 'acoustic_codec/model_1.safetensors'),
                (t2s_model, 't2s_model/model.safetensors'),
                (s2a_1layer, 's2a_model/s2a_model_1layer/model.safetensors'),
                (s2a_full, 's2a_model/s2a_model_full/model.safetensors'),
            ]
            for model, filename in weights:
                checkpoint = checkpoint_dir / filename
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise FileNotFoundError(f'Missing MaskGCT checkpoint: {checkpoint}')
                safetensors_torch.load_model(model, str(checkpoint))
            with _semantic_snapshot_loaders(native_utils.Wav2Vec2BertModel,
                                            native_utils.SeamlessM4TFeatureExtractor, semantic_dir):
                inference_pipeline = MaskGCT_Inference_Pipeline(
                    semantic_model, semantic_codec, codec_encoder, codec_decoder,
                    t2s_model, s2a_1layer, s2a_full, semantic_mean, semantic_std, device,
                )

            parameter_count = None
            for model_obj in (t2s_model, s2a_1layer, s2a_full):
                count = _maybe_count_parameters(model_obj)
                if count is not None:
                    parameter_count = (parameter_count or 0) + count

        _emit(
            {
                "event": "ready",
                "ok": True,
                "device": str(device),
                "parameter_count": parameter_count,
                "espeak_runtime": espeak_runtime,
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

        if not isinstance(request, dict):
            _emit({"event": "response", "ok": False, "error": "Request must be a JSON object"})
            continue
        request_id = request.get('request_id')
        action = request.get("action")
        if action == "close":
            _emit({"event": "response", "ok": True})
            return 0

        if action != "generate":
            _emit({"event": "response", "ok": False, "error": f"Unsupported action: {action}"})
            continue

        try:
            if request.get('seed') is not None:
                import random
                import numpy as np
                import torch
                seed = int(request['seed'])
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            prompt_speech_path = Path(str(request["prompt_speech_path"])).expanduser().resolve()
            prompt_text = str(request.get("prompt_text", "")).strip()
            target_text = str(request.get("target_text", "")).strip()
            output_path = Path(str(request["output_path"])).expanduser().resolve()
            prompt_language = str(request.get("prompt_language", "en")).strip() or "en"
            target_language = str(request.get("target_language", prompt_language)).strip() or prompt_language
            target_len = request.get("target_len")
            generation_overrides = request.get("generation_overrides") or {}

            output_path.parent.mkdir(parents=True, exist_ok=True)

            gen_kwargs = dict(generation_kwargs)
            gen_kwargs.update(generation_overrides)

            infer_kwargs: dict[str, Any] = {
                "prompt_speech_path": str(prompt_speech_path),
                "prompt_text": prompt_text,
                "target_text": target_text,
                "language": prompt_language.lower(),
                "target_language": target_language.lower(),
            }
            if target_len is not None:
                infer_kwargs["target_len"] = float(target_len)
            infer_kwargs.update(gen_kwargs)

            with redirect_stdout(sys.stderr):
                recovered_audio = inference_pipeline.maskgct_inference(**infer_kwargs)
            sf.write(str(output_path), recovered_audio, 24000)

            _emit(
                {
                    "event": "response",
                    "ok": True,
                    "output_path": str(output_path),
                    "request_id": request_id,
                    "seed": request.get("seed"),
                }
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _emit({"event": "response", "ok": False, "request_id": request_id, "error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
