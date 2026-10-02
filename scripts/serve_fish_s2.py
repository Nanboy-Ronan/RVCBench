#!/usr/bin/env python3
"""Start the official single-worker Fish API with seed setup before warmup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-path', type=Path, required=True)
    parser.add_argument('--startup-seed', type=int, default=42)
    parser.add_argument('--stable-codec-activations', action='store_true',
                        help='Use the same Snake expression eagerly on owned codec modules')
    parser.add_argument('--diagnostic-trace', type=Path,
                        help='Record reference and generated token hashes; introduces CUDA synchronization')
    parser.add_argument('server_args', nargs=argparse.REMAINDER,
                        help='Official API arguments after --')
    args = parser.parse_args()
    trace_path = args.diagnostic_trace.resolve() if args.diagnostic_trace else None
    source = args.code_path.resolve()
    entry = source / 'tools/api_server.py'
    if not entry.is_file():
        parser.error(f'Official API entry is missing: {entry}')
    if not 0 <= args.startup_seed <= 2**31:
        parser.error('startup-seed must be between 0 and 2**31')
    native_args = args.server_args
    if native_args[:1] == ['--']:
        native_args = native_args[1:]
    worker_parser = argparse.ArgumentParser(add_help=False)
    worker_parser.add_argument('--workers', type=int, default=1)
    workers, _ = worker_parser.parse_known_args(native_args)
    if workers.workers != 1:
        parser.error('This launcher requires workers=1 so startup seed setup stays in the serving process')
    # This is a dedicated process. Source imports and cwd belong to its lifetime.
    sys.path.insert(0, str(source))
    os.chdir(source)
    from fish_speech.utils import set_seed
    module_path = Path(sys.modules[set_seed.__module__].__file__).resolve()
    if not module_path.is_relative_to(source):
        raise RuntimeError(f'Fish utilities resolved outside requested checkout: {module_path}')
    set_seed(args.startup_seed)
    import torch
    print(f'Fish startup seed={args.startup_seed}; '
          f'cudnn.deterministic={torch.backends.cudnn.deterministic}; '
          f'cudnn.benchmark={torch.backends.cudnn.benchmark}', flush=True)
    sys.argv = [str(entry), *native_args]
    if trace_path or args.stable_codec_activations:
        # Optional observations on the owned engine instance only. Copies to CPU
        # synchronize CUDA; these diagnostic runs are not timing measurements.
        from tools.api_server import API
        from tools.server.api_utils import parse_args
        import uvicorn
        native = parse_args()
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            if trace_path.exists():
                raise FileExistsError(f'Refusing to overwrite diagnostic trace: {trace_path}')

        def record(kind, tensor, **fields):
            value = tensor.detach().cpu().contiguous()
            row = {'kind': kind, 'shape': list(value.shape), 'dtype': str(value.dtype),
                   'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(), **fields}
            with trace_path.open('a') as stream:
                stream.write(json.dumps(row) + '\n')

        class DiagnosticAPI(API):
            async def initialize_app(self, app):
                await super().initialize_app(app)
                engine = app.state.model_manager.tts_inference_engine
                if args.stable_codec_activations:
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
                    from rvcbench.models.stable_codec_activations import stabilize_codec_snake
                    count = stabilize_codec_snake(engine.decoder_model)
                    if not count:
                        raise RuntimeError('No DAC Snake activations found; stable variant not applied')
                    print(f'Owned codec eager Snake activations={count}', flush=True)
                if not trace_path:
                    return
                encode = engine.encode_reference
                decode = engine.get_audio_segment

                def traced_encode(reference_audio, enable_reference_audio):
                    tokens = encode(reference_audio, enable_reference_audio)
                    if tokens is not None:
                        record('reference_tokens', tokens,
                               audio_sha256=hashlib.sha256(reference_audio).hexdigest())
                    return tokens

                def traced_decode(result):
                    record('generated_tokens', result.codes)
                    return decode(result)

                engine.encode_reference = traced_encode
                engine.get_audio_segment = traced_decode

        host, port = native.listen.rsplit(':', 1)
        uvicorn.run(DiagnosticAPI(args=native).app, host=host.strip('[]'), port=int(port),
                    workers=1, log_level='info')
        return
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
