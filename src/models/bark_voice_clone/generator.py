"""Bark reference encoding with native Fairseq HuBERT and learned quantization."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import tempfile
import types
from zipfile import ZipFile

import numpy as np
import torch

from src.models.model import BaseModel


@dataclass
class BarkVoiceCloneGeneratorConfig:
    code_path: Path
    models_dir: Path | None = None
    hubert_checkpoint: Path | None = None
    hubert_tokenizer: Path | None = None
    text_tokenizer_path: Path | None = None
    cache_dir: Path | None = None
    prompt_cache_dir: Path | None = None
    text_temperature: float = .7
    text_top_k: int | None = None
    text_top_p: float | None = None
    coarse_temperature: float = .7
    coarse_top_k: int | None = None
    coarse_top_p: float | None = None
    fine_temperature: float = .5
    semantic_use_kv_cache: bool = True
    coarse_use_kv_cache: bool = True
    silent: bool = True
    force_reload_models: bool = False
    max_prompt_seconds: float | None = None
    hubert_layer: int = 9


class CPULoadTorch:
    """Private loader binding; shared torch.load remains untouched."""
    def __getattr__(self, key):
        return getattr(torch, key)

    def load(self, path, *args, **kwargs):
        kwargs['map_location'] = 'cpu'
        checkpoint = torch.load(path, *args, **kwargs)
        state = checkpoint.get('model', checkpoint)
        if any('lora_' in key for key in state):
            raise ValueError('Bark LoRA checkpoints require a distinct runtime; refusing to discard adapters')
        return checkpoint


def private_runtime(module, device, tokenizer_path):
    namespace = dict(vars(module))
    namespace.update(models={}, models_devices={}, USE_SMALL_MODELS=False, OFFLOAD_CPU=False,
                     torch=CPULoadTorch())
    for name, value in vars(module).items():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(module):
            fn = types.FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            fn.__kwdefaults__ = value.__kwdefaults__
            namespace[name] = fn
    namespace['_grab_best_device'] = lambda use_gpu=True: str(device)
    original_tokenizer = module.BertTokenizer
    class LocalTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return original_tokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    namespace['BertTokenizer'] = LocalTokenizer
    for name in ('GPT', 'FineGPT'):
        parent = namespace[name]
        class StrictModel(parent):
            def load_state_dict(self, state, strict=True, **kwargs):
                result = super().load_state_dict(state, strict=False, **kwargs)
                missing = [key for key in result.missing_keys if not key.endswith('.attn.bias')]
                extra = [key for key in result.unexpected_keys if not key.endswith('.attn.bias')]
                if missing or extra:
                    raise ValueError(f'Bark checkpoint mismatch: missing={missing}, unexpected={extra}')
                return result
        namespace[name] = StrictModel
    return namespace


class BarkVoiceCloneGenerator(BaseModel):
    def __init__(self, config, device, logger):
        super().__init__(model_name_or_path=str(config.models_dir), device=device, logger=logger)
        self.config, self.device = config, torch.device(device)
        self._runtime = self._hubert = self._quantizer = None
        if config.hubert_layer != 9:
            raise ValueError('The released Bark reference tokenizer uses HuBERT layer 9')
        if config.max_prompt_seconds is not None and (not np.isfinite(config.max_prompt_seconds) or config.max_prompt_seconds <= 0):
            raise ValueError('max_prompt_seconds must be finite and positive')

    @contextmanager
    def _scope(self):
        tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        benchmark = torch.backends.cudnn.benchmark
        precision = torch.get_float32_matmul_precision()
        scope = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
        try:
            with scope:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            torch.backends.cudnn.benchmark = benchmark
            torch.set_float32_matmul_precision(precision)

    def load_model(self):
        cfg = self.config
        root = Path(cfg.code_path).resolve()
        for value in (cfg.models_dir, cfg.hubert_checkpoint, cfg.hubert_tokenizer, cfg.text_tokenizer_path):
            if value is None or not Path(value).exists():
                raise FileNotFoundError(f'Bark requires explicit local model, HuBERT, quantizer and text tokenizer assets: {value}')
        for name in ('text_2.pt', 'coarse_2.pt', 'fine_2.pt'):
            if not (Path(cfg.models_dir) / name).is_file():
                raise FileNotFoundError(Path(cfg.models_dir) / name)
        sys.path.insert(0, str(root)) if str(root) not in sys.path else None
        try:
            with self._scope():
                generation = importlib.import_module('bark.generation')
                quantizer = importlib.import_module('hubert.customtokenizer')
                for module in (generation, quantizer):
                    if not Path(module.__file__).resolve().is_relative_to(root):
                        raise ImportError('Bark runtime imported outside configured code_path')
                self._runtime = private_runtime(generation, self.device, cfg.text_tokenizer_path)
                self._runtime['preload_models'](path=str(cfg.models_dir))
                import fairseq
                checkpoint = torch.load(cfg.hubert_checkpoint, map_location='cpu', weights_only=False)
                # This released checkpoint contains legacy argparse metadata.
                # Native task/model factories migrate it through their registered
                # dataclasses without clearing the runner's Hydra singleton.
                args = checkpoint['args']
                task = fairseq.tasks.setup_task(args, from_checkpoint=True)
                task.load_state_dict(checkpoint['task_state'])
                model = fairseq.models.build_model(args, task, from_checkpoint=True)
                model.load_state_dict(checkpoint['model'], strict=True)
                self._hubert = model.to(self.device).eval()
                del checkpoint
                with ZipFile(cfg.hubert_tokenizer) as archive:
                    info = [name for name in archive.namelist() if name.endswith('/.info')]
                    if len(info) > 1:
                        raise ValueError('Bark learned tokenizer architecture metadata is ambiguous')
                    data = quantizer.Data.load(archive.read(info[0]).decode('utf-8')) if info else None
                self._quantizer = (quantizer.CustomTokenizer(data.hidden_size, data.input_size, data.output_size, data.version)
                                   if data else quantizer.CustomTokenizer())
                self._quantizer.load_state_dict(torch.load(cfg.hubert_tokenizer, map_location='cpu', weights_only=True), strict=True)
                self._quantizer.to(self.device).eval()
            self.model = self._runtime['models']
        except Exception:
            self.close()
            raise

    def generate(self, *, text, prompt_audio, sample_index=0):
        del sample_index  # Serial runner seeds with the original source index.
        if not str(text or '').strip():
            raise ValueError('Bark requires nonempty target text')
        self.ensure_model()
        import torchaudio
        cfg, runtime = self.config, self._runtime
        with self._scope(), torch.inference_mode():
            audio, rate = torchaudio.load(str(prompt_audio))
            audio = audio.mean(dim=0, keepdim=True).to(self.device)
            if cfg.max_prompt_seconds is not None:
                audio = audio[:, :int(cfg.max_prompt_seconds * rate)]
            if not audio.numel() or not torch.isfinite(audio).all():
                raise ValueError('Bark reference audio is empty or nonfinite')
            semantic_audio = torchaudio.functional.resample(audio, rate, 16000)
            vectors = self._hubert(semantic_audio, features_only=True, mask=False, output_layer=9)['x'].reshape(-1, 768)
            semantic = self._quantizer.get_token(vectors).cpu().numpy()
            codec = runtime['models']['codec']
            codec_audio = torchaudio.functional.resample(audio, rate, 24000)
            codes = torch.cat([frame[0] for frame in codec.encode(codec_audio.unsqueeze(0))], dim=-1)[0].cpu().numpy()
            if not semantic.size or semantic.min() < 0 or semantic.max() >= 10000 or codes.ndim != 2 or codes.shape[0] != 8:
                raise ValueError('Bark reference encoding returned malformed tokens')
            with tempfile.TemporaryDirectory(prefix='rvcbench-bark-prompt-') as temporary:
                prompt = Path(temporary) / 'reference.npz'
                np.savez(prompt, semantic_prompt=semantic, coarse_prompt=codes[:2], fine_prompt=codes)
                kwargs = dict(history_prompt=str(prompt), silent=cfg.silent)
                x = runtime['generate_text_semantic'](text.strip(), temp=cfg.text_temperature, top_k=cfg.text_top_k,
                    top_p=cfg.text_top_p, use_kv_caching=cfg.semantic_use_kv_cache, **kwargs)
                x = runtime['generate_coarse'](x, temp=cfg.coarse_temperature, top_k=cfg.coarse_top_k,
                    top_p=cfg.coarse_top_p, use_kv_caching=cfg.coarse_use_kv_cache, **kwargs)
                x = runtime['generate_fine'](x, temp=cfg.fine_temperature, **kwargs)
                output = np.asarray(runtime['codec_decode'](x), dtype=np.float32)
        if output.ndim != 1 or not output.size or not np.isfinite(output).all():
            raise ValueError('Bark produced empty, nonfinite or multiple waveform output')
        return output, 24000

    def close(self):
        if self._runtime is not None:
            self._runtime['models'].clear()
        self._runtime = self._hubert = self._quantizer = self.model = None
        self._model_ready = False
