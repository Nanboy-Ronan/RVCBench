"""Local SparkTTS asset validation and checked native BiCodec loading."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

DERIVED_MEL_BUFFERS = frozenset({
    'mel_transformer.spectrogram.window', 'mel_transformer.mel_scale.fb',
})


def model_directory(code_path, model_dir):
    from hydra.utils import to_absolute_path
    path = Path(model_dir).expanduser()
    source = Path(to_absolute_path(str(Path(code_path).expanduser())))
    return (path if path.is_absolute() else source / path).resolve()


def _required(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'SparkTTS asset is missing or empty: {path}')
    return path


def _weights(root):
    for name in ('model.safetensors', 'model.safetensors.index.json',
                 'pytorch_model.bin', 'pytorch_model.bin.index.json'):
        path = root / name
        if path.exists():
            _required(path)
            if not name.endswith('.index.json'):
                return [path]
            index = json.loads(path.read_text())
            mapping = index.get('weight_map') if isinstance(index, dict) else None
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f'SparkTTS weight index has no valid weight_map: {path}')
            names = list(mapping.values())
            if any(not isinstance(n, str) or not n or Path(n).is_absolute()
                   or '..' in Path(n).parts for n in names):
                raise ValueError(f'SparkTTS weight index contains invalid shard paths: {path}')
            return [path] + [_required(root / name) for name in sorted(set(names))]
    raise FileNotFoundError(f'SparkTTS model weights are missing: {root}')


def checkpoint_files(root):
    root = Path(root)
    files = [_required(root / name) for name in (
        'config.yaml', 'BiCodec/config.yaml', 'BiCodec/model.safetensors',
        'LLM/config.json', 'wav2vec2-large-xlsr-53/config.json',
        'wav2vec2-large-xlsr-53/preprocessor_config.json')]
    llm = root / 'LLM'
    if (llm / 'tokenizer.json').exists():
        files.append(_required(llm / 'tokenizer.json'))
    else:
        files.extend(_required(llm / name) for name in ('vocab.json', 'merges.txt'))
    return files + _weights(llm) + _weights(root / 'wav2vec2-large-xlsr-53')


@contextmanager
def checked_bicodec_loading(cls, receipts):
    """Retain configuration-derived buffers, rejecting incomplete learned state."""
    original = cls.load_state_dict

    def checked(model, state, *args, **kwargs):
        result = original(model, state, *args, **kwargs)
        missing, unexpected = set(result.missing_keys), set(result.unexpected_keys)
        buffers, parameters = dict(model.named_buffers()), dict(model.named_parameters())
        invalid = missing - DERIVED_MEL_BUFFERS
        invalid |= {name for name in missing if name not in buffers or name in parameters}
        if invalid or unexpected:
            raise RuntimeError(f'SparkTTS BiCodec checkpoint mismatch: missing={sorted(invalid)}, '
                               f'unexpected={sorted(unexpected)}')
        derived = {}
        for name in sorted(missing):
            import torch
            tensor = buffers[name].detach().cpu().contiguous()
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f'SparkTTS derived mel buffer is nonfinite: {name}')
            derived[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                             'sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
        receipts.append({'checkpoint_keys': len(state), 'missing_derived_buffers': derived,
                         'missing_learned_state': [], 'unexpected_keys': []})
        return result

    with patch.object(cls, 'load_state_dict', checked):
        yield
