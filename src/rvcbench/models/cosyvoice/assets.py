"""Local asset validation for CosyVoice native model constructors."""
from pathlib import Path
import json

COMMON_FILES = ('campplus.onnx', 'flow.pt', 'hift.pt', 'llm.pt')


def canonical_variant(value):
    name = str(value or 'cosyvoice2').lower().strip()
    if name in {'cosyvoice2', 'cozyvoice2', 'cosy2'}:
        return 'cosyvoice2'
    if name in {'cosyvoice', 'cozyvoice', 'cosy1'}:
        return 'cosyvoice'
    raise ValueError(f"Unknown CosyVoice variant {value!r}; expected cosyvoice or cosyvoice2")


def _required(path):
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f'CosyVoice asset is missing or empty: {path}')
    return path


def model_directory(supplied, variant='cosyvoice2'):
    root = Path(supplied).expanduser().resolve()
    name = canonical_variant(variant)
    config = name + '.yaml'
    tokenizer = 'speech_tokenizer_v2.onnx' if name == 'cosyvoice2' else 'speech_tokenizer_v1.onnx'
    def payload(path):
        return all((path / f).is_file() for f in (*COMMON_FILES, config, tokenizer))
    if payload(root):
        return root
    if root.is_dir():
        # Search only inside the explicitly requested directory. Resolve each
        # candidate before checking containment so directory symlinks cannot
        # select an unrelated checkpoint.
        for config_path in sorted(root.rglob(config)):
            candidate = config_path.parent.resolve()
            if candidate.is_relative_to(root) and payload(candidate):
                return candidate
    raise FileNotFoundError(f'CosyVoice model_dir not found or incomplete at {root}; '
                            f'required: {", ".join((*COMMON_FILES, config, tokenizer))}')


def _qwen_weights(root):
    # Match Transformers' standard safetensors-before-PyTorch selection order.
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
                raise ValueError(f'CosyVoice Qwen weight index has no valid weight_map: {path}')
            shards = list(mapping.values())
            if any(not isinstance(n, str) or not n or Path(n).is_absolute()
                   or '..' in Path(n).parts for n in shards):
                raise ValueError(f'CosyVoice Qwen weight index contains invalid shard paths: {path}')
            return [path] + [_required(root / n) for n in sorted(set(shards))]
    raise FileNotFoundError(f'CosyVoice Qwen weights are missing: {root}')


def checkpoint_files(directory, variant='cosyvoice2', zero_shot_speaker_id=''):
    root = Path(directory)
    name = canonical_variant(variant)
    tokenizer = 'speech_tokenizer_v2.onnx' if name == 'cosyvoice2' else 'speech_tokenizer_v1.onnx'
    files = [_required(root / f) for f in (*COMMON_FILES, name + '.yaml', tokenizer)]
    if name == 'cosyvoice2':
        # Native CosyVoice2 overrides qwen_pretrain_path to this exact folder.
        qwen = root / 'CosyVoice-BlankEN'
        files.extend(_required(qwen / f) for f in ('config.json', 'vocab.json', 'merges.txt'))
        files.extend(_qwen_weights(qwen))
        files.extend(_required(qwen / f) for f in ('tokenizer_config.json', 'generation_config.json')
                     if (qwen / f).exists())
    bank = root / 'spk2info.pt'
    if zero_shot_speaker_id or bank.exists():
        files.append(_required(bank))
    return files
