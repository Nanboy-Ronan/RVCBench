"""The six checkpoint files consumed by the native MaskGCT worker."""
from pathlib import Path

CHECKPOINT_FILES = (
    'semantic_codec/model.safetensors',
    'acoustic_codec/model.safetensors',
    'acoustic_codec/model_1.safetensors',
    't2s_model/model.safetensors',
    's2a_model/s2a_model_1layer/model.safetensors',
    's2a_model/s2a_model_full/model.safetensors',
)


def checkpoint_files(directory):
    root = Path(directory).expanduser().absolute()
    result = [root / name for name in CHECKPOINT_FILES]
    missing = [name for name, path in zip(CHECKPOINT_FILES, result)
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError('Incomplete MaskGCT checkpoint directory: ' + ', '.join(missing))
    return result

SEMANTIC_REPO = 'facebook/w2v-bert-2.0'
SEMANTIC_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')


def semantic_files(directory):
    root = Path(directory).expanduser().absolute()
    result = [root / name for name in SEMANTIC_FILES]
    missing = [name for name, path in zip(SEMANTIC_FILES, result)
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError('Incomplete MaskGCT semantic snapshot: ' + ', '.join(missing))
    return result
