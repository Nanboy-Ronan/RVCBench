"""Asset identities consumed by the supported ZipVoice inference runtimes."""
from pathlib import Path

MODEL_REPO = 'k2-fsa/ZipVoice'
MODEL_DIRS = {'zipvoice': 'zipvoice', 'zipvoice_distill': 'zipvoice_distill'}
VOCODER_REPO = 'charactr/vocos-mel-24khz'
VOCODER_FILES = ('config.yaml', 'pytorch_model.bin')


def required_files(directory, names, *, component):
    root = Path(directory).expanduser()
    files = [root / name for name in names]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'ZipVoice {component} file is missing or empty: {path}')
    return files


def model_files(directory, checkpoint_name='model.pt'):
    return required_files(directory, (checkpoint_name, 'model.json', 'tokens.txt'), component='model')


def vocoder_files(directory):
    return required_files(directory, VOCODER_FILES, component='vocoder')
