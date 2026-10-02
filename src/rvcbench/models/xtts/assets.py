"""Files consumed by native XTTS inference, without loading model weights."""
from pathlib import Path

SNAPSHOT_FILES = ('config.json', 'model.pth', 'vocab.json', 'speakers_xtts.pth')


def checkpoint_files(directory, *, config_path=None, checkpoint_path=None,
                     vocab_path=None, speaker_file_path=None):
    root = Path(directory).expanduser()
    files = {
        'config_path': Path(config_path).expanduser() if config_path else root / 'config.json',
        'checkpoint_path': Path(checkpoint_path).expanduser() if checkpoint_path else root / 'model.pth',
        'vocab_path': Path(vocab_path).expanduser() if vocab_path else root / 'vocab.json',
    }
    speaker = Path(speaker_file_path).expanduser() if speaker_file_path else root / 'speakers_xtts.pth'
    # The native cloning path supports checkpoints without a preset speaker bank.
    if speaker_file_path or speaker.exists():
        files['speaker_file_path'] = speaker
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'XTTS {name} is missing or empty: {path}')
    return files
