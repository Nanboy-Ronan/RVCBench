#!/usr/bin/env python3
"""Check all public quickstart commands using temporary synthetic data, offline."""
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]


def main():
    import pandas as pd
    with tempfile.TemporaryDirectory(prefix='rvcbench-quickstarts-') as tmp:
        root = Path(tmp)
        for name, speaker in [('Libritts', '1089'), ('VCTK', 'p225')]:
            folder = root / name / 'audios' / speaker
            folder.mkdir(parents=True)
            with wave.open(str(folder / 'fixture.wav'), 'wb') as f:
                f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                f.writeframes(b'\0\0' * 1600)
            pd.DataFrame([{'pair_id': 'fixture', 'speaker_id': speaker,
                          'prompt_file_name': f'audios/{speaker}/fixture.wav',
                          'target_file_name': f'audios/{speaker}/fixture.wav',
                          'prompt_text': 'fixture', 'target_text': 'fixture'}]).to_parquet(root / name / 'metadata.parquet')
        for script in ('run_qwen3tts_quickstart.py', 'run_fishspeech_quickstart.py',
                       'run_fishspeech_s2_quickstart.py', 'run_protect_qwen3tts_quickstart.py'):
            subprocess.run([sys.executable, str(REPO_DIR / 'scripts' / script), '--dry-run',
                            '--no-hf-download', '--data-dir', tmp], cwd=REPO_DIR, check=True)
    print('All four quickstarts passed offline command/layout checks (no model inference).')


if __name__ == '__main__':
    main()
