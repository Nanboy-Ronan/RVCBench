from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
# Placeholder and system roots that may appear in examples, tests and commands.
ALLOWED_ROOTS = ('path', 'absolute', 'foreign', 'usr', 'tmp', 'etc', 'opt', 'dev', 'proc', 'var', 'bin', 'sys',
                 'lib', 'root', 'workspace', 'content', 'app', r'v\d')
ABSOLUTE_PATH = re.compile(
    r'(?<![\w.:/~$}{\-\\*<>\]])/(?!(?:' + '|'.join(ALLOWED_ROOTS) + r')(?:/|\b))'
    r'[A-Za-z][\w.-]*/[\w.-]+/[\w.-]+(?:/[\w.-]+)*')


def machine_paths(text):
    return [match.group(0) for match in ABSOLUTE_PATH.finditer(text)]


@pytest.mark.parametrize('text', [
    'root = "/home/someone/project/results"',
    'audio_url=\'/lab-storage/someone/results/run/20250101-000000/a.wav\'',
    'p283,103,/mnt/shared/data/VCTK/audios/p283/p283_005_mic1.wav,other',
])
def test_detector_reports_machine_specific_paths(text):
    assert machine_paths(text)


@pytest.mark.parametrize('text', [
    'rvcbench compare-timing /absolute/path/to/generation-a /absolute/path/to/generation-b',
    'git clone https://github.com/Nanboy-Ronan/RVCBench.git checkpoints/fish_speech_s1',
    'configs/ots_vc/clean/libritts/qwen3_tts_ots.yaml and ./results/run/audio and ~/data/VCTK/audios',
    'POST /v1/audio/speech and ${ROOT}/data/Libritts/audios',
])
def test_detector_accepts_placeholders_urls_and_relative_paths(text):
    assert machine_paths(text) == []


def test_tracked_files_do_not_record_machine_specific_paths():
    listed = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode:
        pytest.skip('not a Git checkout')
    found = {}
    for name in filter(None, listed.stdout.split('\0')):
        if Path(name).name == '.gitignore' or ROOT / name == Path(__file__).resolve():
            continue  # root-anchored ignore patterns and this file's own examples
        try:
            text = (ROOT / name).read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        paths = machine_paths(text)
        if paths:
            found[name] = paths[:3]
    assert found == {}
