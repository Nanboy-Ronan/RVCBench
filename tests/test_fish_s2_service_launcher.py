"""Guard the process topology required for seeded service initialization."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('workers', [['--workers', '2'], ['--workers=2']])
def test_seeded_service_rejects_spawned_workers_before_upstream_import(tmp_path, workers):
    checkout = tmp_path / 'fish'
    entry = checkout / 'tools/api_server.py'
    entry.parent.mkdir(parents=True)
    entry.write_text("raise AssertionError('upstream must not start')\n")
    launcher = Path(__file__).resolve().parents[1] / 'scripts/serve_fish_s2.py'
    result = subprocess.run([sys.executable, str(launcher), '--code-path', str(checkout),
                             '--', *workers], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'requires workers=1' in result.stderr
    assert 'Traceback' not in result.stderr
