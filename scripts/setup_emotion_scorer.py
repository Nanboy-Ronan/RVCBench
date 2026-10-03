#!/usr/bin/env python3
"""Fetch pinned native emotion assets. Equivalent to `rvcbench setup-scorers --metrics emotion`."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rvcbench.evaluation.setup import setup_emotion  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify local assets without downloading or modifying files')
    args = parser.parse_args()
    print(json.dumps(setup_emotion(args.check_only), indent=2))
