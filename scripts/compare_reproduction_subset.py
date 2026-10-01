#!/usr/bin/env python3
"""Compare a scored subset to exact historical pairs with speaker-level uncertainty."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.reproduction import compare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=Path)
parser.add_argument('historical_csv', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = compare(args.run_dir, args.historical_csv, args.output)
print(f'{report["model"]}: {report["matched_pairs"]} matched pairs; {args.output}')
