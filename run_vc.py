"""Source-checkout shortcut for `rvcbench run`."""
import sys
from pathlib import Path

# Use this checkout's package even when it has not been (re)installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rvcbench.entrypoints.vc import main

if __name__ == "__main__":
    main()
