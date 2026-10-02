"""RVCBench: a benchmark for voice cloning robustness and audio protection."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rvcbench")
except PackageNotFoundError:  # source checkout that has not been installed
    __version__ = "0+unknown"
