"""RVCBench: a benchmark for voice cloning robustness and audio protection."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rvcbench")
except PackageNotFoundError:  # source checkout that has not been installed
    __version__ = "0+unknown"

__all__ = ["VoiceCloningAdapter", "__version__"]


def __getattr__(name):
    # Imported on demand so that `import rvcbench` stays free of heavy dependencies.
    if name == "VoiceCloningAdapter":
        from rvcbench.adapter import VoiceCloningAdapter
        return VoiceCloningAdapter
    raise AttributeError(f"module 'rvcbench' has no attribute {name!r}")
