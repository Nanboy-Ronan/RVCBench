"""Lazy protection registry.

Importing every protector eagerly pulls in heavyweight optional dependencies
(`transformers`, `peft`, surrogate-model trainers, etc.). That breaks light
quickstarts like `grnoise_on_libritts`, which should not need those stacks.

Expose protectors on demand so `getattr(rvcbench.protection, name)` only imports the
module required by the selected config.
"""

from __future__ import annotations

from importlib import import_module


_REGISTRY = {
    "Dummy": ("rvcbench.protection.dummy", "Dummy"),
    "EMProtector": ("rvcbench.protection.em", "EMProtector"),
    "GRNoiseProtector": ("rvcbench.protection.random_noise", "GRNoiseProtector"),
    "EnkiduProtector": ("rvcbench.protection.enkidu", "EnkiduProtector"),
    "SafeSpeechProtector": ("rvcbench.protection.safespeech_use_wrapper", "SafeSpeechProtector"),
}

__all__ = list(_REGISTRY)


def __getattr__(name: str):
    try:
        module_name, attr_name = _REGISTRY[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
