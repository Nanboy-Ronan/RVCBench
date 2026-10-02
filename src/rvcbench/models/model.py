"""Common base model helpers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class BaseModel(ABC):
    """Baseline interface shared by all runnable models."""

    def __init__(self, *args, **kwargs) -> None:
        auto_load: bool = bool(kwargs.pop("auto_load", False))
        device_override = kwargs.pop("device", None)

        self.config = None
        self.model_config = None
        self.dataset_config = None
        self.checkpoint_path = None
        self.model_name = None

        logger: Optional[logging.Logger]

        if len(args) >= 3:
            config, model_config, dataset_config = args[:3]
            remaining = args[3:]

            logger = kwargs.pop("logger", None)
            if remaining:
                if logger is not None:
                    raise TypeError("BaseModel received logger both positionally and via keyword.")
                if len(remaining) > 1:
                    raise TypeError("BaseModel received unexpected positional arguments after logger.")
                logger = remaining[0]

            if len(args) > 4:
                raise TypeError("BaseModel received too many positional arguments.")

            self.config = config
            self.model_config = model_config
            self.dataset_config = dataset_config
            self.checkpoint_path = getattr(config, "checkpoint_path", None)
            self.model_name = getattr(model_config, "name", getattr(model_config, "model_name", None))
        else:
            logger = kwargs.pop("logger", None)
            model_name = kwargs.pop("model_name_or_path", None)
            if model_name is None:
                model_name = kwargs.pop("model_name", None)
            if model_name is None:
                raise TypeError(
                    "BaseModel requires either (config, model_config, dataset_config) positional arguments "
                    "or a 'model_name_or_path'/'model_name' keyword argument."
                )

            self.model_name = str(model_name)
            self.config = kwargs.pop("config", None)
            self.model_config = kwargs.pop("model_config", None)
            self.dataset_config = kwargs.pop("dataset_config", None)
            self.checkpoint_path = kwargs.pop("checkpoint_path", None)

        if kwargs:
            unexpected = ", ".join(sorted(map(str, kwargs.keys())))
            raise TypeError(f"BaseModel received unexpected keyword arguments: {unexpected}")

        self.device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._model_ready = False
        self.dataset_name = getattr(self.dataset_config, "name", None) if self.dataset_config is not None else None

        if self.model_name is None:
            self.model_name = self.__class__.__name__

        self.logger = logger or logging.getLogger(self.model_name)
        self.logger.info("Loaded model: %s", self.model_name)

        if auto_load:
            self.ensure_model()

    def to(self, device):
        self.model.to(device)

    def ensure_model(self) -> None:
        """Load model weights if they have not been prepared yet."""
        if not self.is_model_ready():
            self.logger.debug("Loading model '%s' on %s", self.model_name, self.device)
            self.load_model()
            if hasattr(self, "model") and isinstance(self.model, torch.nn.Module):
                param_count = sum(p.numel() for p in self.model.parameters())
                human_readable = f"{param_count:,}"
                self.logger.info(
                    "Model '%s' initialised with %s trainable parameters.",
                    self.model_name,
                    human_readable,
                )
                self.logger.debug(
                    "Model '%s' parameter count (exact): %d",
                    self.model_name,
                    param_count,
                )
            self._model_ready = True

    def is_model_ready(self) -> bool:
        """Whether the subclass has finished preparing runtime resources."""
        return bool(self._model_ready)

    @abstractmethod
    def load_model(self) -> None:
        """Materialise parameters and runtime state needed before inference."""

    def generate(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement 'generate'.")
