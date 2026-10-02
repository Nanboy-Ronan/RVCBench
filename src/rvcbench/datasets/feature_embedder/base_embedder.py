from abc import ABC, abstractmethod
from pathlib import Path

class BaseEmbedder(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def get_speaker_path(self) -> Path:
        pass

    @abstractmethod
    def get_train_files(self) -> list:
        pass