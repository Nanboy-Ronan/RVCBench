from pathlib import Path
from .base_embedder import BaseEmbedder

class STFTEmbedder(BaseEmbedder):
    def __init__(self, config):
        super().__init__(config)

    def get_speaker_path(self) -> Path:
        # Implement the method to return the speaker's audio directory
        pass

    def get_train_files(self) -> list:
        # Implement the method to return the training audio files
        pass