"""CPU fixture adapter for pipeline checks only; never a benchmark model."""
import shutil
from pathlib import Path
from .base_adversary import BaseAdversary


class SmokeAdversary(BaseAdversary):
    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.logger = logger

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        self._init_synthesis_timings(Path(output_path))
        for sample in dataset.get_zero_shot_samples():
            path = self._speaker_output_dir(Path(output_path), sample.speaker_id) / self._cloned_filename(sample, sample.index)
            shutil.copyfile(sample.prompt_path, path)
