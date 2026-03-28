from src.protection.base_protector import BaseProtector
import shutil
from pathlib import Path

class Dummy(BaseProtector):
    def __init__(self, model_config, dataset_config, logger, output_dir, config, device):
        super().__init__(output_dir=output_dir, config=config, dataset_config=dataset_config, logger=logger, device=device)

    def protect(self):
        self.logger.info("Applying dummy protection (copying files)...")
        train_files_map = self.speaker_data.get_train_files_map()
        for original_path, transcript in train_files_map.items():
            original_path = Path(original_path)
            # Create the speaker directory in the output directory
            speaker_dir = Path(self.output_dir) / original_path.parent.name
            speaker_dir.mkdir(parents=True, exist_ok=True)
            # Copy the file
            shutil.copy(original_path, speaker_dir / original_path.name)
        self.logger.info(f"Copied {len(train_files_map)} files to {self.output_dir}")
