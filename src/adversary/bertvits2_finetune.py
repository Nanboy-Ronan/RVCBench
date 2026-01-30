import os
import sys
import subprocess
from pathlib import Path
from .base_adversary import BaseAdversary

class BertVits2FinetuneAdversary(BaseAdversary):
    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.original_cwd = os.getcwd()
        self.logger = logger

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        """
        Fine-tunes BERT-VITS2 on the protected data and then synthesizes test sentences.
        """
        del dataset
        del protected_audio_path

        protection_mode = self.config['run_name'].replace('bert_vits2_', '')

        self.logger.info("[Adversary] Step 1: Fine-tuning on protected audio...")
        subprocess.run([
            sys.executable, 'train.py',
            '--dataset', 'LibriTTS',
            '--model', self.config['model_name'],
            '--mode', 'SPEC', # The training script uses this to determine filelist
            '--checkpoint-path', self.config['checkpoint_path'],
            '--batch-size', str(self.config['batch_size']),
            '--gpu', str(self.device.index) if self.device.type == 'cuda' else "-1"
        ], cwd=ORIGINAL_CODE_PATH, check=True)
        
        self.logger.info(f"[Adversary] Step 2: Synthesizing audio to {output_path}...")
        subprocess.run([
            sys.executable, 'evaluate.py',
            '--dataset', 'LibriTTS',
            '--model', self.config['model_name'],
            '--mode', 'SPEC',
            '--checkpoint-path', self.config['checkpoint_path'],
            '--gpu', str(self.device.index) if self.device.type == 'cuda' else "-1"
        ], cwd=ORIGINAL_CODE_PATH, check=True)

        # The evaluate script will save files inside its own structure.
        # We need to copy them to our main results directory.
        # This is a common challenge when wrapping command-line tools.
        source_eval_dir = Path(ORIGINAL_CODE_PATH) / 'evaluation/data/LibriTTS/SPEC'
        dest_eval_dir = Path(self.original_cwd) / output_path
        dest_eval_dir.mkdir(parents=True, exist_ok=True)
        
        if source_eval_dir.exists():
            for f in source_eval_dir.glob('*.wav'):
                os.rename(f, dest_eval_dir / f.name)
