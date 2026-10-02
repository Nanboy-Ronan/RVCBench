import os
import sys
import torch
import subprocess
from pathlib import Path
from ..base_protector import BaseProtector

ORIGINAL_CODE_PATH = str(Path(__file__).parent / "original_code")

class SafeSpeechProtector(BaseProtector):
    def __init__(self, config, device, logger):
        super().__init__(config, device)
        self.original_cwd = os.getcwd()
        self.logger = logger
        # We don't chdir here anymore, but specify cwd for each command

    def protect(self, train_filelist, output_path):
        """
        Runs the full SafeSpeech protection pipeline.
        1. Preprocess text for clean audio.
        2. Generate BERT features for the surrogate model.
        3. Generate adversarial perturbations.
        4. Save the protected audio files.
        5. Run ASR on protected audio to create new transcripts for the adversary.
        6. Preprocess the ASR transcripts.
        7. Generate BERT features for the adversary's training.
        """
        self.logger.info("[SafeSpeech] Step 1: Preprocessing original text...")
        subprocess.run([
            sys.executable, 'preprocess_text.py',
            '--file-path', train_filelist
        ], cwd=ORIGINAL_CODE_PATH, check=True)
        
        self.logger.info("[SafeSpeech] Step 2: Generating BERT for surrogate model...")
        subprocess.run([
            sys.executable, 'bert_gen.py',
            '--dataset', 'LibriTTS',
            '--mode', 'clean'
        ], cwd=ORIGINAL_CODE_PATH, check=True)
        
        self.logger.info("[SafeSpeech] Step 3: Generating adversarial perturbation...")
        wavlm_path = 'microsoft/wavlm-base-plus'
        subprocess.run([
            sys.executable, 'protect.py',
            '--dataset', 'LibriTTS',
            '--model', 'BERT_VITS2',
            '--mode', self.config['mode'],
            '--checkpoint-path', self.config['checkpoint_path'],
            '--epsilon', str(self.config['epsilon']),
            '--perturbation-epochs', str(self.config['perturbation_epochs']),
            '--batch-size', str(self.config['batch_size']),
            '--gpu', str(self.device.index) if self.device.type == 'cuda' else "-1",
            '--wavlm-path', wavlm_path
        ], cwd=ORIGINAL_CODE_PATH, check=True)

        self.logger.info(f"[SafeSpeech] Step 4: Saving protected audio to {output_path}...")
        subprocess.run([
            sys.executable, 'save_audio.py',
            '--dataset', 'LibriTTS',
            '--mode', self.config['mode'],
            '--batch-size', str(self.config['batch_size'])
        ], cwd=ORIGINAL_CODE_PATH, check=True)
        
        self.logger.info("[SafeSpeech] Step 5: Running ASR on protected audio for adversary training...")
        subprocess.run([
            sys.executable, 'asr.py',
            '--dataset', 'LibriTTS',
            '--mode', self.config['mode'],
            '--gpu', str(self.device.index) if self.device.type == 'cuda' else "-1"
        ], cwd=ORIGINAL_CODE_PATH, check=True)

        self.logger.info("[SafeSpeech] Step 6: Generating BERT for adversary model...")
        subprocess.run([
            sys.executable, 'bert_gen.py',
            '--dataset', 'LibriTTS',
            '--mode', self.config['mode']
        ], cwd=ORIGINAL_CODE_PATH, check=True)
        
        protected_train_filelist = os.path.join(
            ORIGINAL_CODE_PATH, f"filelists/libritts_train_asr.txt.cleaned"
        )
        
        os.chdir(self.original_cwd)
        return protected_train_filelist