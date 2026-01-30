"""Lightweight audio-only dataset loader for denoiser (no text/BERT dependencies)."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.utils.data
import torchaudio
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader


class SimpleAudioDataset(torch.utils.data.Dataset):
    """Loads audio without text processing or BERT embeddings."""

    def __init__(
        self,
        data_conf,
        speaker_id: str,
        sampling_rate: int,
        logger=None,
        audio_paths: Optional[List[Path]] = None,
    ):
        self.logger = logger
        self.speaker_id = speaker_id
        self.sampling_rate = sampling_rate
        self.max_wav_value = float(data_conf.max_wav_value)

        json_file_path = os.path.join(data_conf.root_path, "filelists", f"{speaker_id}.json")
        self.data_root = Path(data_conf.root_path)

        self.audio_paths = []
        self.transcripts = {}

        if audio_paths is not None:
            self.audio_paths = list(audio_paths)
        else:
            if os.path.exists(json_file_path):
                with open(json_file_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for record in records:
                    ori_pth = self.data_root / record.get("ori_pth", "")
                    if ori_pth.exists():
                        self.audio_paths.append(ori_pth)
                        self.transcripts[str(ori_pth.resolve())] = record.get("ori_text", "")

        if logger:
            logger.info(f"[SimpleAudioDataset] Loaded {len(self.audio_paths)} files for speaker {speaker_id}")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        audio, sr = torchaudio.load(audio_path)
        
        if sr != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sampling_rate)
        
        # Only normalize if audio is not already in [-1, 1] range
        # If max value is close to 1, audio is already normalized
        if audio.abs().max() > 1.5:
            audio_norm = audio / self.max_wav_value
        else:
            audio_norm = audio
        
        audio_raw = audio
        
        return {
            "wav": audio_norm,
            "audio_raw": audio_raw,
            "path": audio_path,
        }


class SimpleAudioBatch:
    """Batch container matching the interface expected by denoiser."""
    
    def __init__(self, wav, wav_len, audio_raw, audio_raw_len, path_out):
        self.wav = wav
        self.wav_len = wav_len
        self.audio_raw = audio_raw
        self.audio_raw_len = audio_raw_len
        self.path_out = path_out
    
    def to(self, device):
        self.wav = self.wav.to(device)
        self.wav_len = self.wav_len.to(device)
        self.audio_raw = self.audio_raw.to(device)
        self.audio_raw_len = self.audio_raw_len.to(device)
        return self


def simple_collate_fn(batch):
    """Collate audio samples into a batch."""
    max_wav_len = max(item["wav"].size(1) for item in batch)
    max_audio_raw_len = max(item["audio_raw"].size(1) for item in batch)
    
    batch_size = len(batch)
    wav_padded = torch.zeros(batch_size, 1, max_wav_len)
    audio_raw_padded = torch.zeros(batch_size, 1, max_audio_raw_len)
    wav_lengths = torch.LongTensor(batch_size)
    audio_raw_lengths = torch.LongTensor(batch_size)
    paths = []
    
    for i, item in enumerate(batch):
        wav = item["wav"]
        audio_raw = item["audio_raw"]
        
        wav_padded[i, :, :wav.size(1)] = wav
        wav_lengths[i] = wav.size(1)
        
        audio_raw_padded[i, :, :audio_raw.size(1)] = audio_raw
        audio_raw_lengths[i] = audio_raw.size(1)
        
        paths.append(item["path"])
    
    return SimpleAudioBatch(
        wav=wav_padded,
        wav_len=wav_lengths,
        audio_raw=audio_raw_padded,
        audio_raw_len=audio_raw_lengths,
        path_out=paths,
    )


class SimpleAllSpeakerData:
    """Lightweight dataset wrapper for denoiser (no text/BERT dependencies)."""
    
    def __init__(self, config, dataset_config, logger):
        self.config = config
        self.dataset_config = dataset_config
        self.logger = logger
        
        root_path = Path(to_absolute_path(str(dataset_config.root_path))).resolve()
        self._dataset_root = root_path
        self.dataset_root = root_path
        self._filelists_dir = self._dataset_root / "filelists"
        
        self._speaker_dirs = {}
        if dataset_config.speaker_id is not None:
            speaker = str(dataset_config.speaker_id)
            self.speakers_ids = [speaker]
            speaker_dir = self._dataset_root / speaker
            if speaker_dir.exists():
                self._speaker_dirs[speaker] = speaker_dir
        else:
            filelist_paths = []
            if self._filelists_dir.exists():
                filelist_paths = sorted(self._filelists_dir.glob("*.json"))
            if filelist_paths:
                self.speakers_ids = [p.stem for p in filelist_paths]
            else:
                speaker_dirs = [
                    p for p in self._dataset_root.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                ]
                if speaker_dirs:
                    self.speakers_ids = sorted(p.name for p in speaker_dirs)
                    self._speaker_dirs = {p.name: p for p in speaker_dirs}
                else:
                    self.speakers_ids = ["default"]
                    self._speaker_dirs["default"] = self._dataset_root
        
        self.speaker_datasets = {}
        self.speaker_dataloaders = {}
        
        batch_size = getattr(config, "batch_size", 4)
        num_workers = int(os.environ.get("DATA_LOADER_WORKERS", 0))
        
        for speaker_id in self.speakers_ids:
            audio_paths = None
            speaker_dir = self._speaker_dirs.get(speaker_id)
            if speaker_dir is not None:
                audio_paths = sorted(
                    p for p in speaker_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
                )
            dataset = SimpleAudioDataset(
                dataset_config,
                speaker_id,
                sampling_rate=config.dataset.sampling_rate,
                logger=logger,
                audio_paths=audio_paths,
            )
            self.speaker_datasets[speaker_id] = dataset
            
            try:
                self.speaker_dataloaders[speaker_id] = DataLoader(
                    dataset,
                    num_workers=num_workers,
                    shuffle=False,
                    collate_fn=simple_collate_fn,
                    batch_size=batch_size,
                    pin_memory=False,
                    drop_last=False,
                )
            except PermissionError as exc:
                if num_workers > 0:
                    logger.warning(f"DataLoader failed with num_workers={num_workers}, retrying with 0")
                    self.speaker_dataloaders[speaker_id] = DataLoader(
                        dataset,
                        num_workers=0,
                        shuffle=False,
                        collate_fn=simple_collate_fn,
                        batch_size=batch_size,
                        pin_memory=False,
                        drop_last=False,
                    )
                else:
                    raise
    
    def get_train_files_map(self) -> Dict[str, str]:
        """Return mapping of audio paths to transcripts for fidelity evaluation."""
        transcripts = {}
        for speaker_id, dataset in self.speaker_datasets.items():
            if dataset.transcripts:
                transcripts.update(dataset.transcripts)
            else:
                for path in dataset.audio_paths:
                    transcripts[str(Path(path).resolve())] = ""
        return transcripts
