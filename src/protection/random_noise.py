import sys
import importlib
from types import SimpleNamespace
from src.protection.base_protector import BaseProtector
import torch
from src.datasets.data_utils import TextAudioSpeakerDataset, TextAudioSpeakerCollate
from transformers import TrainingArguments
from src.datasets.mel_preprocessing import *
import src.utils.commons as commons
import os
from torch.autograd import Variable
from pathlib import Path
from torch.utils.data import DataLoader
import soundfile as sf
from src.losses.bertvits2_loss import compute_reconstruction_loss, compute_perceptual_loss, compute_kl_divergence, kl_loss, feature_loss, generator_loss
from tqdm import tqdm
from hydra.utils import to_absolute_path
from src import models
from copy import deepcopy

# ===================== Filelist 小工具 =====================

def _read_lines(p: Path):
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _write_lines(p: Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _split_line(ln: str):
    # audio_path|speaker|language|norm_text|phones|tones|word2ph
    parts = ln.split("|")
    if len(parts) != 7:
        raise ValueError(f"Malformed filelist line (expected 7 fields): {ln}")
    return parts  # (audio_path, speaker, language, text, phones, tones, word2ph)


def _sig(language, text, phones, tones, word2ph):
    # 用于跨路径匹配（旧命名也能对上）
    return "|".join([language, text, phones, tones, word2ph])


class GRNoiseProtector(BaseProtector):

    def __init__(
            self,
            model_config,
            # logger,
            **kwargs
    ):
        """
        model_config, protect_config = config, device, logger
        """
        # self.dataset_config = dataset_config
        self.model_config = model_config
        # self.logger = logger
        super().__init__(**kwargs)

        self.noises = {}
        self.train_loader = None

    def perturb(self, batch_data):
        """
        生成扰动（随机噪声版）
        使用 epsilon 作为噪声标准差：noise ~ N(0, epsilon^2)
        返回: (loss_items, noise)，其中 noise 形状与 batch_data.wav 一致
        """
        batch_data = batch_data.to(self.device)
        wav = batch_data.wav  # 期望形状 [B, 1, T] 或 [B, T]
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        epsilon =self.config.epsilon
        noise = torch.randn_like(wav, device=self.device) * self.config.epsilon
        loss_items = {"loss_random_std": f"{epsilon:.6e}"}
        return loss_items, noise

    def generate_perturbations(self):
        noise_path = self.output_dir / f"{self.protect_method}.noise"
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        speaker_ids = [self.dataset_config.speaker_id] if self.dataset_config.speaker_id is not None else [] # TODO careful check it

        if len(speaker_ids) == 0:
            speaker_ids = self.speaker_data.speakers_ids

        for sid in speaker_ids:
            self.logger.info(f"Generate perturbations for speaker {sid}")
            train_loader = self.speaker_data.speaker_dataloaders[sid]
            noise = [None] * len(train_loader)
            for batch_idx, batch_data in enumerate(train_loader):
                loss_item, noise[batch_idx] = self.perturb(batch_data)
            self.noises[sid] = noise

        torch.save(self.noises, noise_path)
        self.logger.info(f"Saved noises to {noise_path}")

    def protect(self):
        self.generate_perturbations()
        self.save_protected_audio()