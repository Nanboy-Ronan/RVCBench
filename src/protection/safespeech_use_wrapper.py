import sys
import importlib
from types import SimpleNamespace

from src.trainers import BertVITS2Trainer
from src.protection.base_protector import BaseProtector
import torch
from src.datasets.data_utils import TextAudioSpeakerDataset, TextAudioSpeakerCollate
from src.utils.commons import latest_checkpoint_path, load_checkpoint
from transformers import TrainingArguments
from src.datasets.mel_preprocessing import *
import torch.nn.functional as F
import os
from torch.autograd import Variable
from pathlib import Path
from torch.utils.data import DataLoader
import soundfile as sf
from src.losses.bertvits2_loss import compute_reconstruction_loss, compute_perceptual_loss, compute_kl_divergence
from tqdm import tqdm
from hydra.utils import to_absolute_path
from src import models
from copy import deepcopy
import src.utils.commons as commons

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


class SafeSpeechProtector(BaseProtector):

    def __init__(
            self,
            model_config,
            **kwargs
    ):
        """
        model_config, protect_config = config, device, logger
        """
        self.model_config = model_config
        super().__init__(**kwargs)

        self.noises = {}
        model_name = getattr(models, self.model_config.name+"Wrapper")
        self.model = model_name(self.config, self.model_config,self.dataset_config, logger=self.logger)
        self.model.to(self.device)

    # =============== 扰动生成（PGD/SafeSpeech） ===============
    def _get_spec(self, waves, waves_len):
        device = waves.device
        spec_list, spec_lengths = [], torch.LongTensor(len(waves))
        for i, wave in enumerate(waves):
            audio_norm = wave[:, :waves_len[i]]
            spec = spectrogram_torch(audio_norm, self.dataset_config.filter_length, self.dataset_config.sampling_rate,
                                     self.dataset_config.hop_length, self.dataset_config.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            spec_list.append(spec)
            spec_lengths[i] = spec.size(1)
        max_len = int(spec_lengths.max())
        spec_pad = torch.zeros(len(waves), spec_list[0].size(0), max_len, device=device)
        for i in range(len(waves)):
            spec_pad[i, :, :spec_lengths[i]] = spec_list[i]
        return spec_pad, spec_lengths.to(device)

    def _mel(self, wav):
        d = self.dataset_config
        return mel_spectrogram_torch(
            wav.squeeze(1).float(), d.filter_length, d.n_mel_channels,
            d.sampling_rate, d.hop_length, d.win_length, d.mel_fmin, d.mel_fmax
        )

    def perturb(self, batch_data):
        """
        生成扰动
        """
        weight_alpha, weight_beta = self.config.weight_alpha, self.config.weight_beta
        alpha = self.config.epsilon / 10
        max_epoch = self.config.perturbation_epochs
        batch_data= batch_data.to(self.device)
        noise = torch.zeros(batch_data.wav.shape).to(self.device)
        ori_wav = deepcopy(batch_data.wav)
        p_wav = Variable(ori_wav.data + noise, requires_grad=True)
        p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)

        opt_noise = torch.optim.SGD([p_wav], lr=self.config.learning_rate, weight_decay=self.config.lr_decay)
        for _ in tqdm(range(max_epoch)):
            opt_noise.zero_grad()
            batch_data.wav = p_wav
            p_spec, spec_len = self._get_spec(batch_data.wav, batch_data.wav_len)
            wav_hat, ids_slice= self.model.generate(input=batch_data,p_spec= p_spec, spec_len= spec_len)
            torch.manual_seed(self.config.seed)
            random_z = torch.randn(wav_hat.shape).to(self.device)
            # SPEC 主目标

            if ids_slice is not None:
                p_wav_slice = commons.slice_segments(p_wav, ids_slice * self.dataset_config.hop_length, self.config.segment_size)
            else:
                p_wav_slice = p_wav
            loss_mel = compute_reconstruction_loss(self.dataset_config, p_wav_slice, wav_hat, c_mel=self.config.c_mel)
            loss_kl = compute_kl_divergence(self.dataset_config, wav_hat, random_z)
            loss_nr = compute_reconstruction_loss(self.dataset_config, wav_hat, random_z, c_mel=self.config.c_mel)

            if self.config.mode == "SPEC":
                loss = loss_mel + weight_beta * (loss_nr + loss_kl)
                loss_items = {
                    "loss_mel": f"{loss_mel.item():.6f}",
                    "loss_nr": f"{loss_nr.item():.6f}",
                    "loss_kl": f"{loss_kl.item():.6f}"
                }
            elif self.config.mode == "SafeSpeech":
                loss_perceptual = compute_perceptual_loss(self.config.sampling_rate, p_wav, ori_wav)
                loss = loss_mel + weight_beta * (loss_nr + loss_kl) + weight_alpha * loss_perceptual
                loss_items = {
                    "loss_mel": f"{loss_mel.item():.6f}",
                    "loss_nr": f"{loss_nr.item():.6f}",
                    "loss_kl": f"{loss_kl.item():.6f}",
                    "loss_perception": f"{loss_perceptual.item():.6f}"
                }
            else:
                raise TypeError("The protective mode is wrong!")

            p_wav.retain_grad = True
            loss.backward()
            grad = p_wav.grad

            # 更新扰动（FGSM/PGD 风格）
            noise = alpha * torch.sign(grad) * -1.
            p_wav = Variable(p_wav.data + noise, requires_grad=True)
            noise = torch.clamp(p_wav.data - ori_wav.data, min=-self.config.epsilon, max=self.config.epsilon)
            p_wav = Variable(ori_wav.data + noise, requires_grad=True)
            p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)

        del p_wav, ori_wav, batch_data
        return loss_items, noise

    def generate_perturbations(self):
        noise_path = self.output_dir / f"{self.protect_method}.noise"
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        speaker_ids = [self.dataset_config.speaker_id] if self.dataset_config.speaker_id is not None else []  
        if len(speaker_ids)==0:
            speaker_ids = self.speaker_data.speakers_ids
        for sid in speaker_ids:
            self.logger.info(f"Generate perturbations for speaker {sid}")
            train_loader = self.speaker_data.speaker_dataloaders[sid]
            noise = [None] * len(train_loader)
            for batch_idx, batch_data in enumerate(train_loader):
                
                loss_item, noise_batch = self.perturb(batch_data)
                noise[batch_idx]= noise_batch.cpu()
                del batch_data
            self.noises[sid] = noise
            torch.cuda.empty_cache()

        torch.save(self.noises, noise_path)
        self.logger.info(f"Saved noises to {noise_path}")


    def protect(self):
        self.generate_perturbations()
        self.save_protected_audio()