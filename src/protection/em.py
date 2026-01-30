from src.protection.base_protector import BaseProtector
import torch
from src.datasets.mel_preprocessing import *
import src.utils.commons as commons
from torch.autograd import Variable
from pathlib import Path
from src.losses.bertvits2_loss import (
    compute_reconstruction_loss,
    kl_loss,
    feature_loss,
    generator_loss,
)
from tqdm import tqdm
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


class EMProtector(BaseProtector):

    def __init__(
        self,
        model_config,
        # dataset_config,
        # logger,
        **kwargs,
    ):
        """
        model_config, protect_config = config, device, logger
        """
        # self.dataset_config = dataset_config
        self.model_config = model_config
        # self.logger = logger
        super().__init__(**kwargs)

        self.noises = {}
        model_name = getattr(models, self.model_config.name + "Wrapper")
        self.model = model_name(
            self.config, self.model_config, self.dataset_config, logger=self.logger
        )
        self.model.to(self.device)

    # =============== 扰动生成（PGD/SafeSpeech） ===============
    @torch.no_grad()
    def _get_spec(self, waves, waves_len):
        device = waves.device
        spec_list, spec_lengths = [], torch.LongTensor(len(waves))
        for i, wave in enumerate(waves):
            audio_norm = wave[:, : waves_len[i]]
            spec = spectrogram_torch(
                audio_norm,
                self.dataset_config.filter_length,
                self.dataset_config.sampling_rate,
                self.dataset_config.hop_length,
                self.dataset_config.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            spec_list.append(spec)
            spec_lengths[i] = spec.size(1)
        max_len = int(spec_lengths.max())
        spec_pad = torch.zeros(len(waves), spec_list[0].size(0), max_len, device=device)
        for i in range(len(waves)):
            spec_pad[i, :, : spec_lengths[i]] = spec_list[i]
        return spec_pad, spec_lengths.to(device)

    def _mel(self, wav):
        d = self.dataset_config
        return mel_spectrogram_torch(
            wav.squeeze(1).float(),
            d.filter_length,
            d.n_mel_channels,
            d.sampling_rate,
            d.hop_length,
            d.win_length,
            d.mel_fmin,
            d.mel_fmax,
        )

    def perturb(self, batch_data):
        """
        生成扰动（PGD / I-FGSM 风格）
        """
        # # 模型只做前向，grad 只在 p_wav 上
        # self.model.model.eval()
        # self.model.model.zero_grad()

        weight_alpha, weight_beta = self.config.weight_alpha, self.config.weight_beta
        epsilon = self.config.epsilon
        alpha = epsilon / 10
        max_epoch = self.config.perturbation_epochs

        # 移动到 device
        batch_data = batch_data.to(self.device)

        # 原始语音，不参与梯度
        ori_wav = batch_data.wav.detach()  # 不要 clone，detach 就够了
        noise = torch.zeros_like(ori_wav, device=self.device)  # 不需要 requires_grad

        loss_items = {}

        for _ in tqdm(range(max_epoch)):
            # 构造当前对抗语音，并限制在 [-1, 1]
            p_wav = (ori_wav + noise).clamp(-1.0, 1.0).detach()
            p_wav.requires_grad_(True)

            # 把对抗语音装回 batch
            batch_data.wav = p_wav

            # 计算谱
            p_spec, spec_len = self._get_spec(batch_data.wav, batch_data.wav_len)

            (
                wav_hat,
                ids_slice,
                (
                    l_length,
                    z_p,
                    logs_q,
                    m_p,
                    logs_p,
                    z_mask,
                    fmap_r,
                    fmap_g,
                    wav_d_hat_g,
                ),
            ) = self.model.generate(
                input=batch_data,
                p_spec=p_spec,
                spec_len=spec_len,
                mode=self.config.mode,
            )

            # 固定随机种子（如果你确实想每轮一样的 random_z）
            torch.manual_seed(self.config.seed)
            random_z = torch.randn_like(wav_hat).to(self.device)

            # ====== SPEC 主目标 ======
            if ids_slice is not None:
                p_wav_slice = commons.slice_segments(
                    p_wav,
                    ids_slice * self.dataset_config.hop_length,
                    self.config.segment_size,
                )
            else:
                p_wav_slice = p_wav

            # 5. 计算 Loss
            loss_mel = compute_reconstruction_loss(
                self.dataset_config, p_wav_slice, wav_hat, c_mel=self.config.c_mel
            )
            loss_dur = torch.sum(l_length.float())
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.config.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(wav_d_hat_g)

            loss = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            # 记录 Loss 用于显示
            loss_items = {
                "loss_mel": f"{loss_mel.item():.6f}",
                "loss_fm": f"{loss_fm.item():.6f}",
                "loss_gen": f"{loss_gen.item():.6f}",
                "loss_dur": f"{loss_dur.item():.6f}",
                "loss_kl": f"{loss_kl.item():.6f}",
                "total_loss": f"{loss.item():.6f}",
            }

            # 反向，只需要 p_wav 的梯度
            self.model.model.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                grad_sign = p_wav.grad.sign()

                # 标准 PGD / I-FGSM 更新：在噪声空间更新，然后投影回 ε-球内
                noise = noise - alpha * grad_sign
                noise = noise.clamp(-epsilon, epsilon)

                # 再确保最终波形在 [-1, 1] 范围内
                adv_wav = (ori_wav + noise).clamp(-1.0, 1.0)
                # 重新对齐 noise，避免数值漂移
                noise = adv_wav - ori_wav

            # 下一轮会重新用 (ori_wav + noise) 构造 p_wav，所以这里不需要再手动改 p_wav

        # 不要随便 del batch_data，外面可能还要用
        # del p_wav, ori_wav, batch_data
        return loss_items, noise

    def generate_perturbations(self):
        noise_path = self.output_dir / f"{self.protect_method}.noise"
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        speaker_ids = (
            [self.dataset_config.speaker_id]
            if self.dataset_config.speaker_id is not None
            else []
        )
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
