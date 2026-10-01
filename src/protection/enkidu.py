"""Enkidu universal spectral perturbation with run-scoped noise artifacts."""
from pathlib import Path
import random

import torch
import torchaudio
from torch import Tensor, nn

from src.protection.base_protector import BaseProtector
from src.models.enkidu_model import WienerFilter


class EnkiduProtector(BaseProtector):

    def __init__(
            self,
            model_config,
            **kwargs
    ):
        """
        model_config, protect_config = config, device, logger

            mask_ratio: float number to control the mask place on source waveform, when only for adding noise, should using default value
            random_offset: bool to control random initialize the offset or not


        """
        config = kwargs['config']
        if config.batch_size != 1:
            raise ValueError('Enkidu requires batch_size=1; flattening a batch would merge references')
        if config.sampling_rate != 16000:
            raise ValueError('Enkidu speaker encoder requires sampling_rate=16000')
        self.model_config = model_config
        super().__init__(**kwargs)
        # remove to class init
        self.mask_ratio = self.config.mask_ratio #   0.0,  # for augmentation
        self.random_offset  = self.config.random_offset # False,  # for augmentation
        self.noise_smooth = self.config.noise_smooth #  True,
        self.lambda_perceptual = self.config.lambda_perceptual # 0.1,
        self.perturbation_epochs= self.config.perturbation_epochs
        self.batch_size = self.config.batch_size
        self.learning_rate = self.config.learning_rate
        self.decay = self.config.lr_decay
        self.alpha = self.config.alpha
        self.augmentation = self.config.augmentation
        self.noise_level = self.config.noise_level
        self.frame_length = self.config.frame_length
        self.sampling_rate = self.config.sampling_rate
        self.n_fft = self.config.n_fft
        self.hop_length = self.config.hop_length
        self.win_length = self.config.win_length
        self.eps_loss_weight = self.config.eps_loss_weight
        self.window = torch.hann_window(self.n_fft).to(self.device)
        self.loss_func = nn.MSELoss()

        self.noises = {}
        if Path(model_config.checkpoint_path).exists():
            source= model_config.checkpoint_path
        else:
            source= model_config.source
        from speechbrain.inference import SpeakerRecognition
        self.model = SpeakerRecognition.from_hparams(source=source, run_opts={"device": self.device},savedir=model_config.checkpoint_path)
        self.model.to(self.device)


    @staticmethod
    def extract_embedding(x: Tensor, model, device: str | torch.device = 'cuda:0') -> Tensor:
        """
        Extract the embedding from the input waveform tensor.
        """
        x = x.reshape(1, -1).to(device)
        embedding = model.encode_batch(x).reshape(1, -1).to(device)

        return embedding

    @staticmethod
    def perceptual_loss(
        clean_waveform: Tensor,
        noisy_waveform: Tensor,
        eps: float = 1e-8
    ) -> Tensor:
        diff = noisy_waveform - clean_waveform
        num = (diff ** 2).mean(dim=1)
        den = (clean_waveform ** 2).mean(dim=1) + eps
        return (num / den).mean()

    def _protect_losses(
        self,
        clean_waveform: Tensor,
        noisy_waveform: Tensor
    ) -> Tensor:
        clean_waveform= clean_waveform.squeeze(0)
        clean_embedding = self.extract_embedding(clean_waveform, self.model, self.device)
        noisy_embedding = self.extract_embedding(noisy_waveform, self.model, self.device)
        loss_main = 1 - self.loss_func(clean_embedding, noisy_embedding)
        loss_perceptual = self.perceptual_loss(clean_waveform, noisy_waveform, self.eps_loss_weight)
        loss_total=  loss_main + self.lambda_perceptual * loss_perceptual
        loss_items = {
            "loss_main": f"{loss_main.item():.6f}",
            "loss_perceptual": f"{loss_perceptual.item():.6f}",
            "loss_total": f"{loss_total.item():.6f}",
        }
        return loss_total, loss_items

    def perturb(
            self,
            batch_data,
            noise_real,
            noise_imag,
        ):
        """
        Adding frequential noise to the source waveform

        Argument:
            source_waveform: shape [1, N] tensor ready to be tiled noise
            noise_real: real number of the frequential noise
            noise_imag: imag number of the frequential noise
        """
        batch_data = batch_data.flatten().to(self.device)
        noise_real = noise_real.to(self.device)
        noise_imag = noise_imag.to(self.device)

        source_stft = torch.stft(
            input=batch_data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            window=self.window
        ).unsqueeze(0).to(self.device)

        # get real and imag part of the stft
        stft_real = source_stft.real
        stft_imag = source_stft.imag
        if self.frame_length > source_stft.size(2):
            stft_real[:, :, :source_stft.size(2)] += noise_real[:, :, :source_stft.size(2)] * self.noise_level
            stft_imag[:, :, :source_stft.size(2)] += noise_imag[:, :, :source_stft.size(2)] * self.noise_level
        else:

            patch_num = source_stft.shape[-1] // self.frame_length
            offset = random.randint(0, source_stft.shape[-1] % self.frame_length) if self.random_offset else 0

            rand_mask = torch.rand((patch_num,)) >= self.mask_ratio

            if self.mask_ratio > 0 and rand_mask.sum() == 0:
                rand_idx = random.randint(0, patch_num - 1)
                rand_mask[rand_idx] = True

            for idx in range(patch_num):
                lower = offset + idx * self.frame_length
                upper = offset + (idx + 1) * self.frame_length
                if rand_mask[idx]:
                    stft_real[:, :, lower:upper] += noise_real * self.noise_level
                    stft_imag[:, :, lower:upper] += noise_imag * self.noise_level

        # merge to complex spectrum
        stft_noisy = torch.complex(stft_real, stft_imag)

        if self.noise_smooth:
            stft_noisy = WienerFilter()(stft_noisy)

        refined_waveform = torch.istft(
            stft_noisy,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=batch_data.size(-1)
        ).reshape(1, -1).to(self.device)
        refined_waveform = refined_waveform.clamp(-1.0, 1.0)

        return refined_waveform

    def generate_perturbations(self, save_path=None):

        noise_path = Path(save_path).expanduser() if save_path is not None else Path(f"{self.protect_method}.noise")
        if not noise_path.is_absolute():
            noise_path = self.output_dir / noise_path
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        speaker_ids = [self.dataset_config.speaker_id] if self.dataset_config.speaker_id is not None else []  
        if len(speaker_ids) == 0:
            speaker_ids = self.speaker_data.speakers_ids
        for sid in speaker_ids:
            self.logger.info(f"Generate perturbations for speaker {sid}")
            universal_noise_real = torch.randn((1, self.n_fft // 2 + 1, self.frame_length), requires_grad=True,
                                               device=self.device)
            universal_noise_imag = torch.randn((1, self.n_fft // 2 + 1, self.frame_length), requires_grad=True,
                                               device=self.device)
            self.optimizer = torch.optim.Adam(params=[universal_noise_real, universal_noise_imag],
                                              lr=self.learning_rate, weight_decay=self.decay)
            train_loader = self.speaker_data.speaker_dataloaders[sid]

            for epoch in range(self.perturbation_epochs):
                for batch_idx, batch_data in enumerate(train_loader):  # todo batch_size is expected to be 1
                    
                    batch_data = batch_data.wav
                    batch_data = batch_data.to(self.device)
                    perturbed_audio = self.perturb(batch_data, universal_noise_real, universal_noise_imag)
                    loss_total, loss_items = self._protect_losses(batch_data, perturbed_audio)
                    loss_total.backward()
                    # print(f"Batch {batch_idx}: Loss {loss_items}")
                self.optimizer.step()

            noise = {
                'real': universal_noise_real.detach().cpu(),
                'imag': universal_noise_imag.detach().cpu()
            }
            self.noises[sid] = noise
        torch.save(self.noises, noise_path)
        self.logger.info(f"Saved noises to {noise_path}")

    def save_protected_audio(self):
        for sid in self.speaker_data.speakers_ids:
            self.save_protected_audio_by_speaker(sid)

    def save_protected_audio_by_speaker(self, speaker_id):
        """Write 16 kHz references without modifying source manifests."""
        out_dir = self.output_dir / f"{speaker_id}/"
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.noises is None:
            noise_path = self.output_dir / f"{self.protect_method}.noise"
            self.noises = torch.load(noise_path, map_location="cpu", weights_only=True)

        noises= self.noises[speaker_id]
        noise_real= noises['real']
        noise_imag= noises['imag']

        total_saved = 0
        train_loader = self.speaker_data.speaker_dataloaders[speaker_id]

        for batch_idx, batch in enumerate(train_loader):
            paths = batch.path_out
            wav_len= batch.wav_len
            wav_clean_batch= batch.wav
            perturbed_batch= self.perturb(wav_clean_batch, noise_real, noise_imag)
            perturbed_batch= perturbed_batch.detach().cpu()


            for i, p_wav_i in enumerate(perturbed_batch):
                wav_len_i = wav_len[i]
                p_wav_i = p_wav_i[:wav_len_i]

                if paths and i < len(paths):
                    orig_path = str(paths[i])
                    orig_stem = Path(orig_path).stem
                else:
                    # 少数情况下没有 paths（尽量避免）
                    orig_path = f"__no_path__/idx_{batch_idx}_{i}"
                    orig_stem = f"idx_{batch_idx}_{i}"

                new_name = f"{orig_stem}.wav"
                out_path = out_dir / new_name
                torchaudio.save(str(out_path), p_wav_i.unsqueeze(0), self.sampling_rate)
                
                total_saved += 1


        self.logger.info(
            f"Saved {total_saved} protected audios to {out_dir}. "
        )

    def protect(self):
        self.generate_perturbations()
        self.save_protected_audio()