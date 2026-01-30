import torch
import torchaudio
from transformers import AutoModel
from torch_stoi import NegSTOILoss
from src.datasets.mel_preprocessing import spectrogram_torch, mel_spectrogram_torch
import torch.nn.functional as F


def compute_kl_divergence(data_config, x_hat, z):
    '''
        Return the KL-divergence loss of the input distributions.
    '''
    x_mel = mel_spectrogram_torch(
        x_hat.squeeze(1).float(),
        data_config.filter_length,
        data_config.n_mel_channels,
        data_config.sampling_rate,
        data_config.hop_length,
        data_config.win_length,
        data_config.mel_fmin,
        data_config.mel_fmax
    )
    z_mel = mel_spectrogram_torch(
        z.squeeze(1).float(),
        data_config.filter_length,
        data_config.n_mel_channels,
        data_config.sampling_rate,
        data_config.hop_length,
        data_config.win_length,
        data_config.mel_fmin,
        data_config.mel_fmax
    )

    p_log = F.log_softmax(x_mel, dim=-1)
    q = F.softmax(z_mel, dim=-1)

    kl_divergence = F.kl_div(p_log, q, reduction="batchmean")

    return kl_divergence


def compute_reconstruction_loss(data_config, wav, wav_hat, c_mel=0.1):
    '''
        Return the mel loss of the real and synthesized speech.
    '''
    wav_mel = mel_spectrogram_torch(
        wav.squeeze(1).float(),
        data_config.filter_length,
        data_config.n_mel_channels,
        data_config.sampling_rate,
        data_config.hop_length,
        data_config.win_length,
        data_config.mel_fmin,
        data_config.mel_fmax
    )
    wav_hat_mel = mel_spectrogram_torch(
        wav_hat.squeeze(1).float(),
        data_config.filter_length,
        data_config.n_mel_channels,
        data_config.sampling_rate,
        data_config.hop_length,
        data_config.win_length,
        data_config.mel_fmin,
        data_config.mel_fmax
    )
    loss_mel_wav = F.l1_loss(wav_mel, wav_hat_mel) * c_mel

    return loss_mel_wav


def compute_stoi(sample_rate, waveforms, perturb_waveforms):
    '''
        Return the STOI loss of the clean and protected speech
    '''
    device = waveforms.device
    stoi_function = NegSTOILoss(sample_rate=sample_rate).to(device)

    loss_stoi = stoi_function(waveforms, perturb_waveforms).mean()
    return loss_stoi


def compute_stft(waveforms, perturb_waveforms):
    '''
        Return the STFT loss with L_2 norm of the clean and protected speech
    '''
    stft_clean = torch.stft(waveforms, n_fft=2048, win_length=2048, hop_length=512, return_complex=False)
    stft_p = torch.stft(perturb_waveforms, n_fft=2048, win_length=2048, hop_length=512, return_complex=False)
    loss_stft = torch.norm(stft_p - stft_clean, p=2)

    return loss_stft


def compute_perceptual_loss(sampling_rate, p_wav, wav):
    '''
        Return the proposed perceptual loss  of the clean and protected speech
    '''
    loss_stoi = compute_stoi(sampling_rate, wav, p_wav)
    loss_stft = compute_stft(wav.squeeze(1), p_wav.squeeze(1))
    loss_perceptual = loss_stoi + loss_stft

    return loss_perceptual


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


class WavLMLoss(torch.nn.Module):
    def __init__(self, model_name_or_path, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model_name_or_path)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(), output_hidden_states=True
        ).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs