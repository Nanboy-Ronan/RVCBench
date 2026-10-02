import os
import torch
import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from torch_stoi import NegSTOILoss
from tqdm import tqdm

import sys
sys.path.append("bert_vits2/")

import bert_vits2.commons as commons
import bert_vits2.utils as utils
from bert_vits2.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from bert_vits2.losses import WavLMLoss
from bert_vits2.mel_processing import mel_spectrogram_torch, spectrogram_torch
from bert_vits2.text.symbols import symbols
from toolbox import build_models_noise, build_optims


def get_args():
    parser = argparse.ArgumentParser(description='The protection code of SafeSpeech.')

    parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
    parser.add_argument('--model', type=str, default='BERT_VITS2', help='the surrogate model')
    parser.add_argument('--batch-size', type=int, default=27, help='the batch size of protection')
    parser.add_argument('--gpu', type=int, default=0, help='use which gpu')
    parser.add_argument('--mode', type=str, default="SPEC", choices=["SPEC", "SafeSpeech"], help='the protection mode')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints', help='the storing path of the checkpoints')
    parser.add_argument('--epsilon', type=int, default=8, help='the perturbation radius')
    parser.add_argument('--perturbation-epochs', type=int, default=200, help='the iteration numbers of the noise')
    parser.add_argument('--wavlm-path', type=str, help='the path to the WavLM model')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model_name = args.model
    dataset_name = args.dataset
    mode = args.mode

    gpu = int(args.gpu)
    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"

    config_path = f"bert_vits2/configs/{dataset_name.lower()}_{model_name.lower()}.json"
    hps = utils.get_hparams_from_file(config_path)
    hps.train.batch_size = int(args.batch_size)

    checkpoint_folder = args.checkpoint_path
    os.makedirs(checkpoint_folder, exist_ok=True)
    hps.model_dir = f"{checkpoint_folder}/base_models"
    assert os.listdir(hps.model_dir) != 4

    # Build dataset to be protected
    seed = hps.train.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset,
                              num_workers=4,
                              shuffle=False,
                              collate_fn=collate_fn,
                              batch_size=hps.train.batch_size,
                              pin_memory=True,
                              drop_last=False)

    # Build model and optimizers for perturbation generation
    nets = build_models_noise(hps, device)
    net_g, net_d, net_wd, net_dur_disc = nets

    optims = build_optims(hps, nets)
    optim_g, optim_d, optim_wd, optim_dur_disc = optims

    dur_resume_lr = hps.train.learning_rate
    wd_resume_lr = hps.train.learning_rate

    _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
        net_dur_disc,
        optim_dur_disc,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_dur_disc.param_groups[0].get("initial_lr"):
        optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

    _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
        net_g,
        optim_g,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
        net_d,
        optim_d,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_g.param_groups[0].get("initial_lr"):
        optim_g.param_groups[0]["initial_lr"] = g_resume_lr
    if not optim_d.param_groups[0].get("initial_lr"):
        optim_d.param_groups[0]["initial_lr"] = d_resume_lr

    epoch_str = max(epoch_str, 1)
    global_step = int(utils.get_steps(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")))

    _, optim_wd, wd_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "WD_*.pth"),
        net_wd,
        optim_wd,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_wd.param_groups[0].get("initial_lr"):
        optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr

    wl = WavLMLoss(
        args.wavlm_path,
        net_wd,
        hps.data.sampling_rate,
        hps.model.slm.sr,
    ).to(device)

    # Forbidden the gradient when perturbation generation
    for param in net_g.parameters():
        param.requires_grad = False
    for param in net_d.parameters():
        param.requires_grad = False
    for param in net_dur_disc.parameters():
        param.requires_grad = False
    for param in net_wd.parameters():
        param.requires_grad = False
    for param in wl.parameters():
        param.requires_grad = False
    
    noises = [None] * len(train_loader)
    max_epoch = int(args.perturbation_epochs)
    epsilon = int(args.epsilon) / 255
    alpha = epsilon / 10

    weight_alpha = 0.05
    weight_beta = 10

    # Generate the perturbation by batch format.
    for batch_index, batch_data in enumerate(train_loader):
        loss, noises[batch_index] = perturb(hps, net_g, batch_data, epsilon, alpha, max_epoch,
                                            [weight_alpha, weight_beta], mode, device)

        print(f"Batch {batch_index}: Loss {loss}")

        torch.cuda.empty_cache()
    
    # Save the generated perturbation
    checkpoint_path = args.checkpoint_path
    os.makedirs(f"{checkpoint_path}/{dataset_name}/noises/", exist_ok=True)
    noise_save_path = f"{checkpoint_path}/{dataset_name}/noises/{model_name}_{mode}_{dataset_name}.noise"
    torch.save(noises, noise_save_path)
    print(f"Save the noise to {noise_save_path}!")


def perturb(hps, net_g, batch_data, epsilon, alpha, max_epoch, weights, mode, device):
    '''
        The perturbation generation function based on settings of SafeSpeech.
        Output: loss items for presentation and noise for protection.
    '''
    weight_alpha, weight_beta = weights

    text, text_len, spec, spec_len, wav, wav_len, speakers, \
        tone, language, bert, ja_bert, en_bert = batch_data
    text, text_len = text.to(device), text_len.to(device)

    wav, wav_len = wav.to(device), wav_len.to(device)
    speakers, tone, language = speakers.to(device), tone.to(device), language.to(device)
    bert, ja_bert, en_bert = bert.to(device), ja_bert.to(device), en_bert.to(device)
    noise = torch.zeros(wav.shape).to(device)

    ori_wav = wav
    p_wav = Variable(ori_wav.data + noise, requires_grad=True)
    p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)

    opt_noise = torch.optim.SGD([p_wav], lr=5e-2)

    net_g.train()
    for iteration in tqdm(range(max_epoch)):

        opt_noise.zero_grad()

        p_spec, spec_len = get_spec(p_wav, wav_len, hps.data)

        wav_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_, logw_sdp), g \
            = net_g(text, text_len, p_spec, spec_len, speakers, \
                    tone, language, bert, ja_bert, en_bert, is_clip = False)

        torch.manual_seed(hps.train.seed)
        random_z = torch.randn(wav_hat.shape).to(device)

        # The pivotal objective, i.e, the mel loss
        loss_mel = compute_reconstruction_loss(hps, p_wav, wav_hat)

        # Speech PErturbative Concealment based on KL-divergence
        loss_kl = compute_kl_divergence(hps, wav_hat, random_z)
        loss_nr = compute_reconstruction_loss(hps, wav_hat, random_z)

        if mode == "SPEC":
            loss = loss_mel + weight_beta * (loss_nr + loss_kl)
            loss_items = {
                "loss_mel": f"{loss_mel.item():.6f}", 
                "loss_nr": f"{loss_nr.item():.6f}", 
                "loss_kl": f"{loss_kl.item():.6f}"
            }
        elif mode == "SafeSpeech":
            # Conbining SPEC with perceptual loss for human perception
            loss_perceptual = compute_perceptual_loss(hps, p_wav, wav)

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

        # Update the perturbation
        noise = alpha * torch.sign(grad) * -1.
        p_wav = Variable(p_wav.data + noise, requires_grad=True)
        noise = torch.clamp(p_wav.data - ori_wav.data, min=-epsilon, max=epsilon)
        p_wav = Variable(ori_wav.data + noise, requires_grad=True)
        p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)


    return loss_items, noise


def get_spec(waves, waves_len, hps):
    '''
        Convert the waveforms to mel-spectrogram
    '''
    spec_np = []
    spec_lengths = torch.LongTensor(len(waves))

    device = waves.device
    for index, wave in enumerate(waves):
        audio_norm = wave[:, :waves_len[index]]
        spec = spectrogram_torch(audio_norm,
                                 hps.filter_length, hps.sampling_rate,
                                 hps.hop_length, hps.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        spec_np.append(spec)
        spec_lengths[index] = spec.size(1)

    max_spec_len = max(spec_lengths)
    spec_padded = torch.FloatTensor(len(waves), spec_np[0].size(0), max_spec_len)
    spec_padded.zero_()

    for i, spec in enumerate(waves):
        spec_padded[i][:, :spec_lengths[i]] = spec_np[i]

    return spec_padded.to(device), spec_lengths.to(device)


def compute_kl_divergence(hps, x_hat, z):
    '''
        Return the KL-divergence loss of the input distributions.
    '''
    x_mel = mel_spectrogram_torch(
        x_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    z_mel = mel_spectrogram_torch(
        z.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )

    p_log = F.log_softmax(x_mel, dim=-1)
    q = F.softmax(z_mel, dim=-1)

    kl_divergence = F.kl_div(p_log, q, reduction="batchmean")

    return kl_divergence


def compute_reconstruction_loss(hps, wav, wav_hat):
    '''
        Return the mel loss of the real and synthesized speech.
    '''
    wav_mel = mel_spectrogram_torch(
        wav.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    wav_hat_mel = mel_spectrogram_torch(
        wav_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    loss_mel_wav = F.l1_loss(wav_mel, wav_hat_mel) * hps.train.c_mel

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


def compute_perceptual_loss(hps, p_wav, wav):
    '''
        Return the proposed perceptual loss  of the clean and protected speech
    '''
    loss_stoi = compute_stoi(hps.data.sampling_rate, wav, p_wav)
    loss_stft = compute_stft(wav.squeeze(1), p_wav.squeeze(1))
    loss_perceptual = loss_stoi + loss_stft

    return loss_perceptual


if __name__ == "__main__":
    main()
