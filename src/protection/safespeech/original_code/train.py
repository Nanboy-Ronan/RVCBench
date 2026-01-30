import os
import time
import torch
import argparse
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.append("bert_vits2/")

import bert_vits2.commons as commons
import bert_vits2.utils as utils
from bert_vits2.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from bert_vits2.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    WavLMLoss,
)
from bert_vits2.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from bert_vits2.text.symbols import symbols
from toolbox import build_models, build_optims, build_schedulers


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # If encountered training problem,please try to disable TF32.
)
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # Not available if torch version is lower than 2.0
global_step = 0


def get_args():
    parser = argparse.ArgumentParser(description="The fine-tuning code of SafeSpeeh.")

    parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
    parser.add_argument('--model', type=str, default='BERT_VITS2', help='the surrogate model')
    parser.add_argument('--batch-size', type=int, default=64, help='the batch size of protected and training')
    parser.add_argument('--gpu', type=int, default=0, help='use which gpu')
    parser.add_argument('--random-seed', type=int, default=1234, help='random seed')
    parser.add_argument('--mode', type=str, default="clean", choices=["clean", "SPEC", "SafeSpeech"], 
                        help='the fine-tuning mode')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints', help='the storing path of the checkpoints')

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
    hps = utils.get_hparams_from_file(config_path=config_path)

    batch_size = int(args.batch_size)
    print(f"The batch size is set as {batch_size} now.")
    # assert batch_size == 64
    hps.train.batch_size = batch_size

    if mode != "clean":
        hps.data.training_files = f"filelists/{dataset_name.lower()}_train_asr.txt.cleaned"
    
    checkpoint_folder = args.checkpoint_path
    os.makedirs(checkpoint_folder, exist_ok=True)
    hps.model_dir = f"{checkpoint_folder}/base_models"
    assert os.listdir(hps.model_dir) != 4

    # Builde data for fine-tuning
    global global_step
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=hps.train.batch_size,
        drop_last=False
    )

    # Build model and optimizers
    models = build_models(hps, device)
    net_g, net_d, net_wd, net_dur_disc = models

    optims = build_optims(hps, models)
    optim_g, optim_d, optim_wd, optim_dur_disc = optims

    # Loading the pretrained checkpoint of model and optimizers
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
    

    schedulers = build_schedulers(hps, optims, epoch_str)
    scheduler_g, scheduler_d, scheduler_wd, scheduler_dur_disc = schedulers

    scaler = GradScaler(enabled=hps.train.bf16_run)

    wl = WavLMLoss(
        hps.model.slm.model,
        net_wd,
        hps.data.sampling_rate,
        hps.model.slm.sr,
    ).to(device)


    # Begin to fine-tuning!
    start_time = time.time()
    for epoch in range(1, hps.train.epochs + 1):
        loss = train(
            hps,
            [net_g, net_d, net_dur_disc, net_wd, wl],
            [optim_g, optim_d, optim_dur_disc, optim_wd],
            train_loader,
            scaler,
            device
        )

        loss_gen_all, loss_disc_all, loss_dur_disc_all, loss_slm = loss
        
        scheduler_g.step()
        scheduler_d.step()
        scheduler_wd.step()
        scheduler_dur_disc.step()

        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))
        print(f"[{formatted_time}] Epoch {epoch}: G {loss_gen_all:.6f}, D {loss_disc_all:.6f} "
              f"Dur {loss_dur_disc_all:.6f}, Sim {loss_slm:.6f}")
    
    os.makedirs(f"{checkpoint_folder}/{dataset_name}", exist_ok=True)
    save_path = f"{checkpoint_folder}/{dataset_name}/{model_name}_{mode}_{dataset_name}_{epoch}.pth"
    torch.save(net_g.state_dict(), save_path)


def train(hps, nets, optims, train_loader, scaler, device):
    '''
        Input:
            hps: The hyperparameter dict
            nets: Five model used for fine-tuning
            optims: The optimizers
            train_loader: The dataset for model training
            device: which device is used for this code

        Return:
            loss_items: The losses of backward and training.
    '''
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims

    global global_step

    net_g.train()
    net_d.train()
    net_wd.train()
    net_dur_disc.train()

    for batch_idx, batch in enumerate(train_loader):
        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, \
        tone, language, bert, ja_bert, en_bert = batch

        if net_g.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.mas_noise_scale_initial
                - net_g.noise_scale_delta * global_step
            )
            net_g.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
            
        x, x_lengths = x.to(device, non_blocking=True), x_lengths.to(device, non_blocking=True)
        spec, spec_lengths = spec.to(device, non_blocking=True), spec_lengths.to(device, non_blocking=True)
        y, y_lengths = y.to(device, non_blocking=True), y_lengths.to(device, non_blocking=True)
        speakers = speakers.to(device, non_blocking=True)
        tone = tone.to(device, non_blocking=True)
        language = language.to(device, non_blocking=True)
        bert = bert.to(device, non_blocking=True)
        ja_bert = ja_bert.to(device, non_blocking=True)
        en_bert = en_bert.to(device, non_blocking=True)

        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_, logw_sdp),
                g,
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                ja_bert,
                en_bert,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                    g.detach(),
                )
                y_dur_hat_r_sdp, y_dur_hat_g_sdp = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw_sdp.detach(),
                    g.detach(),
                )
                y_dur_hat_r = y_dur_hat_r + y_dur_hat_r_sdp
                y_dur_hat_g = y_dur_hat_g + y_dur_hat_g_sdp
                with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                # torch.nn.utils.clip_grad_norm_(
                #     parameters=net_dur_disc.parameters(), max_norm=100
                # )
                grad_norm_dur = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            loss_slm = wl.discriminator(
                y.detach().squeeze(), y_hat.detach().squeeze()
            ).mean()

        optim_wd.zero_grad()
        scaler.scale(loss_slm).backward()
        scaler.unscale_(optim_wd)
        # torch.nn.utils.clip_grad_norm_(parameters=net_wd.parameters(), max_norm=200)
        grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
        scaler.step(optim_wd)

        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
                _, y_dur_hat_g_sdp = net_dur_disc(hidden_x, x_mask, logw_, logw_sdp, g)
                y_dur_hat_g = y_dur_hat_g + y_dur_hat_g_sdp
            with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                loss_dur = torch.sum(l_length.float())

                # Compute the mel loss
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                loss_lm = wl(y.detach().squeeze(), y_hat.squeeze()).mean()
                loss_lm_gen = wl.generator(y_hat.squeeze())

                loss_gen_all = (
                    loss_gen
                    + loss_fm
                    + loss_mel
                    + loss_dur
                    + loss_kl
                    + loss_lm
                    + loss_lm_gen
                )
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        global_step += 1

    return loss_gen_all.item(), loss_disc_all.item(), loss_dur_disc_all.item(), loss_slm.item()


if __name__ == "__main__":
    main()
