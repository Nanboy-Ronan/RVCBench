import torch

from bert_vits2.text.symbols import symbols


def build_models_noise(hps, device):
    '''
        Build model for perturbation genetation.
    '''
    from bert_vits2.models_noise import (
        SynthesizerTrn,
        MultiPeriodDiscriminator,
        DurationDiscriminator,
        WavLMDiscriminator,
    )

    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6

    net_dur_disc = DurationDiscriminator(
        hps.model.hidden_channels,
        hps.model.hidden_channels,
        3,
        0.1,
        gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
    ).to(device)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).to(device)

    for param in net_g.enc_p.bert_proj.parameters():
        param.requires_grad = False
    for param in net_g.enc_p.ja_bert_proj.parameters():
        param.requires_grad = False

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    net_wd = WavLMDiscriminator(
        hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
    ).to(device)

    return net_g, net_d, net_wd, net_dur_disc


def build_models(hps, device):
    '''
        Builde model for fine-tuning [This is the original model without modification.]
    '''

    from bert_vits2.models import (
        SynthesizerTrn,
        MultiPeriodDiscriminator,
        DurationDiscriminator,
        WavLMDiscriminator,
    )
    
    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6

    net_dur_disc = DurationDiscriminator(
        hps.model.hidden_channels,
        hps.model.hidden_channels,
        3,
        0.1,
        gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
    ).to(device)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).to(device)

    for param in net_g.enc_p.bert_proj.parameters():
        param.requires_grad = False
    for param in net_g.enc_p.ja_bert_proj.parameters():
        param.requires_grad = False

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    net_wd = WavLMDiscriminator(
        hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
    ).to(device)

    return net_g, net_d, net_wd, net_dur_disc


def build_optims(hps, nets):
    '''
        Build the optimizers for fine-tuning
    '''
    
    net_g, net_d, net_wd, net_dur_disc = nets
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_wd = torch.optim.AdamW(
        net_wd.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_dur_disc = torch.optim.AdamW(
        net_dur_disc.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    return optim_g, optim_d, optim_wd, optim_dur_disc


def build_schedulers(hps, optims, epoch_str):
    '''
        Build the schedulers for optimizers when fine-tuning.
    '''
    optim_g, optim_d, optim_wd, optim_dur_disc = optims
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_wd = torch.optim.lr_scheduler.ExponentialLR(
        optim_wd, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
        optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    
    return scheduler_g, scheduler_d, scheduler_wd, scheduler_dur_disc