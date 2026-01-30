# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import os
# modeling_bertvits2.py
from typing import Dict, Literal, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass, asdict
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
from src.utils.commons import latest_checkpoint_path, load_checkpoint

# 你自己的工具函数
from src.datasets.mel_preprocessing import spec_to_mel_torch, mel_spectrogram_torch
from src.losses import WavLMLoss, feature_loss, discriminator_loss, generator_loss, kl_loss
import src.utils.commons as commons
from src.models.modules.bertvits2_module import SynthesizerTrn, MultiPeriodDiscriminator, DurationDiscriminator, \
    WavLMDiscriminator
from src.models.model import BaseModel
import numpy as np

whisper_model_path = "openai/whisper-large-v2"

""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")


class BertVits2Wrapper(BaseModel):
    def __init__(self,config, model_config,dataset_config, logger, **kwargs):
        super().__init__(config, model_config, dataset_config, logger)
        self.model_config.net_g['n_vocab'] =len(symbols)
        self.model_config.net_g['spec_channels'] = self.dataset_config.filter_length // 2 + 1
        self.model_config.net_g['segment_size'] =self.config.segment_size // self.dataset_config.hop_length
        self.model_config.net_g['n_speakers'] =self.dataset_config.n_speakers
        self.hop_size= self.dataset_config.hop_length
        self.n_fft= self.dataset_config.filter_length
        self.load_model()

    def load_model(self):
        net_g_config = SynthesizerTrnConfig(**self.model_config.net_g)
        net_d_config = MultiPeriodDiscriminatorConfig(**self.model_config.net_d)
        net_wd_config = WavLMDiscriminatorConfig(model_sr=self.config.sampling_rate, **self.model_config.net_wd)
        net_dur_config = DurationDiscriminatorConfig(**self.model_config.net_dur_disc)

        bertvits2_config = BertVITS2Config(
            net_g_config=net_g_config,
            net_d_config=net_d_config,
            net_wd_config=net_wd_config,
            net_dur_disc_config=net_dur_config,
        )

        model_path = Path(self.checkpoint_path) / "safespeech" / "bert_vits2" / "base_models"
        legacy_model_path = Path(self.checkpoint_path) / "safespeech" / "base_models"
        if not model_path.exists():
            if legacy_model_path.exists():
                self.logger.warning("Falling back to legacy checkpoint path: %s", legacy_model_path)
                model_path = legacy_model_path
            else:
                raise FileNotFoundError(
                    f"Checkpoint directory not found at '{model_path}' or '{legacy_model_path}'. "
                    "Please download the SafeSpeech BertVITS2 weights."
                )

        self.model = BertVITS2Model(config=bertvits2_config)
        def _try_load(glob, module):
            ckpt_path = latest_checkpoint_path(model_path, glob)
            if ckpt_path:
                load_checkpoint(ckpt_path, module, self.logger, None, skip_optimizer=True)
                self.logger.info(f"Loaded: {ckpt_path}")

        _try_load("G_*.pth", self.model.net_g)
        _try_load("D_*.pth", self.model.net_d)
        _try_load("WD_*.pth", self.model.net_wd)
        _try_load("DUR_*.pth", self.model.net_dur_disc)
        self.model.net_g.train()  # 只用 G 前向
        self.model.net_d.eval()
        self.model.net_wd.eval()
        self.model.net_dur_disc.eval()

    #
    # def pad_wav(self, y):
    #     y = torch.nn.functional.pad(
    #         y,
    #         (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
    #         mode="reflect",
    #     )
    #     return y

    def generate(self, input, p_spec, spec_len, mode='SafeSpeech'):
        self.model.net_g.train()
        if mode in ["SPEC", "SafeSpeech"]:
            is_clip = False
        else:
            is_clip = True
        wav_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_, logw_sdp), g = \
            self.model.net_g(input.text, input.text_len, p_spec, spec_len, input.sid, input.tone, input.language, input.bert, input.ja_bert, input.en_bert, is_clip=is_clip)

        if mode in ["SPEC", "SafeSpeech"]:
            return wav_hat, ids_slice
        elif mode in ['EM']:
            if ids_slice is not None:
                p_wav_slice = commons.slice_segments(
                    input.wav, ids_slice* self.dataset_config.hop_length, self.config.segment_size)
            else:
                p_wav_slice = input.wav
            wav_d_hat_r, wav_d_hat_g, fmap_r, fmap_g = self.model.net_d(p_wav_slice, wav_hat)
            return wav_hat,ids_slice, (l_length, z_p, logs_q, m_p, logs_p, z_mask, fmap_r, fmap_g,wav_d_hat_g )
        else:
            raise NotImplementedError('BertVits2 only support SPEC, SafeSpeech or EM')


@dataclass
class SynthesizerTrnConfig:
    n_vocab: int # len(symbols),
    spec_channels: int # data.filter_length // 2 + 1,
    segment_size: int # hps.train.segment_size // hps.data.hop_length,
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: list
    resblock_dilation_sizes: list
    upsample_rates: list
    upsample_initial_channel: int
    upsample_kernel_sizes: list
    n_speakers: int   # data.n_speakers,
    gin_channels: int = 256
    n_layers_trans_flow: int = 4
    use_spk_conditioned_encoder: bool = True
    use_sdp: bool = True
    n_flow_layer: int = 4
    flow_share_parameter: bool = False
    use_transformer_flow: bool = True
    use_noise_scaled_mas: bool = False
    mas_noise_scale_initial: float = 0.01
    noise_scale_delta: float = 2e-6
    current_mas_noise_scale: float = mas_noise_scale_initial


@dataclass
class MultiPeriodDiscriminatorConfig:
    use_spectral_norm: bool = False


@dataclass
class DurationDiscriminatorConfig:
    in_channels: int
    filter_channels: int
    kernel_size: int
    p_dropout: float
    gin_channels: int = 0


@dataclass
class WavLMDiscriminatorConfig:
    model_name_or_path: str
    model_sr: int
    slm_sr: int =16000
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64
    use_spectral_norm: bool = False


class BertVITS2Config(PretrainedConfig):
    model_type = "bertvits2"

    def __init__(self, net_g_config, net_d_config, net_wd_config, net_dur_disc_config,
                 data=None, train=None, **kwargs):
        self.net_g_config = net_g_config
        self.net_d_config = net_d_config
        self.net_wd_config =  net_wd_config
        self.net_dur_disc_config =  net_dur_disc_config
        self.data = data or {}
        self.train = train or {}
        super().__init__(**kwargs)


@dataclass
class GANStepOutput(ModelOutput):
    loss: torch.Tensor
    extras: Optional[Dict[str, torch.Tensor]] = None


class BertVITS2Model(PreTrainedModel):
    config_class = BertVITS2Config
    base_model_prefix = "bertvits2"
    supports_gradient_checkpointing = True
    # A mapping from Hub filenames to the attribute names in this class
    # This is the key to loading the weights correctly.
    _sub_module_files = {
        "G_0.pth": "net_g",
        "D_0.pth": "net_d",
        "WD_0.pth": "net_wd",
        "DUR_0.pth": "net_dur_disc",
    }

    def __init__(self, config: BertVITS2Config):
        super().__init__(config)
        self.config = config

        # 假设这些构造器接受 kwargs dict
        self.net_g = SynthesizerTrn(**asdict(config.net_g_config))
        self.net_d = MultiPeriodDiscriminator(**asdict(config.net_d_config))  # ← 判别器（多周期）
        self.net_wd = WavLMDiscriminator(**asdict(config.net_wd_config))  # ← SLM 判别器模块（若只是判别器的骨架）
        self.net_dur_disc = DurationDiscriminator(**asdict(config.net_dur_disc_config))  # ← 时长判别器（可选）

        # 可选：如果你有 WavLMLoss 这类需要外部资源的模块，不要在这里硬编码 hps；
        # 提供一个 setter，训练脚本创建好再注入：
        wl_config= config.net_wd_config
        self.wl = WavLMLoss(
            model_name_or_path=wl_config.model_name_or_path,
            wd= self.net_wd,
            model_sr=wl_config.model_sr,
            slm_sr=wl_config.slm_sr
        )  # note share the parms
        self.forbidden_grad()

        self.post_init()
        
        
    def forbidden_grad(self):
        # Forbidden the gradient when perturbation generation
        for param in self.net_g.parameters():
            param.requires_grad = False
        for param in self.net_d.parameters():
            param.requires_grad = False
        for param in self.net_dur_disc.parameters():
            param.requires_grad = False
        for param in self.net_wd.parameters():
            param.requires_grad = False
        for param in self.wl.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """Override default train() to keep frozen modules (e.g. WavLM) in eval mode."""
        super().train(mode)
        if getattr(self, "wl", None) is not None:
            # WavLM should remain frozen / eval-only so that it does not flip back to training mode.
            self.wl.eval()
            if hasattr(self.wl, "wavlm"):
                self.wl.wavlm.eval()
                # Avoid gradient checkpoint helper toggling requires_grad on non-leaf tensors.
                if hasattr(self.wl.wavlm, "feature_extractor"):
                    self.wl.wavlm.feature_extractor._requires_grad = False
        return self


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Override the default from_pretrained to handle multiple weight files.
        """
        # Resolve the model path: can be a Hub ID or local path
        if os.path.isdir(pretrained_model_name_or_path):
            model_dir = pretrained_model_name_or_path
        else:
            # Download all relevant files from the Hub to the cache
            model_dir = snapshot_download(repo_id=pretrained_model_name_or_path)

        # 1. Load the configuration
        config = cls.config_class.from_pretrained(model_dir, **kwargs)

        # 2. Instantiate the model structure *without* weights
        model = cls(config)

        # 3. Load the weights for each sub-module
        for filename, module_name in cls._sub_module_files.items():
            sub_module = getattr(model, module_name)
            if sub_module is not None:
                try:
                    # Construct the full path to the weight file
                    weight_file_path = os.path.join(model_dir, filename)
                    if os.path.exists(weight_file_path):
                        print(f"Loading weights for {module_name} from {filename}...")
                        state_dict = torch.load(weight_file_path, map_location="cpu")
                        sub_module.load_state_dict(state_dict)
                    else:
                        print(f"Warning: Weight file {filename} not found in {model_dir}. Skipping {module_name}.")
                except Exception as e:
                    print(f"Error loading weights for {module_name}: {e}")

        # Ensure the model is in evaluation mode by default after loading
        model.eval()
        return model

    def save_pretrained(self, save_directory, **kwargs):
        """
        Override the default save_pretrained to save weights into separate files.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 1. Save the configuration
        self.config.save_pretrained(save_directory)

        # 2. Save the weights for each sub-module
        for filename, module_name in self._sub_module_files.items():
            sub_module = getattr(self, module_name)
            if sub_module is not None:
                print(f"Saving weights for {module_name} to {filename}...")
                weight_file_path = os.path.join(save_directory, filename)
                torch.save(sub_module.state_dict(), weight_file_path)


    # 允许训练器在每步更新噪声缩放等内部状态
    def set_train_step(self, step: int):
        if getattr(self.net_g, "use_noise_scaled_mas", False):
            cur = self.net_g.mas_noise_scale_initial - self.net_g.noise_scale_delta * step
            self.net_g.current_mas_noise_scale = max(cur, 0.0)

    def _forward_g_core(
            self,
            *,
            x, x_lengths, spec, spec_lengths, y, speakers, tone, language, bert, ja_bert, en_bert
    ):
        """一次 G 前向，返回生成语音及中间变量。假设 batch 已在正确 device 上。"""
        (y_hat, l_length, attn, ids_slice, x_mask, z_mask,
         (z, z_p, m_p, logs_p, m_q, logs_q),
         (hidden_x, logw, logw_, logw_sdp), g) = self.net_g(
            x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, ja_bert, en_bert
        )  # question nn.Module

        mel = spec_to_mel_torch(
            spec,
            self.dataset_config.filter_length,
            self.dataset_config.n_mel_channels,
            self.dataset_config.sampling_rate,
            self.dataset_config.mel_fmin,
            self.dataset_config.mel_fmax,
        )
        y_mel = commons.slice_segments(
            mel, ids_slice, self.config.segment_size // self.dataset_config.hop_length
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            self.dataset_config.filter_length,
            self.dataset_config.n_mel_channels,
            self.dataset_config.sampling_rate,
            self.dataset_config.hop_length,
            self.dataset_config.win_length,
            self.dataset_config.mel_fmin,
            self.dataset_config.mel_fmax,
        )
        y_seg = commons.slice_segments(
            y, ids_slice * self.dataset_config.hop_length, self.config.segment_size
        )

        cache = dict(
            y_hat=y_hat, l_length=l_length, ids_slice=ids_slice,
            x_mask=x_mask, z_mask=z_mask,
            z_p=z_p, logs_q=logs_q, m_p=m_p, logs_p=logs_p,
            hidden_x=hidden_x, logw=logw, logw_=logw_, logw_sdp=logw_sdp, g=g,
            y_mel=y_mel, y_hat_mel=y_hat_mel, y_seg=y_seg
        )
        return cache

    def forward(
            self,
            batch_data,
            step: Literal["D", "DUR", "WD", "G", "EM"] = "G"
    ) -> GANStepOutput:

        # Trainer/Accelerate 已把 batch 搬到 device，这里不要再 .to(device)
        # 也不要开 autocast/做 optimizer.step
        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, tone, language, bert, ja_bert, en_bert= batch_data
        # G 的中间量，大多数步骤都需要；D / DUR / WD 可用 no_grad 提前生成
        need_grad_for_g = (step == "G")
        with torch.enable_grad() if need_grad_for_g else torch.no_grad():
            cache = self._forward_g_core(
                x=x, x_lengths=x_lengths, spec=spec, spec_lengths=spec_lengths, y=y,
                speakers=speakers, tone=tone, language=language, bert=bert, ja_bert=ja_bert, en_bert=en_bert
            )

        y_hat = cache["y_hat"]
        y_seg = cache["y_seg"]
        y_mel = cache["y_mel"]
        y_hat_mel = cache["y_hat_mel"]

        # -------- 判别器步骤 --------
        if step == "D":
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y_seg, y_hat.detach())
            d_loss, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            return GANStepOutput(loss=d_loss, extras={"loss_disc": d_loss})

        if step == "DUR" and self.net_dur_disc is not None:
            hx, xm, lw_, lw, g = cache["hidden_x"], cache["x_mask"], cache["logw_"], cache["logw"], cache["g"]
            y_dur_hat_r, y_dur_hat_g = self.net_dur_disc(
                hx.detach(), xm.detach(), lw_.detach(), lw.detach(), g.detach()
            )
            y_dur_hat_r_s, y_dur_hat_g_s = self.net_dur_disc(
                hx.detach(), xm.detach(), lw_.detach(), cache["logw_sdp"].detach(), g.detach()
            )
            y_dur_hat_r = y_dur_hat_r + y_dur_hat_r_s
            y_dur_hat_g = y_dur_hat_g + y_dur_hat_g_s
            dur_loss, _, _ = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
            return GANStepOutput(loss=dur_loss, extras={"loss_dur_disc": dur_loss})

        if step == "WD":
            assert self.wl is not None, "WavLMLoss 未注入：请在训练前调用 model.wl = WavLMLoss(...)"
            loss_slm = self.wl.discriminator(y_seg.detach().squeeze(), y_hat.detach().squeeze()).mean()
            return GANStepOutput(loss=loss_slm, extras={"loss_slm": loss_slm})

        # -------- 生成器步骤 --------
        # 判别器输出（需要梯度）
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y_seg, y_hat)

        loss_dur = torch.sum(cache["l_length"].float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.config.train.c_mel
        loss_kl = kl_loss(cache["z_p"], cache["logs_q"], cache["m_p"], cache["logs_p"],
                          cache["z_mask"]) * self.config.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)

        loss_lm = self.wl(y_seg.detach().squeeze(), y_hat.squeeze()).mean() if self.wl is not None else 0.0
        loss_lm_gen = self.wl.generator(y_hat.squeeze()) if self.wl is not None else 0.0

        if step =="EM":
            total = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            return GANStepOutput(
                loss=total,
                extras={
                    "loss_gen_all": total,
                    "loss_mel": loss_mel,
                    "loss_kl": loss_kl,
                    "loss_fm": loss_fm,
                    "loss_adv": loss_gen,
                    "loss_dur": loss_dur,
                }
            )
        else:
            total = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_lm + loss_lm_gen

            if self.net_dur_disc is not None:
                _, y_dur_hat_g = self.net_dur_disc(
                    cache["hidden_x"], cache["x_mask"], cache["logw_"], cache["logw"], cache["g"]
                )
                _, y_dur_hat_g_sdp = self.net_dur_disc(
                    cache["hidden_x"], cache["x_mask"], cache["logw_"], cache["logw_sdp"], cache["g"]
                )
                y_dur_hat_g = y_dur_hat_g + y_dur_hat_g_sdp
                loss_dur_gen, _ = generator_loss(y_dur_hat_g)
                total = total + loss_dur_gen

            return GANStepOutput(
                loss=total,
                extras={
                    "loss_gen_all": total,
                    "loss_mel": loss_mel,
                    "loss_kl": loss_kl,
                    "loss_fm": loss_fm,
                    "loss_adv": loss_gen,
                    "loss_dur": loss_dur,
                }
            )
