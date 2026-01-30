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
from speechbrain.inference import SpeakerRecognition
from torch import Tensor
import random
from src.models.antifake_model import TorchMelSpectrogram, TacotronSTFT, SpeakerEncoder, AE
import torchaudio
from torch import nn
import librosa
import numpy as np
import pickle

sys.path.insert(0, "../../checkpoints/antifake/rtvc")
from encoder import inference as encoder
from encoder import audio
from encoder.params_data import *
from utils.default_models import ensure_default_models

sys.path.insert(0, "../../checkpoints/antifake/TTS")
from TTS.api import TTS

from adaptive_voice_conversion.adaptivevc_backward import Inferencer, extract_speaker_embedding_torch, get_spectrograms_tensor
from adaptive_voice_conversion.model import SpeakerEncoder

sys.path.insert(0, "../../checkpoints/antifake/tortoise")
from tortoise.tortoise_backward import load_voice_path, TextToSpeech, get_conditioning_latents_torch, format_conditioning, ConditioningEncoder, pad_or_truncate, wav_to_univnet_mel, AttentionBlock



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


def get_spectrograms(audio_wav, top_db, preemphasis, n_fft, hop_length, win_length, sampling_ratio, n_mels,ref_db,max_db):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''

    # print('fpath spect input')
    # print(y.shape)

    # Trimming
    y, _ = librosa.effects.trim(audio_wav, top_db=top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length= hop_length,
                          win_length= win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sampling_ratio, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) /  max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    # print(mel.shape)
    return y, mel, mag

def get_spectrograms_tensor(audio, sampling_rate,n_mels, n_fft, hop_length, win_length, norm="slaney",
                                                           mel_scale="slaney",
                                                           power=1):
    # y = torch.cat((y[:1], y[1:] - 0.97 * y[:-1]))

    # Create MelSpectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate,
                                                           n_mels=n_mels,
                                                           n_fft=n_fft,
                                                           hop_length=hop_length,
                                                           win_length=win_length,
                                                           norm=norm,
                                                           mel_scale=mel_scale,
                                                           power=power
                                                           ).cuda()

    # Compute Mel spectrogram
    mel = mel_spectrogram(audio)
    mel = mel.squeeze(0)

    mel = 20 * torch.log10(torch.maximum(torch.tensor(1e-5), mel))
    mel = torch.clamp((mel - 20 + 100) / 100, 1e-8, 1)

    # print(mel.shape)

    return mel


def format_conditioning(clip, cond_length=132300, device='cuda'):
    """
    Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
    """
    # print('clip shape here')
    # print(clip.shape)
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).to(device)

def pad_or_truncate(t, length):
    """
    Utility function for forcing <t> to have the specified sequence length, whether by clipping it or padding it with 0s.
    """
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length-t.shape[-1]))
    else:
        return t[..., :length]



TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254
def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1

def wav_to_univnet_mel(wav, do_normalization=False, device='cuda'):
    # print('1')
    # print(wav.requires_grad)
    stft = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000)
    stft = stft.to(device)
    mel = stft.mel_spectrogram(wav)
    # print('2')
    # print(mel.requires_grad)
    if do_normalization:
        mel = normalize_tacotron_mel(mel)

    # print('3')
    # print(mel.requires_grad)
    return mel

class AntiFakeProtector(BaseProtector):
    def __init__(
            self,
            model_config,
            dataset_config,
            logger,
            **kwargs
    ):
        """
        model_config, protect_config = config, device, logger

            mask_ratio: float number to control the mask place on source waveform, when only for adding noise, should using default value
            random_offset: bool to control random initialize the offset or not


        """
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.logger = logger
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

        # Dataset（建议在 data_utils 里支持 return_path=True）
        self.train_data = TextAudioSpeakerDataset(self.dataset_config,sampling_rate=self.config.sampling_rate,  logger=self.logger)
        # Trainer 的 collate：若末尾有 paths 列表，则剔除，保持训练接口一致
        _base_collate = TextAudioSpeakerCollate()

        def _strip_paths_collate(batch):
            out = _base_collate(batch)
            if isinstance(out, tuple) and len(out) > 0 and isinstance(out[-1], list):
                if len(out[-1]) == 0 or isinstance(out[-1][0], (str, Path)):
                    return out[:-1]
            return out

        self.noises = None
        self.train_loader = None

        # follow the run.py in AntiFake
        self.model = SpeakerEncoder() # TODO 1 add configs
        self.model.load_state_dict(torch.load(Path(self.config.rtvc_default_model_path + '/encoder.pt')))
        # note: preprocess audio by resampling and normalizing --> done by Dataset




        # load model and attr
        self.model = AE(model_config)
        self.model.load_state_dict(torch.load(f'{self.model_path}'))

        self.model.to(self.device)
        self.model.eval()
        self.attr = './adaptive_voice_conversion/attr.pkl'
        with open(self.attr, 'rb') as f:
            self.attr = pickle.load(f)
        self.AVC_ENCODER_MODEL = SpeakerEncoder()

    def embed_frames_batch(self, frames_batch):
        """
        Computes embeddings for a batch of mel spectrogram.

        :param frames_batch: a batch mel of spectrogram as a PyTorch tensor of shape
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a PyTorch tensor of shape (batch_size, model_embedding_size)
        """
        if self.model is None:
            raise Exception("Model was not loaded. Call load_model() before inference.")

        frames = frames_batch.to(self.device)
        embed = self.model(frames)
        return embed

    def embed_utterance_preprocess(self,wav, using_partials=True, **kwargs):
        # Process the entire utterance if not using partials
        if not using_partials:
            frames = audio.wav_to_mel_spectrogram(wav)
            embed = self.embed_frames_batch(frames[None, ...])[0]

            print("Using the entire utterance, please change the code to use partials.")

            return embed

        # Compute where to split the utterance into partials and pad if necessary
        wave_slices, mel_slices = self.compute_partial_slices(len(wav), **kwargs)
        # print("len(wav): ", len(wav))
        # print("wave_slices: ", wave_slices)
        # print("mel_slices: ", mel_slices)
        # exit(0)
        # wave_slices, mel_slices = compute_partial_slices_torch(len(wav), **kwargs)
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
            # wav = torch.cat((wav, torch.zeros(max_wave_length - len(wav))), 0)

        return wav, wave_slices, mel_slices

    def compute_partial_slices(self, n_samples,
                               min_pad_coverage=0.75, overlap=0.5):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
        partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
        spectrogram slices are returned, so as to make each partial utterance waveform correspond to
        its spectrogram. This function assumes that the mel spectrogram parameters used are those
        defined in params_data.py.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
        utterance
        :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
        then the last partial utterance will be considered, as if we padded the audio. Otherwise,
        it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
        utterance, this parameter is ignored so that the function always returns at least 1 slice.
        :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
        utterances are entirely disjoint.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        assert 0 <= overlap < 1
        assert 0 < min_pad_coverage <= 1

        n_samples = torch.tensor(n_samples)
        partial_utterance_n_frames= self.config.partials_n_frames
        partial_utterance_n_frames = torch.tensor(partial_utterance_n_frames)
        overlap = torch.tensor(overlap)

        samples_per_frame = torch.tensor(int((self.config.sampling_rate * self.config.mel_window_step / 1000)))
        n_frames = int(torch.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(torch.round(partial_utterance_n_frames * (1 - overlap))), 1)

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = torch.tensor([i, i + partial_utterance_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]
            # # Exameple of the output
        # mel_slices:  [slice(0, 160, None), slice(80, 240, None)]
        # wav_slices:  [slice(0, 25600, None), slice(12800, 38400, None)]

        return wav_slices, mel_slices

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    def utt_make_frames(self, x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.size(0) % frame_size
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out

    def extract_speaker_embedding_torch(self, original_wav, target_wav):
        # Note Inferencer== self.modal
        original_wav, original_mel, _ = get_spectrograms(original_wav, self.dataset_config...) # todo: 1 add params in self
        target_wav, target_mel, _ = get_spectrograms(target_wav, self.dataset_config...)
        original_mel = torch.from_numpy(self.normalize(original_mel)).cuda()
        target_mel = torch.from_numpy(self.normalize(target_mel)).cuda()
        original_mel.requires_grad_()
        target_mel.requires_grad_()

        x_original = self.utt_make_frames(original_mel)
        x_target = self.utt_make_frames(target_mel)
        original_emb = self.model.get_speaker_embeddings(x_original)
        target_emb = self.model.get_speaker_embeddings(x_target)
        _model = self.model.get_speaker_encoder

        return original_wav, target_wav, original_mel, target_mel, original_emb, target_emb

    # ------------------------------------------------------------------
    # Loss functions copied verbatim from run.py
    # ------------------------------------------------------------------
    def avc_loss(self, wav_tensor_updated: torch.Tensor,
                 avc_embed_initial: torch.Tensor,
                 avc_embed_target: torch.Tensor,
                 avc_embed_threshold: torch.Tensor) -> torch.Tensor:
        """Compute the adaptive voice conversion (AVC) loss.

        This method reproduces the ``avc_loss`` function from run.py.  It
        resamples the waveform, computes the spectrogram and speaker
        embeddings, then evaluates either a threshold‑based or direct
        distance objective depending on the ``THRESHOLD_BASE`` flag.
        """
        sr = self.cfg["sampling_rate"]
        # Resample to the AVC model's expected rate
        wav_resampled = torchaudio.functional.resample(wav_tensor_updated, sr, 24000)
        frames_tensor = get_spectrograms_tensor(wav_resampled)
        frame_tensor = frames_tensor.unsqueeze(0).to(self.device)
        assert self.AVC_ENCODER_MODEL is not None
        self.AVC_ENCODER_MODEL.train()
        embed = self.AVC_ENCODER_MODEL.forward(frame_tensor)
        if self.cfg.get("threshold_base", False):
            elu = torch.nn.ELU()
            delta_l2 = elu(avc_embed_threshold - torch.norm(embed - avc_embed_initial, p=2)) * self.cfg["avc_scale"]
        else:
            delta_l2 = torch.norm(embed - avc_embed_target, p=2) * self.cfg["avc_scale"]
        return delta_l2

    def coqui_loss(self, wav_tensor_updated: torch.Tensor,
                   coqui_embed_initial: torch.Tensor,
                   coqui_embed_target: torch.Tensor,
                   coqui_embed_threshold: torch.Tensor) -> torch.Tensor:
        """Compute the Coqui (YourTTS) loss.

        Mirrors the ``coqui_loss`` function from run.py.
        """
        sr = self.cfg["sampling_rate"]
        wav_resampled = torchaudio.functional.resample(wav_tensor_updated, sr, 16000)
        assert self.COQUI_ENCODER_MODEL is not None
        embed = self.COQUI_ENCODER_MODEL.encoder.compute_embedding(wav_resampled)
        if self.cfg.get("threshold_base", False):
            elu = torch.nn.ELU()
            delta_l2 = elu(coqui_embed_threshold - torch.norm(embed - coqui_embed_initial, p=2)) * self.cfg[
                "coqui_scale"]
        else:
            delta_l2 = torch.norm(embed - coqui_embed_target, p=2) * self.cfg["coqui_scale"]
        return delta_l2

    def tortoise_autoregressive_loss(self, wav_tensor_updated: torch.Tensor,
                                     tortoise_source_emb_autoregressive: torch.Tensor,
                                     tortoise_target_emb_autoregressive: torch.Tensor,
                                     tortoise_threshold_autoregressive: torch.Tensor) -> torch.Tensor:
        """Compute the tortoise autoregressive loss, matching run.py.
        """
        sr = self.cfg["sampling_rate"]
        wav_resampled = torchaudio.functional.resample(wav_tensor_updated, sr, 22050)
        frames = format_conditioning(wav_resampled).to(self.device).unsqueeze(0)
        assert self.TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE is not None
        self.TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE.train()
        embed = self.TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE.forward(frames[0])
        if self.cfg.get("threshold_base", False):
            elu = torch.nn.ELU()
            delta_l2 = elu(
                tortoise_threshold_autoregressive - torch.norm(embed - tortoise_source_emb_autoregressive, p=2))
            delta_l2 = delta_l2 * self.cfg["tortoise_autoregressive_scale"] * 50
        else:
            delta_l2 = torch.norm(embed - tortoise_target_emb_autoregressive, p=2) * self.cfg[
                "tortoise_autoregressive_scale"]
        return delta_l2

    def tortoise_diffusion_loss(self, wav_tensor_updated: torch.Tensor,
                                tortoise_source_emb_diffusion: torch.Tensor,
                                tortoise_target_emb_diffusion: torch.Tensor,
                                tortoise_threshold_diffusion: torch.Tensor) -> torch.Tensor:
        """Compute the tortoise diffusion loss, matching run.py.
        """
        sr = self.cfg["sampling_rate"]
        wav_resampled = torchaudio.functional.resample(wav_tensor_updated, sr, 24000)
        # pad/truncate to 102400 samples
        wav_padded = pad_or_truncate(wav_resampled, 102400)
        mel = wav_to_univnet_mel(wav_padded.to(self.device), do_normalization=False, device=self.device)
        mel = mel.unsqueeze(0)
        assert self.TORTOISE_ENCODER_MODEL_DIFFUSION is not None
        self.TORTOISE_ENCODER_MODEL_DIFFUSION.train()
        embed = self.TORTOISE_ENCODER_MODEL_DIFFUSION.forward(mel[0])
        embed = embed.mean(dim=-1)
        if self.cfg.get("threshold_base", False):
            elu = torch.nn.ELU()
            delta_l2 = elu(tortoise_threshold_diffusion - torch.norm(embed - tortoise_source_emb_diffusion, p=2))
            delta_l2 = delta_l2 * self.cfg["tortoise_diffusion_scale"] * 50
        else:
            delta_l2 = torch.norm(embed - tortoise_target_emb_diffusion, p=2) * self.cfg["tortoise_diffusion_scale"]
        return delta_l2

    def rtvc_loss(self, wav_tensor_updated: torch.Tensor,
                  rtvc_mel_slices: Sequence[slice],
                  rtvc_frame_tensor_list: List[torch.Tensor],
                  rtvc_embeds_list: List[torch.Tensor],
                  rtvc_embed_initial: torch.Tensor,
                  rtvc_embed_target: torch.Tensor,
                  rtvc_embed_threshold: torch.Tensor) -> torch.Tensor:
        """Compute the Real‑Time Voice Cloning (RTVC) loss.
        """
        frames_tensor = audio.wav_to_mel_spectrogram_torch(wav_tensor_updated).to(self.device)
        delta_l2_total = 0.0
        assert self.RTVC_ENCODER_MODEL is not None
        for i, s in enumerate(rtvc_mel_slices):
            frame_tensor = frames_tensor[s].unsqueeze(0).to(self.device)
            rtvc_frame_tensor_list[i] = frame_tensor
            self.RTVC_ENCODER_MODEL.train()
            embed = self.RTVC_ENCODER_MODEL.forward(frame_tensor)
            rtvc_embeds_list[i] = embed
        if self.cfg.get("threshold_base", False):
            elu = torch.nn.ELU()
            for i in range(len(rtvc_frame_tensor_list)):
                delta_l2 = torch.norm(rtvc_embeds_list[i] - rtvc_embed_initial, p=2) * self.cfg["rtvc_scale"]
                delta_l2_total += delta_l2
            delta_l2_total = delta_l2_total / len(rtvc_frame_tensor_list)
            delta_l2_total = elu(rtvc_embed_threshold - delta_l2_total)
        else:
            for i in range(len(rtvc_frame_tensor_list)):
                delta_l2 = torch.norm(rtvc_embeds_list[i] - rtvc_embed_target, p=2) * self.cfg["rtvc_scale"]
                delta_l2_total += delta_l2
        return delta_l2_total

    # Compute embedding with RTVC
    def rtvc_embed(self, wav_tensor_initial, mel_slices, target_speaker_path):

        embeds_list = []
        frame_tensor_list = []
        frames_tensor = audio.wav_to_mel_spectrogram_torch(wav_tensor_initial).to(DEVICE)

        # Get source embeddings
        for s in mel_slices:
            frame_tensor = frames_tensor[s].unsqueeze(0).to(DEVICE)
            frame_tensor_list.append(frame_tensor)
            RTVC_ENCODER_MODEL.train()
            embed = RTVC_ENCODER_MODEL.forward(frame_tensor)
            embeds_list.append(embed)

        partial_embeds = torch.stack(embeds_list, dim=0)
        raw_embed_initial = torch.mean(partial_embeds, dim=0, keepdim=True)

        # Get target embeddings
        preprocessed_wav_target = encoder.preprocess_wav(target_speaker_path, SAMPLING_RATE)
        wav_target, _, _, _, _ = encoder.embed_utterance_preprocess(preprocessed_wav_target, using_partials=True)

        wav_tensor_target = torch.from_numpy(wav_target).unsqueeze(0).to(DEVICE)
        frames_tensor_target = audio.wav_to_mel_spectrogram_torch(wav_tensor_target).to(DEVICE)
        embeds_list_target = []

        for s in mel_slices:
            try:
                frame_tensor_target = frames_tensor_target[s].unsqueeze(0).to(DEVICE)
                embed_target = RTVC_ENCODER_MODEL.forward(frame_tensor_target)
                embeds_list_target.append(embed_target)
            except:
                pass

        partial_embeds_target = torch.stack(embeds_list_target, dim=0)
        raw_embed_target = torch.mean(partial_embeds_target, dim=0, keepdim=True)

        return mel_slices, frame_tensor_list, embeds_list, raw_embed_initial, raw_embed_target

    # Compute embedding with RTVC
    def avc_embed(self,source_speaker_path, target_speaker_path):
        with open(AVC_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        inferencer = Inferencer(config=config, original=source_speaker_path, target=target_speaker_path)
        _, _, _, _, avc_initial_emb, avc_target_emb = extract_speaker_embedding_torch(inferencer)
        global AVC_ENCODER_MODEL
        AVC_ENCODER_MODEL = SpeakerEncoder(**inferencer.config['SpeakerEncoder']).cuda()
        return avc_initial_emb, avc_target_emb

    # Compute embedding with COQUI
    def coqui_embed(self,source_speaker_path, target_speaker_path):
        null_stream = io.StringIO()
        sys.stdout = null_stream
        tts = TTS(model_name=COQUI_YOURTTS_PATH, progress_bar=True, gpu=True)
        speaker_manager = tts.synthesizer.tts_model.speaker_manager
        source_wav = speaker_manager.encoder_ap.load_wav(source_speaker_path, sr=speaker_manager.encoder_ap.sample_rate)
        target_wav = speaker_manager.encoder_ap.load_wav(target_speaker_path, sr=speaker_manager.encoder_ap.sample_rate)
        sys.stdout = sys.__stdout__
        source_wav = torch.from_numpy(source_wav).cuda().unsqueeze(0)
        target_wav = torch.from_numpy(target_wav).cuda().unsqueeze(0)
        coqui_source_emb = speaker_manager.encoder.compute_embedding(source_wav)
        coqui_target_emb = speaker_manager.encoder.compute_embedding(target_wav)
        global COQUI_ENCODER_MODEL
        COQUI_ENCODER_MODEL = speaker_manager
        return coqui_source_emb, coqui_target_emb

    def tortoise_embed(self,source_speaker_path, target_speaker_path):
        tts = TextToSpeech()
        source_wav = load_voice_path(source_speaker_path)
        target_wav = load_voice_path(target_speaker_path)

        tortoise_source_emb_autoregressive, tortoise_source_emb_diffusion, _, _ = get_conditioning_latents_torch(tts,
                                                                                                                 source_wav,
                                                                                                                 return_mels=True)
        tortoise_target_emb_autoregressive, tortoise_target_emb_diffusion, _, _ = get_conditioning_latents_torch(tts,
                                                                                                                 target_wav,
                                                                                                                 return_mels=True)

        if TORTOISE_AUTOREGRESSIVE_LOSS:
            global TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE
            TORTOISE_ENCODER_MODEL_AUTOREGRESSIVE = ConditioningEncoder(80, 1024, num_attn_heads=8).cuda()

        if TORTOISE_DIFFUSION_LOSS:
            model_channels = 1024
            in_channels = 100
            num_heads = 16
            global TORTOISE_ENCODER_MODEL_DIFFUSION
            TORTOISE_ENCODER_MODEL_DIFFUSION = nn.Sequential(
                nn.Conv1d(in_channels, model_channels, 3, padding=1, stride=2),
                nn.Conv1d(model_channels, model_channels * 2, 3, padding=1, stride=2),
                AttentionBlock(model_channels * 2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                AttentionBlock(model_channels * 2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                AttentionBlock(model_channels * 2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                AttentionBlock(model_channels * 2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                AttentionBlock(model_channels * 2, num_heads, relative_pos_embeddings=True, do_checkpoint=False)).cuda()

        return tortoise_source_emb_autoregressive, tortoise_source_emb_diffusion, tortoise_target_emb_autoregressive, tortoise_target_emb_diffusion

    @staticmethod
    def extract_embedding(x: Tensor, model: SpeakerRecognition, device: str | torch.device = 'cuda:0') -> Tensor:
        """
        Extract the embedding from the input waveform tensor.
        """
        x = x.reshape(1, -1).to(device)
        embedding = model.encode_batch(x).reshape(1, -1).to(device)

        return embedding



    def _protect_losses(
        self,
        clean_waveform: Tensor,
        noisy_waveform: Tensor
    ) -> Tensor:

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


    def select_target_speaker(self):
        '''
        Return the most different target speaker
        '''
        # Compute source and target embedding differences, also load each encoder model to the global variables
        print("Computing target speakers embedding differences...")
        rtvc_embedding_diffs = []
        avc_embedding_diffs = []
        coqui_embedding_diffs = []
        tortoise_autoregressive_embedding_diffs = []
        tortoise_diffusion_embedding_diffs = []

        for target_speaker_path in target_speakers_selected:
            if RTVC_LOSS:
                rtvc_mel_slices, rtvc_frame_tensor_list, rtvc_embeds_list, rtvc_embed_initial, rtvc_embed_target = rtvc_embed(
                    wav_tensor_initial, mel_slices, target_speaker_path)
                rtvc_embedding_diffs.append(torch.sum(torch.abs(rtvc_embed_initial - rtvc_embed_target)).item())
            if AVC_LOSS:
                avc_embed_initial, avc_embed_target = avc_embed(source_speaker_path, target_speaker_path)
                avc_embedding_diffs.append(torch.sum(torch.abs(avc_embed_initial - avc_embed_target)).item())
            if COQUI_LOSS:
                coqui_embed_initial, coqui_embed_target = coqui_embed(source_speaker_path, target_speaker_path)
                coqui_embedding_diffs.append(torch.sum(torch.abs(coqui_embed_initial - coqui_embed_target)).item())
            if TORTOISE_AUTOREGRESSIVE_LOSS or TORTOISE_DIFFUSION_LOSS:
                tortoise_source_emb_autoregressive, tortoise_source_emb_diffusion, tortoise_target_emb_autoregressive, tortoise_target_emb_diffusion = tortoise_embed(
                    source_speaker_path, target_speaker_path)
                if TORTOISE_AUTOREGRESSIVE_LOSS:
                    tortoise_autoregressive_embedding_diffs.append(
                        torch.abs(tortoise_source_emb_autoregressive - tortoise_target_emb_autoregressive).sum().item())
                if TORTOISE_DIFFUSION_LOSS:
                    tortoise_diffusion_embedding_diffs.append(
                        torch.abs(tortoise_source_emb_diffusion - tortoise_target_emb_diffusion).sum().item())

        # Normalize embedding diffs, summing the normalized embedding diffs
        all_lists = [rtvc_embedding_diffs, avc_embedding_diffs, coqui_embedding_diffs,
                     tortoise_autoregressive_embedding_diffs, tortoise_diffusion_embedding_diffs]
        all_lists = [[i / (sum(diffs) / len(diffs)) if diffs else 0 for i in diffs] or [0] * NUM_RANDOM_TARGET_SPEAKER
                     for diffs in all_lists]
        rtvc_embedding_diffs, avc_embedding_diffs, coqui_embedding_diffs, tortoise_autoregressive_embedding_diffs, tortoise_diffusion_embedding_diffs = all_lists
        total_embedding_diffs = [sum(values) for values in zip(*all_lists)]

        # Select target speaker that has the largest difference from the source with the analytic hierarchy process
        # Normalize the scores from list1 and list2
        user_scores_weights = np.array(user_scores) / np.sum(user_scores)
        ltotal_embedding_diffs_weights = np.array(total_embedding_diffs) / np.sum(total_embedding_diffs)
        # Aggregate the weights
        # overall_weights = 0.5 * user_scores_weights + 0.5 * ltotal_embedding_diffs_weights # note only keep embedding difference
        # Find the item with the highest score
        selected_target_speaker_path = target_speakers_selected[np.argmax(ltotal_embedding_diffs_weights)]


    def perturb(
            self,
            batch_data,
            noise_real,
            noise_imag,
        ):

        global learning_rate

        for iter in range(self.config.perturbation_epochs):

            if iter % (self.config.lr_decay_iter) == 0 and iter != 0:
                learning_rate = learning_rate * self.config.lr_decay

            loss = 0
            wav_tensor_updated = wav_tensor_list[0]

            # increment loss for each encoder
            if self.config.avc_loss:
                avc_delta_L2 = self.avc_loss(wav_tensor_updated, avc_embed_initial, avc_embed_target, avc_embed_threshold)
                loss += avc_delta_L2

            if self.config.coqui_loss:
                coqui_delta_L2 = self.coqui_loss(wav_tensor_updated, coqui_embed_initial, coqui_embed_target,
                                            coqui_embed_threshold)
                loss += coqui_delta_L2

            if self.config.tortoise_autoregressive_loss:
                delta_L2_autoregressive = self.tortoise_autoregressive_loss(wav_tensor_updated,
                                                                       tortoise_source_emb_autoregressive,
                                                                       tortoise_target_emb_autoregressive,
                                                                       tortoise_threshold_autoregressive)
                loss += delta_L2_autoregressive

            if self.config.tortoise_diffusion_loss:
                delta_L2_diffusion = self.tortoise_diffusion_loss(wav_tensor_updated, tortoise_source_emb_diffusion,
                                                             tortoise_target_emb_diffusion,
                                                             tortoise_threshold_diffusion)
                loss += delta_L2_diffusion

            if self.config.rtvc_loss:
                delta_L2_rtvc = self.rtvc_loss(wav_tensor_updated, rtvc_mel_slices, rtvc_frame_tensor_list, rtvc_embeds_list,
                                          rtvc_embed_initial, rtvc_embed_target, rtvc_embed_threshold)
                loss += delta_L2_rtvc

            # calculate quality norm
            quality_l2_norm = torch.norm(wav_tensor_updated - wav_tensor_initial, p=2)

            # calculate snr
            diff_waveform_squared = torch.square(wav_tensor_updated - wav_tensor_initial)
            signal_power = torch.mean(torch.square(wav_tensor_updated))
            noise_power = torch.mean(diff_waveform_squared)
            quality_snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

            # calculate frequency filter
            quality_frequency = frequency_filter(wav_tensor_updated - wav_tensor_initial)

            # aggregate loss
            quality_term = quality_weight_snr * quality_snr - quality_weight_L2 * quality_l2_norm - quality_weight_frequency * quality_frequency
            loss = -loss + quality_term

            print("Quality term: ", quality_term)
            print("Loss: ", loss)

            loss.backward(retain_graph=True)

            attributions = wav_tensor_updated.grad.data

            with torch.no_grad():

                mean_attributions = torch.mean(torch.abs(attributions))
                # print("Attributions_mean: ", mean_attributions)
                sign_attributions = torch.sign(attributions)
                wav_tensor_updated_clone = wav_tensor_updated.clone()
                wav_tensor_updated_clone += learning_rate * sign_attributions

                # Clip the values of the wav_tensor_updated_clone by using tanh function
                wav_tensor_updated_clone = torch.clamp(wav_tensor_updated_clone, -1, 1)

                wav_tensor_list[0] = wav_tensor_updated_clone
                wav_tensor_list[0].requires_grad = True
                # Clear gradients for the next iteration
                wav_tensor_updated.grad.zero_()

            if iter == ATTACK_ITERATIONS - 1: # note save audio
                wav_updated = wav_tensor_updated.detach().cpu().numpy().squeeze()
                sf.write(OUTPUT_DIR, wav_updated, SAMPLING_RATE)

            # Calculate the progress of the attack
            progress = (iter + 1) / ATTACK_ITERATIONS

            # Update the progress bar
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rProgress: |{bar}| {progress:.2%}', end='', flush=True)
            print("\n")

        end_time = time.time()

        used_time = end_time - start_time

        # Print the optimization time in hours, minutes and seconds
        print("Time used: %d hours, %d minutes, %d seconds" % (used_time // 3600, (used_time % 3600) // 60,
                                                               used_time % 60))

        return refined_waveform

    def generate_perturbations(self, train_data, save_path= 'enkidu.noise'):

        collate_fn = TextAudioSpeakerCollate()
        self.train_loader = DataLoader(
            train_data,
            num_workers=4,
            shuffle=False,  # 有 paths 后可随意
            collate_fn=collate_fn,
            batch_size=self.config.batch_size,
            pin_memory=True,
            drop_last=False
        )

        universal_noise_real = torch.randn((1, self.n_fft // 2 + 1, self.frame_length), requires_grad=True,
                                           device=self.device)
        universal_noise_imag = torch.randn((1, self.n_fft // 2 + 1, self.frame_length), requires_grad=True,
                                           device=self.device)
        self.optimizer = torch.optim.Adam(params=[universal_noise_real, universal_noise_imag], lr=self.learning_rate, weight_decay=self.decay)

        for epoch in range(self.perturbation_epochs):
            for batch_idx, batch_data in enumerate(self.train_loader): # todo batch_size is expected to be 1
                batch_data= batch_data.wav
                batch_data= batch_data.to(self.device)
                perturbed_audio= self.perturb(batch_data, universal_noise_real, universal_noise_imag)
                loss_total, loss_items  = self._protect_losses(batch_data, perturbed_audio)
                loss_total.backward()
                # print(f"Batch {batch_idx}: Loss {loss_items}")
            self.optimizer.step()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        noises={
            'real': universal_noise_real.detach().cpu(),
            'imag': universal_noise_imag.detach().cpu()
        }
        torch.save(noises, save_path)
        self.noises = noises
        self.logger.info(f"Saved noises to {save_path}")

    def save_protected_audio(self):
        """
        仅写 self.output_dir，并更新 data/libritts/filelists/libritts_train_asr.txt.cleaned
        只替换 audio_path；其余字段原样保留。
        """
        if self.noises is None:
            noise_path = os.path.join(self.output_dir, "enkidu.noise")
            self.noises = torch.load(noise_path, map_location="cpu")
            print(f"The noise path is {noise_path}")

        noise_real= self.noises['real']
        noise_imag= self.noises['imag']

        suffix = f"{self.config.mode.lower()}_protected"
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 只更新这个 ASR filelist
        asr_filelist = Path("./data/libritts/filelists/libritts_train_asr.txt.cleaned")
        # 预读，构造 meta 索引（用于 signature 匹配）
        path2meta = {}
        sig2path = {}
        for ln in _read_lines(asr_filelist):
            try:
                ap, spk, lang, txt, ph, tn, w2p = _split_line(ln)
            except ValueError:
                continue
            path2meta[ap] = (spk, lang, txt, ph, tn, w2p)
            sig2path[_sig(lang, txt, ph, tn, w2p)] = ap

        # 构建映射
        mapping_by_path = {}  # orig_path -> new_result_path
        mapping_by_sig = {}  # signature -> new_result_path
        mapping_by_stem = {}  # stem -> new_result_path

        total_saved = 0
        for batch_idx, batch in enumerate(self.train_loader):
            paths = batch.path_out
            wav_len= batch.wav_len
            wav_clean_batch= batch.wav
            perturbed_batch = self.perturb(wav_clean_batch, noise_real, noise_imag)
            perturbed_batch= perturbed_batch.detach().cpu()

            for i, p_wav_i in enumerate(perturbed_batch):
                wav_len_i = wav_len[i]
                p_wav_i = p_wav_i[:wav_len_i]
                save_sr = self.dataset_config.sampling_rate

                if paths and i < len(paths):
                    orig_path = str(paths[i])
                    orig_stem = Path(orig_path).stem
                else:
                    # 少数情况下没有 paths（尽量避免）
                    orig_path = f"__no_path__/idx_{batch_idx}_{i}"
                    orig_stem = f"idx_{batch_idx}_{i}"

                new_name = f"{orig_stem}_{suffix}.wav"
                out_path = out_dir / new_name
                sf.write(str(out_path), p_wav_i.numpy(), samplerate=save_sr)

                mapping_by_path[orig_path] = str(out_path)
                mapping_by_stem[orig_stem] = str(out_path)

                meta = path2meta.get(orig_path)
                if meta is not None:
                    spk, lang, txt, ph, tn, w2p = meta
                    mapping_by_sig[_sig(lang, txt, ph, tn, w2p)] = str(out_path)

                total_saved += 1

        # 回写 ASR filelist：只改音频路径
        lines = _read_lines(asr_filelist)
        updated = 0
        new_lines = []
        for ln in lines:
            try:
                ap, spk, lang, txt, ph, tn, w2p = _split_line(ln)
            except ValueError:
                new_lines.append(ln)
                continue

            new_ap = None
            # ① 按原/当前路径精确替换
            if ap in mapping_by_path:
                new_ap = mapping_by_path[ap]
            # ② 按 signature 替换（覆盖旧命名）
            if new_ap is None:
                s = _sig(lang, txt, ph, tn, w2p)
                if s in mapping_by_sig:
                    new_ap = mapping_by_sig[s]
            # ③ 按 stem 兜底
            if new_ap is None:
                st = Path(ap).stem
                if st in mapping_by_stem:
                    new_ap = mapping_by_stem[st]

            if new_ap is not None and new_ap != ap:
                new_lines.append("|".join([new_ap, spk, lang, txt, ph, tn, w2p]))
                updated += 1
            else:
                new_lines.append(ln)

        asr_generated_filelist = Path("./data/libritts/filelists/libritts_enkidu_train_asr.txt.cleaned")

        if updated > 0:
            _write_lines(asr_generated_filelist, new_lines)

        self.logger.info(
            f"Saved {total_saved} protected audios to {out_dir} (suffix='{suffix}'). "
            f"Updated {updated} entries in {asr_filelist}."
        )

    def protect(self):
        noise_path = os.path.join(self.output_dir, "enkidu.noise")
        self.generate_perturbations(
            train_data=self.train_data,
            save_path=noise_path
        )
        self.save_protected_audio()