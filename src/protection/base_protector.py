from abc import ABC, abstractmethod
from src.datasets.data_utils import AllSpeakerData
from pathlib import Path
import torch
import soundfile as sf


class BaseProtector(ABC):
    """Abstract Base Class for all protection algorithms."""
    def __init__(self, output_dir, config, dataset_config, logger, device):
        self.output_dir= Path(output_dir)
        self.config = config
        self.logger = logger
        self.dataset_config = dataset_config
        self.protect_method= config.mode.lower()
        self.device = device
        self.get_data()


    def get_data(self):
        self.speaker_data=AllSpeakerData(self.config, self.dataset_config, self.logger)

    @abstractmethod
    def protect(self, input_path, output_path):
        """
        Applies the protection algorithm to the audio files.
        Args:
            input_path (str): Path to the directory of original audio files.
            output_path (str): Path to save the protected audio files.
        """
        pass

    def save_protected_audio(self):
        for sid in self.speaker_data.speakers_ids:
            self.save_protected_audio_by_speaker(sid)

    def save_protected_audio_by_speaker(self, speaker_id):
        """
        仅写 self.output_dir，并更新 data/libritts/filelists/libritts_train_asr.txt.cleaned
        只替换 audio_path；其余字段原样保留。
        """
        out_dir= self.output_dir/f"{speaker_id}/"
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.noises is None:
            noise_path = output_dir/f"{self.protect_method}.noise"
            self.noises = torch.load(noise_path, map_location="cpu")

        total_saved = 0
        train_loader = self.speaker_data.speaker_dataloaders[speaker_id]
        for batch_idx, batch in enumerate(train_loader):
            noise = self.noises[speaker_id][batch_idx].cpu()
            batch= batch.to('cpu')
            wav, wav_len = batch.wav, batch.wav_len
            paths = batch.path_out

            perturbed_wav = torch.clamp(wav + noise, -1., 1.)

            for i, p_wav_i in enumerate(perturbed_wav):
                wav_len_i = wav_len[i]
                p_wav_i = p_wav_i[:, :wav_len_i][0]
                save_sr = self.dataset_config.sampling_rate

                if paths and i < len(paths):
                    orig_path = str(paths[i])
                    orig_stem = Path(orig_path).stem
                else:
                    # 少数情况下没有 paths（尽量避免）
                    orig_path = f"__no_path__/idx_{batch_idx}_{i}"
                    orig_stem = f"idx_{batch_idx}_{i}"

                new_name = f"{orig_stem}.wav"
                out_path = out_dir / new_name
                sf.write(str(out_path), p_wav_i.numpy(), samplerate=save_sr)

                total_saved += 1
        self.logger.info(
            f"Saved {total_saved} protected audios to {out_dir}. "
        )