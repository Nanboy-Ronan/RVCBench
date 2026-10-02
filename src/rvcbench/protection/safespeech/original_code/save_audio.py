import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import torch
import argparse
import torchaudio
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
import soundfile as sf

import sys
sys.path.append("bert_vits2/")
import bert_vits2.commons as commons
import bert_vits2.utils as utils
from bert_vits2.data_utils import (
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader
)


def get_args():
    parser = argparse.ArgumentParser(description="Save audio after generate perturbation")

    parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
    parser.add_argument('--model', type=str, default='BERT_VITS2', choices='BERT_VITS2', help='the surrogate model')
    parser.add_argument('--mode', type=str, default="clean", choices=["clean", "SPEC", "SafeSpeech"], 
                        help='the saving mode of audio files')
    parser.add_argument('--batch-size', type=int, default=27, help='the batch size of protection')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints', help='the storing path of the checkpoints')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model_name = args.model
    dataset_name = args.dataset
    mode = args.mode

    config_path = f"bert_vits2/configs/{dataset_name.lower()}_{model_name.lower()}.json"
    hps = utils.get_hparams_from_file(config_path=config_path)
    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed(hps.train.seed)

    # Build dataset to save
    hps.train.batch_size = int(args.batch_size)
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset,
                              num_workers=4,
                              shuffle=False,
                              collate_fn=collate_fn,
                              batch_size=hps.train.batch_size,
                              pin_memory=True,
                              drop_last=False)
    
    output_folder = f"data/{dataset_name}/protected/{mode}"
    os.makedirs(output_folder, exist_ok=True)

    checkpoint_path = args.checkpoint_path

    if mode == "clean":
        # If the fine-tuning mode is clean
        # Then we save the original file
        count = 0
        for batch_index, batch in enumerate(train_loader):
            wav = batch[4]
            wav_len = batch[5]
            speakers = batch[6]

            for index, wav_i in enumerate(wav):
                wav_len_i = wav_len[index]
                wav_i = wav_i[:, :wav_len_i][0]
                speaker_i = speakers[index]

                output_path = os.path.join(output_folder, f"{speaker_i}_{mode}_{count}.wav")
                sf.write(output_path, wav_i.numpy(), samplerate=hps.data.sampling_rate)

                count += 1
    else:
        # If the fine-tuning mode is SPEC or SafeSpeech
        # Then we save audio file with perturbations

        noise_path = f"{checkpoint_path}/{dataset_name}/noises/{model_name}_{mode}_{dataset_name}.noise"
        noises = torch.load(noise_path, map_location="cpu")
        print(f"The noise path is {noise_path}")
        
        count = 0
        for batch_index, batch in enumerate(train_loader):
            noise = noises[batch_index]
            wav = batch[4]
            wav_len = batch[5]
            speakers = batch[6]
            perturbed_wav = torch.clamp(wav + noise, -1., 1.)

            for index, p_wav_i in enumerate(perturbed_wav):
                wav_len_i = wav_len[index]
                p_wav_i = p_wav_i[:, :wav_len_i][0]
                speaker_i = speakers[index]
                
                protected_wav = p_wav_i
                save_sr = hps.data.sampling_rate

                output_path = os.path.join(output_folder, f"{speaker_i}_{mode}_{count}.wav")
                sf.write(output_path, protected_wav.numpy(), samplerate=save_sr)

                count += 1
    
    # Change the filelist of fine-tuning
    text_file = f"filelists/{dataset_name.lower()}_train_asr.txt.cleaned"
    with open(text_file, "r") as f:
        lines = f.readlines()
    
    with open(text_file, "w") as f:
        for line in lines:
            audio_path, speaker, language, norm_text, phones, tones, word2ph = line.replace("\n", "").split("|")

            audio_name = audio_path.split("/")[-1][:-4]
            audio_index = audio_name.split("_")[-1]

            new_name = f"{speaker}_{mode}_{audio_index}.wav"
            new_path = f"data/{dataset_name}/protected/{mode}/{new_name}"

            info = f"{new_path}|{speaker}|{language}|{norm_text}|{phones}|{tones}|{word2ph}\n"

            f.write(info)


if __name__ == "__main__":
    main()


