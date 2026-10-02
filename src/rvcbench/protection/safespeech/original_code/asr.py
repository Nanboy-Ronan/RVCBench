import os
import torch
import whisper
import argparse

from tqdm import tqdm

import sys
sys.path.append("bert_vits2/")
from bert_vits2.text.cleaner import clean_text


def get_args():
    parser = argparse.ArgumentParser(description="Auto speech recognition of clean mode")

    parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
    parser.add_argument('--mode', type=str, default="clean", choices=["clean", "SPEC", "SafeSpeech"], 
                        help='the protection mode')
    parser.add_argument('--gpu', type=int, default=0, help='use which gpu')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset_name = args.dataset
    mode = args.mode

    audio_folder = f"data/{dataset_name}/protected/{mode}"
    audios = os.listdir(audio_folder)

    gpu = int(args.gpu)
    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"

    model = whisper.load_model("medium.en", device=device)

    with open(f"filelists/{dataset_name.lower()}_train_asr.txt.cleaned", "w") as f:
        for audio in tqdm(audios):
            if not audio.endswith(".wav"):
                continue
            audio_path = os.path.join(audio_folder, audio)
            speaker_id, mode, index = audio.split("_")

            text = model.transcribe(audio_path, language="en")["text"][1:]
            norm_text, phones, tones, word2ph = clean_text(text, "EN")

            target_path = os.path.join(audio_folder, f"{speaker_id}_{mode}_{index}")
            info = "{}|{}|{}|{}|{}|{}|{}\n".format(
                    target_path,
                    speaker_id,
                    "EN",
                    norm_text,
                    " ".join(phones),
                    " ".join([str(i) for i in tones]),
                    " ".join([str(i) for i in word2ph]),
                )

            f.write(info)
        

if __name__ == "__main__":
    main()
