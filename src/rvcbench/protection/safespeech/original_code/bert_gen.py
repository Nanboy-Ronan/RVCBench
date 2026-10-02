import os
import torch
import argparse
import torch.multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

import sys
sys.path.append("bert_vits2/")
import bert_vits2.commons as commons
import bert_vits2.utils as utils
from bert_vits2.text import check_bert_models, cleaned_text_to_sequence, get_bert
from bert_vits2.config import config
preprocess_text_config = config.preprocess_text_config


def get_args():
    parser = argparse.ArgumentParser(description="The code of bert file generation.")

    parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
    parser.add_argument('--model', type=str, default='BERT_VITS2', help='the surrogate model')
    parser.add_argument('--mode', type=str, default="clean", choices=["clean", "SPEC", "SafeSpeech"], 
                        help='the protection mode')
    parser.add_argument("--num_processes", type=int, default=1)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    model_name = args.model
    dataset_name = args.dataset
    mode = args.mode

    config_path = f"bert_vits2/configs/{dataset_name.lower()}_{model_name.lower()}.json"
    hps = utils.get_hparams_from_file(config_path=config_path)

    check_bert_models()
    lines = []

    if mode == "clean":
        txt = f"filelists/{dataset_name.lower()}_train_text.txt.cleaned"
    else:
        txt = f"filelists/{dataset_name.lower()}_train_asr.txt.cleaned"
    with open(txt, encoding="utf-8") as f:
        lines.extend(f.readlines())

    add_blank = [hps.data.add_blank] * len(lines)
    
    if len(lines) != 0:
        num_processes = args.num_processes
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(process_line, zip(lines, add_blank)),
                total=len(lines),
            ):
                pass

    print(f"Bert is generated! A total of {len(lines)} bert.pt generated!")


def process_line(x):
    line, add_blank = x
    device = config.bert_gen_config.device
    if config.bert_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", "_bert.pt")
    bert_path = bert_path.replace("data/", "bert_data/")

    bert = get_bert(text, word2ph, language_str, device)
    assert bert.shape[-1] == len(phone)
    os.makedirs(os.path.dirname(bert_path), exist_ok=True)
    torch.save(bert, bert_path)


if __name__ == "__main__":
    main()