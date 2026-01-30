import os
import argparse

from tqdm import tqdm

import sys
sys.path.append("bert_vits2")
from bert_vits2.text.cleaner import clean_text


def main(train_filelist):
    txt_path = train_filelist
    with open(txt_path, "r") as f:
        lines = f.readlines()

    output_path = txt_path + ".cleaned"
    with open(output_path, "w") as f:
        for line in tqdm(lines):
            utt, spk, _, text = line.strip().split("|")
            language = "EN"
            norm_text, phones, tones, word2ph = clean_text(
                text, language
            )
            f.write(
                "{}|{}|{}|{}|{}|{}|{}\n".format(
                    utt,
                    spk,
                    language,
                    norm_text,
                    " ".join(phones),
                    " ".join([str(i) for i in tones]),
                    " ".join([str(i) for i in word2ph]),
                )
            )


if __name__ == "__main__":
    # This part is for standalone execution, not used when called from protector.py
    # You might want to add a dummy train_filelist here for testing purposes
    # For example:
    # main("filelists/libritts_train_text.txt")
    pass