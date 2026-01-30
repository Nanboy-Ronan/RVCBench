import os
import librosa
import jiwer
import whisper
import argparse
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
import soundfile as sf
from pymcd.mcd import Calculate_MCD
from speechbrain.inference.speaker import SpeakerRecognition


import sys
sys.path.append("bert_vits2/")
import bert_vits2.commons as commons
from bert_vits2.text import cleaned_text_to_sequence, get_bert
from bert_vits2.text.cleaner import clean_text
import bert_vits2.utils as utils
from bert_vits2.models import SynthesizerTrn
from bert_vits2.text.symbols import symbols


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation script after fine-tuning.")

    parser.add_argument('--dataset', type=str, default='LibriTTS', choices=['LibriTTS', 'CMU_ARCTIC'], help='the dataset')
    parser.add_argument('--model', type=str, default='BERT_VITS2', help='the surrogate model')
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

    # The txt file used for test
    test_file = f"filelists/{dataset_name.lower()}_test_text.txt"

    config_path = f'./bert_vits2/configs/{dataset_name.lower()}_bert_vits2.json'
    hps = utils.get_hparams_from_file(config_path=config_path)

    mas_noise_scale_initial = 0.01
    noise_scale_delta = 2e-6

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).to(device)


    checkpoint_folder = args.checkpoint_path
    checkpoint_path = f"{checkpoint_folder}/{dataset_name}/{model_name}_{mode}_{dataset_name}_100.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
    except:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    net_g.load_state_dict(checkpoint)


    evaluation(test_file, net_g, model_name, dataset_name, mode, device)


def evaluation(test_file, model, model_name, dataset_name, mode, device):
    speaker_index = 2

    config_path = f'./bert_vits2/configs/{dataset_name.lower()}_bert_vits2.json'
    hps = utils.get_hparams_from_file(config_path=config_path)

    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed(hps.train.seed)

    model.eval()

    with open(test_file, 'r') as f:
        lines = f.readlines()

    # 1. Generate the evaluation dataset
    output_path = f'evaluation/data/{dataset_name}/{mode}'
    os.makedirs(output_path, exist_ok=True)

    for index, line in tqdm(enumerate(lines), total=len(lines)):
        audio_path, sid, text = line.split('|')
        text = text.replace('\n', '')
        output_audio_name = sid + "_" + audio_path.split('/')[speaker_index] + "_" + str(index) + '.wav'

        language = "EN"
        bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(text, language, hps, device)

        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        speakers = torch.tensor([int(sid)]).to(device)

        noise_scale = 0.2
        noise_scale_w = 0.9
        sdp_ratio = 0.2
        length_scale = 1.0

        audio = model.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert, en_bert,
                            sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                            length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

        rate = hps.data.sampling_rate
        output_file_name = os.path.join(output_path, output_audio_name)

        sf.write(output_file_name, audio, samplerate=rate)

    # 2. Generate the evaluation lists
    syn_path = output_path
    gt_audio_path = test_file
    assert os.path.exists(syn_path), "Synthesis path is not exists!"

    os.makedirs("evaluation/evallists", exist_ok=True)
    eval_list = f'./evaluation/evallists/{model_name}_{mode}_{dataset_name}_text.txt'
    with open(gt_audio_path, 'r') as f:
        gt_audio = f.readlines()

    syn_audio_list = os.listdir(syn_path)
    assert len(syn_audio_list) == len(gt_audio)

    with open(eval_list, 'w') as f:
        for index, gt in enumerate(gt_audio):
            gt_path = gt.split('|')[0]
            text = gt.replace("\n", "").split('|')[2]
            speaker_id = gt_path.split('/')[speaker_index]

            for syn_audio_path in syn_audio_list:
                syn_audio_name = syn_audio_path[:-4]
                inner_sid = syn_audio_name.split('_')[1]
                inner_index = syn_audio_name.split('_')[2]

                if inner_index == str(index):
                    assert inner_sid == speaker_id
                    gt_write_in = gt_path + '|' + text + "\n"
                    syn_write_in = os.path.join(syn_path, syn_audio_path) + '|' + text + "\n"
                    write_in = gt_write_in + syn_write_in
                    f.write(write_in)
                    break


    # 3. Evaluate the generated dataset
    # 3.1 MCD
    with open(eval_list, 'r') as f:
        audio_list = f.readlines()

    gt_audio_list = []
    syn_audio_list = []
    for index, audio_path in enumerate(audio_list):
        if index % 2 == 0:
            gt_audio_list.append(audio_path)
        else:
            syn_audio_list.append(audio_path)

    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    assert len(gt_audio_list) == len(syn_audio_list)

    mcd_value = 0.0
    for gt_path, syn_path in tqdm(zip(gt_audio_list, syn_audio_list), total=len(gt_audio_list)):
        gt_path, syn_path = gt_path.split('|')[0].replace('\n', ''), syn_path.split('|')[0].replace('\n', '')

        # MCD calculation
        mcd = mcd_toolbox.calculate_mcd(gt_path, syn_path)
        mcd_value += mcd

    mcd_value = mcd_value / len(gt_audio_list)
    print(f"Mode {mode}, MCD: ", {mcd_value})


    # 3.2 WER
    model = whisper.load_model("medium.en", device=device).to(device)

    with open(eval_list, 'r') as f:
        lines = f.readlines()

    WER_gt, WER_syn = 0.0, 0.0
    for index, line in enumerate(tqdm(lines)):
        if index % 2 == 0:
            continue
        audio_path, gt_text = line.split('|')
        result = model.transcribe(audio_path, language="en")
        gen_text = result['text']
        wer = jiwer.wer(gt_text, gen_text)

        if index % 2 == 0:
            WER_gt += wer
        else:
            WER_syn += wer

    WER_gt /= (len(lines) // 2)
    WER_syn /= (len(lines) // 2)
    print(f"Mode {mode}: GT WER is {WER_gt:.6f}, Syn WER is {WER_syn:.6f}")


    # 3.3 SIM
    model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            savedir="encoders/spkrec-ecapa-voxceleb",
                                            run_opts={"device": device})

    assert len(gt_audio_list) == len(syn_audio_list)
    with torch.no_grad():
        sim, asr = 0., 0

        for gt_path, syn_path in tqdm(zip(gt_audio_list, syn_audio_list), total=len(gt_audio_list)):
            gt_path, syn_path = gt_path.split('|')[0].replace('\n', ''), syn_path.split('|')[0].replace('\n', '')
            score, prediction = compute_sim(model, gt_path, syn_path)

            sim += score
            if prediction == True:
                asr += 1

        sim = sim / len(gt_audio_list)
        asr = asr / len(gt_audio_list)

        print(f"Mode {mode} on {dataset_name}, SIM {sim:.6f}, ASR {asr:.8f}.")



def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    style_text = None if style_text == "" else style_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text, style_weight
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert_ori
        ja_bert = torch.randn(1024, len(phone))
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "JP":
        bert = torch.randn(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "EN":
        bert = torch.randn(1024, len(phone))
        ja_bert = torch.randn(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def compute_sim(model, path_1, path_2):
    audio_1, sr_1 = torchaudio.load(path_1, channels_first=False)
    audio_1 = model.audio_normalizer(audio_1, sr_1).unsqueeze(0)

    audio_2, sr_2 = torchaudio.load(path_2, channels_first=False)
    audio_2 = model.audio_normalizer(audio_2, sr_2).unsqueeze(0)

    score, decision = model.verify_batch(audio_1, audio_2)

    return score[0].item(), decision[0].item()


if __name__ == "__main__":
    main()