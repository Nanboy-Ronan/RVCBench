import os

from huggingface_hub import hf_hub_download, snapshot_download


def main():
    #### 1. Download pretrained models of BERT-VITS2
    print("Begin to download BERT-VITS2...")
    repo_id = "OedoSoldier/Bert-VITS2-2.3"
    files = ["DUR_0.pth", "D_0.pth", "G_0.pth", "WD_0.pth"]
    download_path = "checkpoints/base_models"
    download(repo_id, files, download_path)


    #### 2. Download pretrained models of DebERTa-V3
    print("Begin to download DeBERTa-V3...")
    repo_id = "microsoft/deberta-v3-large"
    files = ["pytorch_model.bin", "pytorch_model.generator.bin", "spm.model"]
    download_path = "bert/deberta-v3-large"
    download(repo_id, files, download_path)

    
    #### 3. Download pretrained models of WavLM
    print("Begin to download WavLM...")
    repo_id = "microsoft/wavlm-base-plus"
    download_path = "bert_vits2/slm/wavlm-base-plus"
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    snapshot_download(repo_id=repo_id, local_dir=download_path, local_dir_use_symlinks=False)


    #### 4. Download pretrained models of ECAPA-TDNN
    print("Begin to download ECAPA-TDNN...")
    repo_id = "speechbrain/spkrec-ecapa-voxceleb"
    files = ["classifier.ckpt", "embedding_model.ckpt", "label_encoder.txt", "mean_var_norm_emb.ckpt"]
    download_path = "encoders/spkrec-ecapa-voxceleb"
    download(repo_id, files, download_path)


def download(repo_id, files, download_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for file in files:
        try:
            hf_hub_download(repo_id=repo_id, filename=file, local_dir=download_path)
            print(f"{file} has been download successfully!")
        except Exception as e:
            print(f"{file} with error: {e}")


if __name__ == "__main__":
    main()