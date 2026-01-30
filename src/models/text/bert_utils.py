from pathlib import Path

from huggingface_hub import hf_hub_download


def _check_bert(repo_id, files, local_path):
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            hf_hub_download(
                repo_id, file, local_dir=local_path, local_dir_use_symlinks=False
            )

