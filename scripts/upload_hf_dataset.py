"""Upload the staged Hugging Face dataset (maintainers only).

Stage the dataset folders under ``.hf_upload_staging/`` at the repository root, then run
``python scripts/upload_hf_dataset.py``. The dataset card is copied from ``docs/hf_dataset_card.md``.
"""
from pathlib import Path
import shutil

from huggingface_hub import HfApi


repo_root = Path(__file__).resolve().parents[1]
staging_root = repo_root / ".hf_upload_staging"
readme_src = repo_root / "docs" / "hf_dataset_card.md"
readme_dst = staging_root / "README.md"
cache_dir = staging_root / ".cache"

if readme_src.exists():
    shutil.copy2(readme_src, readme_dst)

if cache_dir.exists():
    shutil.rmtree(cache_dir)

api = HfApi()
api.upload_large_folder(
    folder_path=str(staging_root),
    repo_id="Nanboy/RVCBench",
    repo_type="dataset",
)
print("UPLOAD_DONE")
