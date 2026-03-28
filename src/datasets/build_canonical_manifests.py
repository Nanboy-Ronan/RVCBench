import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_UTILS_DIR = Path(__file__).resolve().parent
for candidate in (REPO_ROOT, DATASET_UTILS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from manifest_utils import (
    canonical_manifest_path,
    load_legacy_manifest,
)


def _iter_dataset_configs(config_dir: Path) -> Iterable[Tuple[Path, dict]]:
    for path in sorted(config_dir.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        yield path, data


def _resolve_root(config_path: Path, config_data: dict) -> Path:
    root = config_data.get("root_path")
    if root:
        return (config_path.parents[2] / root).resolve()
    path_value = config_data.get("path")
    if path_value:
        return (config_path.parents[2] / path_value).resolve()
    return Path()


def build_manifest(root_path: Path, dataset_name: str, force: bool = False) -> Tuple[bool, str]:
    if not root_path.exists():
        return False, f"skip missing root: {root_path}"

    if not (root_path / "filelists").exists():
        return False, f"skip no filelists: {root_path}"

    output_path = canonical_manifest_path(root_path)
    if output_path.exists() and not force:
        return False, f"skip existing manifest: {output_path}"

    manifest_df = load_legacy_manifest(root_path, dataset_name=dataset_name)
    if manifest_df.empty:
        return False, f"skip empty manifest: {root_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_parquet(output_path, index=False)
    return True, f"wrote {len(manifest_df)} rows -> {output_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical metadata.parquet manifests for benchmark datasets.")
    parser.add_argument(
        "--config-dir",
        default="configs/dataset",
        help="Directory containing dataset yaml configs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional config stem(s) to build. Example: --dataset vctk --dataset libritts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metadata.parquet files.",
    )
    args = parser.parse_args()

    config_dir = (REPO_ROOT / args.config_dir).resolve()

    selected = {name.lower() for name in args.dataset}
    for config_path, config_data in _iter_dataset_configs(config_dir):
        config_stem = config_path.stem.lower()
        if selected and config_stem not in selected:
            continue

        root_path = _resolve_root(config_path, config_data)
        dataset_name = str(config_data.get("name") or root_path.name or config_stem)
        wrote, message = build_manifest(root_path, dataset_name=dataset_name, force=args.force)
        print(f"[{config_path.name}] {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
