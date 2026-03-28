import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


CANONICAL_MANIFEST_FILENAME = "metadata.parquet"

CANONICAL_COLUMNS: Sequence[str] = (
    "dataset_name",
    "split",
    "pair_id",
    "speaker_id",
    "prompt_file_name",
    "prompt_speaker_id",
    "prompt_language",
    "prompt_text",
    "prompt_phonemes",
    "prompt_tone",
    "prompt_word2ph",
    "target_file_name",
    "target_speaker_id",
    "target_language",
    "target_text",
    "target_phonemes",
    "target_tone",
    "target_word2ph",
    "source_manifest",
    "source_row",
)

LEGACY_TO_CANONICAL: Dict[str, str] = {
    "ori_pth": "prompt_file_name",
    "ori_spk": "prompt_speaker_id",
    "ori_lang": "prompt_language",
    "ori_text": "prompt_text",
    "ori_phonemes": "prompt_phonemes",
    "ori_tone": "prompt_tone",
    "ori_word2ph": "prompt_word2ph",
    "gt_pth": "target_file_name",
    "gt_spk": "target_speaker_id",
    "gt_lang": "target_language",
    "gt_text": "target_text",
    "gt_phonemes": "target_phonemes",
    "gt_tone": "target_tone",
    "gt_word2ph": "target_word2ph",
}

CANONICAL_TO_INTERNAL: Dict[str, str] = {
    "prompt_file_name": "ori_pth",
    "prompt_speaker_id": "ori_spk",
    "prompt_language": "ori_lang",
    "prompt_text": "ori_text",
    "prompt_phonemes": "ori_phonemes",
    "prompt_tone": "ori_tone",
    "prompt_word2ph": "ori_word2ph",
    "target_file_name": "gt_pth",
    "target_speaker_id": "gt_spk",
    "target_language": "gt_lang",
    "target_text": "gt_text",
    "target_phonemes": "gt_phonemes",
    "target_tone": "gt_tone",
    "target_word2ph": "gt_word2ph",
}


def canonical_manifest_path(root_path: Path) -> Path:
    return Path(root_path) / CANONICAL_MANIFEST_FILENAME


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _normalize_record(record: Dict[str, object], dataset_name: str, source_manifest: str, source_row: int) -> Dict[str, object]:
    normalized = dict(record)
    for legacy_key, canonical_key in LEGACY_TO_CANONICAL.items():
        if canonical_key not in normalized and legacy_key in normalized:
            normalized[canonical_key] = normalized.get(legacy_key)

    prompt_speaker = _stringify(normalized.get("prompt_speaker_id") or normalized.get("speaker_id"))
    target_speaker = _stringify(normalized.get("target_speaker_id") or prompt_speaker)
    speaker_id = _stringify(normalized.get("speaker_id") or prompt_speaker or target_speaker)
    prompt_path = _stringify(normalized.get("prompt_file_name"))
    target_path = _stringify(normalized.get("target_file_name") or prompt_path)

    result = {
        "dataset_name": _stringify(normalized.get("dataset_name") or dataset_name),
        "split": _stringify(normalized.get("split") or "default"),
        "pair_id": _stringify(
            normalized.get("pair_id")
            or f"{dataset_name}-{speaker_id or 'unknown'}-{source_row:06d}"
        ),
        "speaker_id": speaker_id,
        "prompt_file_name": prompt_path,
        "prompt_speaker_id": prompt_speaker or speaker_id,
        "prompt_language": _stringify(normalized.get("prompt_language")),
        "prompt_text": _stringify(normalized.get("prompt_text")),
        "prompt_phonemes": _stringify(normalized.get("prompt_phonemes")),
        "prompt_tone": _stringify(normalized.get("prompt_tone")),
        "prompt_word2ph": _stringify(normalized.get("prompt_word2ph")),
        "target_file_name": target_path,
        "target_speaker_id": target_speaker or speaker_id,
        "target_language": _stringify(normalized.get("target_language") or normalized.get("prompt_language")),
        "target_text": _stringify(normalized.get("target_text")),
        "target_phonemes": _stringify(normalized.get("target_phonemes")),
        "target_tone": _stringify(normalized.get("target_tone")),
        "target_word2ph": _stringify(normalized.get("target_word2ph")),
        "source_manifest": source_manifest,
        "source_row": source_row,
    }
    reserved = set(result.keys()) | set(LEGACY_TO_CANONICAL.keys())
    for key, value in normalized.items():
        if key in reserved:
            continue
        result[key] = value
    return result


def canonicalize_records(records: Iterable[Dict[str, object]], *, dataset_name: str, source_manifest: str) -> pd.DataFrame:
    rows = [
        _normalize_record(record, dataset_name=dataset_name, source_manifest=source_manifest, source_row=index)
        for index, record in enumerate(records)
    ]
    if not rows:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    df = pd.DataFrame(rows)
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    ordered = list(CANONICAL_COLUMNS) + [column for column in df.columns if column not in CANONICAL_COLUMNS]
    return df.loc[:, ordered]


def _load_json_records(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def load_canonical_manifest(root_path: Path) -> Optional[pd.DataFrame]:
    manifest_path = canonical_manifest_path(root_path)
    if not manifest_path.exists():
        return None
    df = pd.read_parquet(manifest_path)
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    ordered = list(CANONICAL_COLUMNS) + [column for column in df.columns if column not in CANONICAL_COLUMNS]
    return df.loc[:, ordered]


def list_legacy_manifest_files(root_path: Path) -> List[Path]:
    filelists_dir = Path(root_path) / "filelists"
    if not filelists_dir.exists():
        return []

    candidates: List[Path] = []
    for path in sorted(filelists_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        stem = path.stem.lower()
        if stem.startswith("all_speakers"):
            continue
        candidates.append(path)
    return candidates


def load_legacy_manifest(root_path: Path, *, speaker_id: Optional[str] = None, dataset_name: Optional[str] = None) -> pd.DataFrame:
    root_path = Path(root_path)
    dataset_name = dataset_name or root_path.name

    manifest_files = list_legacy_manifest_files(root_path)
    if speaker_id is not None:
        speaker_id = str(speaker_id)
        manifest_files = [path for path in manifest_files if path.stem == speaker_id]

    frames: List[pd.DataFrame] = []
    for path in manifest_files:
        try:
            records = _load_json_records(path)
        except Exception:
            continue
        frame = canonicalize_records(records, dataset_name=dataset_name, source_manifest=path.name)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def load_dataset_manifest(root_path: Path, *, speaker_id: Optional[str] = None, dataset_name: Optional[str] = None) -> pd.DataFrame:
    root_path = Path(root_path)
    dataset_name = dataset_name or root_path.name

    df = load_canonical_manifest(root_path)
    if df is None:
        df = load_legacy_manifest(root_path, speaker_id=speaker_id, dataset_name=dataset_name)
    elif speaker_id is not None:
        df = df[df["speaker_id"].astype(str) == str(speaker_id)].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    ordered = list(CANONICAL_COLUMNS) + [column for column in df.columns if column not in CANONICAL_COLUMNS]
    return df.loc[:, ordered].reset_index(drop=True)


def to_internal_manifest(df: pd.DataFrame) -> pd.DataFrame:
    internal = df.copy()
    for canonical_key, internal_key in CANONICAL_TO_INTERNAL.items():
        if internal_key not in internal.columns and canonical_key in internal.columns:
            internal[internal_key] = internal[canonical_key]
    return internal


def discover_speakers(root_path: Path, *, explicit_speaker_id: Optional[str] = None, dataset_name: Optional[str] = None) -> List[str]:
    if explicit_speaker_id is not None:
        return [str(explicit_speaker_id)]

    root_path = Path(root_path)
    df = load_dataset_manifest(root_path, dataset_name=dataset_name)
    if not df.empty:
        speakers = sorted({str(value) for value in df["speaker_id"].tolist() if str(value)})
        if speakers:
            return speakers

    audios_root = root_path / "audios"
    if audios_root.exists():
        return sorted(path.name for path in audios_root.iterdir() if path.is_dir() and not path.name.startswith("."))

    return []
