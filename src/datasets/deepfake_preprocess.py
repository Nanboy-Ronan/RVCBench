from pathlib import Path
from typing import List, Dict, Optional, Set
import json
import re


# =========================
# 1. 通用工具函数
# =========================

TIMESTAMP_PATTERN = re.compile(r"^\d{8}-\d{6}$")


def get_latest_timestamp_dir(root_dir: Path) -> Optional[Path]:
    candidates = [
        p for p in root_dir.iterdir()
        if p.is_dir() and TIMESTAMP_PATTERN.match(p.name)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def collect_audio_paths(root: Path, suffix=".wav") -> List[Path]:
    return sorted(root.rglob(f"*{suffix}"))


def extract_speaker_id(folder_name: str) -> Optional[str]:
    """
    p227robocall / p227vctk -> p227
    """
    if folder_name.startswith("p") and folder_name[1:4].isdigit():
        return folder_name[:4]
    return None


# =========================
# 2. Fake audio 收集
# =========================

def collect_fake_audios_for_model(
    model_dir: Path,
) -> Dict[str, List[Path]]:
    """
    只处理一个 xx_ots_on_robotcall
    """
    speaker2audios: Dict[str, List[Path]] = {}

    latest_dir = get_latest_timestamp_dir(model_dir)
    if latest_dir is None:
        return speaker2audios

    gen_audio = latest_dir / "generated_audio"
    if not gen_audio.exists():
        return speaker2audios

    for speaker_dir in gen_audio.iterdir():
        if not speaker_dir.is_dir():
            continue

        speaker_id = extract_speaker_id(speaker_dir.name)
        if speaker_id is None:
            continue

        wavs = collect_audio_paths(speaker_dir)
        if not wavs:
            continue

        speaker2audios.setdefault(speaker_id, []).extend(wavs)

    return speaker2audios


# =========================
# 3. Real audio（VCTK）收集
# =========================
def collect_vctk_audios(
    vctk_root: Path,
    speakers: Set[str],
    max_per_speaker: int,
) -> Dict[str, List[Path]]:

    speaker2audios = {}

    for spk in speakers:
        spk_dir = vctk_root / "audios" / spk
        if not spk_dir.exists():
            continue

        wavs = sorted(spk_dir.glob("*.wav"))
        speaker2audios[spk] = wavs[:max_per_speaker]

    return speaker2audios

# =========================
# 4. JSONL 写入
# =========================

def write_jsonl(
    output_jsonl: Path,
    fake_audios: Dict[str, List[Path]],
    real_audios: Dict[str, List[Path]],
):
    with open(output_jsonl, "w", encoding="utf-8") as f:

        # fake
        for spk, wavs in fake_audios.items():
            for wav in wavs:
                record = {
                    "key": wav.name,
                    "task": "FakeDetection",
                    "messages": [
                        {
                            "role": "user",
                            "content": "<audio> Determine if this speech is real or a deepfake.",
                        }
                    ],
                    "audios": [str(wav)],
                    "labels": "fake",
                    "speaker": spk,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # real
        for spk, wavs in real_audios.items():
            for wav in wavs:
                record = {
                    "key": wav.name,
                    "task": "FakeDetection",
                    "messages": [
                        {
                            "role": "user",
                            "content": "<audio> Determine if this speech is real or a deepfake.",
                        }
                    ],
                    "audios": [str(wav)],
                    "labels": "real",
                    "speaker": spk,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_jsonl_per_model(
    audiobench_logs: Path,
    vctk_root: Path,
    output_dir: Path,
    max_vctk_per_speaker: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in audiobench_logs.iterdir():
        if not model_dir.is_dir():
            continue
        if not model_dir.name.endswith("_ots_on_robotcall"):
            continue

        model_name = model_dir.name.replace("_ots_on_robotcall", "")
        print(f"\n[MODEL] Processing {model_name}")

        fake_map = collect_fake_audios_for_model(model_dir)
        if not fake_map:
            print("  - No fake audios, skip")
            continue

        speakers = set(fake_map.keys())
        real_map = collect_vctk_audios(
            vctk_root,
            speakers,
            max_per_speaker=max_vctk_per_speaker,
        )

        output_jsonl = output_dir / f"fake_detection_{model_name}.jsonl"

        write_jsonl(
            output_jsonl,
            fake_map,
            real_map,
        )

        print(f"  ✔ Saved {output_jsonl}")

# =========================
# 5. main
# =========================

if __name__ == "__main__":

    AUDIOBENCH_LOGS = Path("/home/xenial/scratch/audiobench_logs")
    VCTK_ROOT = Path("/home/xenial/scratch/audiobench_logs/VCTK")
    OUTPUT_DIR = Path("./jsonl_per_model")

    MAX_VCTK_PER_SPEAKER = 30   # ⭐ 你要的参数

    build_jsonl_per_model(
        audiobench_logs=AUDIOBENCH_LOGS,
        vctk_root=VCTK_ROOT,
        output_dir=OUTPUT_DIR,
        max_vctk_per_speaker=MAX_VCTK_PER_SPEAKER,
    )
