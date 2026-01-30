from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import re
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from src.datasets.data_utils import AllSpeakerData
    from src.datasets.data_utils import ZeroShotSample

class BaseAdversary(ABC):
    """Abstract Base Class for all malicious adversaries."""
    def __init__(self, config, device):
        # Hydra sometimes provides the full run config; extract the adversary block if present.
        adversary_cfg = getattr(config, "adversary", None)
        self.config = adversary_cfg or config
        self.device = device
        self._synthesis_timing_records = []
        self._synthesis_timing_path: Optional[Path] = None

    def _speaker_slug(self, speaker_id: str) -> str:
        """Return a filesystem-safe identifier for the supplied speaker identifier."""
        token = str(speaker_id or "").strip()
        slug = re.sub(r"[^0-9A-Za-z_.-]", "_", token)
        return slug or "unknown"

    def _speaker_output_dir(self, base_dir: Path, speaker_id: str) -> Path:
        """Ensure the per-speaker directory exists and return its path."""
        speaker_dir = Path(base_dir) / self._speaker_slug(speaker_id)
        speaker_dir.mkdir(parents=True, exist_ok=True)
        return speaker_dir

    def _resolve_prompt_path(self, sample) -> Optional[Path]:
        raw_path = getattr(sample, "prompt_path", None)
        if raw_path in (None, ""):
            return None
        path = Path(str(raw_path))
        try:
            candidate = path if path.is_absolute() else path.resolve(strict=False)
        except Exception:
            candidate = path
        if candidate.exists():
            return candidate
        return None

    def _count_available_prompts(self, samples: Sequence["ZeroShotSample"]) -> int:
        return sum(1 for sample in samples if self._resolve_prompt_path(sample) is not None)

    def _preview_text(self, text: Optional[str], limit: int = 120) -> str:
        snippet = re.sub(r"\s+", " ", (text or "").strip())
        if len(snippet) > limit:
            return snippet[: limit - 1] + "…"
        return snippet or "<empty>"

    def _describe_audio(self, path: Optional[Path]) -> str:
        if path is None:
            return "<none>"
        try:
            import soundfile as sf  # Local import to avoid hard dependency when unused

            with sf.SoundFile(str(path)) as handle:
                duration = handle.frames / handle.samplerate if handle.samplerate else 0.0
                return f"{path.name} (sr={handle.samplerate}, dur={duration:.2f}s)"
        except Exception:
            return path.name

    def _log_attack_plan(
        self,
        model_label: str,
        samples: Sequence["ZeroShotSample"],
        prompt_count: int,
    ) -> None:
        logger = getattr(self, "logger", None)
        if not logger:
            return
        total = len(samples)
        speakers = [str(sample.speaker_id) for sample in samples]
        unique_speakers = sorted(set(speakers))
        logger.info(
            "[%s] Preparing %d utterances across %d speakers. Prompt audios available: %d.",
            model_label,
            total,
            len(unique_speakers),
            prompt_count,
        )
        if unique_speakers:
            preview = ", ".join(unique_speakers[:8])
            if len(unique_speakers) > 8:
                preview += ", …"
            logger.debug("[%s] Speaker roster: %s", model_label, preview)

    def _log_clone_request(
        self,
        model_label: str,
        index: int,
        total: int,
        speaker_id: str,
        prompt_path: Optional[Path],
        spoken_text: Optional[str],
        *,
        prompt_transcript: Optional[str] = None,
    ) -> None:
        logger = getattr(self, "logger", None)
        if not logger:
            return
        prompt_desc = self._describe_audio(prompt_path)
        text_preview = self._preview_text(spoken_text)
        message = (
            f"[{model_label}] [{index + 1}/{total}] speaker={speaker_id} "
            f"prompt={prompt_desc} text=\"{text_preview}\""
        )
        if prompt_transcript:
            message += f" transcript=\"{self._preview_text(prompt_transcript)}\""
        logger.info(message)

    def _cloned_filename(self, sample, idx: int, suffix: str = "cloned") -> str:
        # Use BOTH prompt_path (input audio) and target_path (ground truth) for unique identification
        prompt_path = getattr(sample, "prompt_path", None)
        target_path = getattr(sample, "target_path", None)
        
        prompt_stem = None
        target_stem = None
        
        # Get prompt stem if available
        if prompt_path is not None:
            try:
                prompt_stem = Path(str(prompt_path)).stem
            except Exception:
                prompt_stem = str(prompt_path)
        
        # Get target stem
        if target_path is not None:
            try:
                target_stem = Path(str(target_path)).stem
            except Exception:
                target_stem = str(target_path)
        
        # Create combined stem using both prompt and target
        if prompt_stem and target_stem:
            stem_source = f"{prompt_stem}_to_{target_stem}"
        elif prompt_stem:
            stem_source = prompt_stem
        elif target_stem:
            stem_source = target_stem
        else:
            stem_source = getattr(sample, "target_stub", None) or f"sample_{idx}"
        
        stem = re.sub(r"[^0-9A-Za-z_+\-]", "_", str(stem_source).strip())
        if not stem:
            stem = f"sample_{idx}"
        return f"{stem}_{suffix}.wav"

    def _init_synthesis_timings(self, output_dir: Path) -> None:
        self._synthesis_timing_records = []
        self._synthesis_timing_path = Path(output_dir) / "synthesis_timings.csv"

    def _record_synthesis_timing(self, generated_path: Path, elapsed_sec: Optional[float]) -> None:
        if elapsed_sec is None:
            return
        if self._synthesis_timing_path is None:
            return
        try:
            resolved = generated_path.resolve()
        except Exception:
            resolved = generated_path
        self._synthesis_timing_records.append(
            {
                "generated_path": str(resolved),
                "synthesis_time_sec": float(elapsed_sec),
            }
        )

    def _flush_synthesis_timings(self) -> None:
        if not self._synthesis_timing_path or not self._synthesis_timing_records:
            return
        self._synthesis_timing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._synthesis_timing_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["generated_path", "synthesis_time_sec"],
            )
            writer.writeheader()
            writer.writerows(self._synthesis_timing_records)
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info("Saved per-sample synthesis timings to %s", self._synthesis_timing_path)

    @abstractmethod
    def attack(
        self,
        *,
        output_path: str,
        dataset: "AllSpeakerData",
        protected_audio_path: Optional[str] = None,
    ) -> None:
        """Generate adversarial audio for the provided dataset."""
        pass
