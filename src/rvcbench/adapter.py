"""Public base class for evaluating your own voice cloning model with RVCBench.

Subclass :class:`VoiceCloningAdapter`, implement :meth:`clone`, and select the class
with ``vc.adapter=your_package.your_module:YourAdapter``. The benchmark runner handles
sample selection, seeding, output naming, run records and scoring.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from rvcbench.adversary.base_adversary import BaseAdversary


class VoiceCloningAdapter(BaseAdversary):
    """Zero-shot voice cloning model driven one utterance at a time.

    ``self.config`` is the ``adversary`` block of the run config, ``self.device`` the
    requested device and ``self.logger`` a standard logger.
    """

    #: Label recorded with each synthesis time. Describe what :meth:`clone` measures.
    timing_scope = "external_adapter_clone_call_excluding_output_write"

    def __init__(self, config, dataset_config, device, logger):
        super().__init__(config, device)
        self.dataset_config = dataset_config
        self.logger = logger
        self._loaded = False

    def load(self) -> None:
        """Load weights and other resources once per run. Optional."""

    def clone(self, *, text: str, reference_audio: Path, reference_text: str,
              language: Optional[str]) -> Tuple[np.ndarray, int]:
        """Speak ``text`` in the voice of ``reference_audio``.

        Return ``(waveform, sample_rate)``: a mono float waveform in ``[-1, 1]`` and its
        sample rate in Hz. Raise an exception when the utterance cannot be generated;
        the runner records the failure for that sample and continues.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Release resources at the end of the run. Optional."""

    # -- runner integration -------------------------------------------------------

    def _ensure_model(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def close(self) -> None:
        try:
            if self._loaded:
                self.unload()
        finally:
            self._loaded = False
            super().close()

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        del protected_audio_path  # protected references already replace sample.prompt_path
        self._ensure_model()
        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_synthesis_timings(output_dir)
        try:
            for sample in dataset.get_zero_shot_samples(max_samples=self.config.get("max_samples")):
                reference = self._resolve_prompt_path(sample)
                if reference is None:
                    raise FileNotFoundError(f"Missing reference audio: {sample.prompt_path}")
                text = (sample.target_text or "").strip()
                if not text:
                    raise ValueError("Target text is empty")
                started = time.perf_counter()
                waveform, sample_rate = self.clone(
                    text=text, reference_audio=reference,
                    reference_text=(sample.prompt_text or "").strip(),
                    language=sample.target_language or sample.prompt_language)
                elapsed = time.perf_counter() - started
                waveform = np.asarray(waveform, dtype=np.float32)
                if waveform.ndim == 2 and 1 in waveform.shape:
                    waveform = waveform.reshape(-1)
                if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                    raise ValueError("clone() must return nonempty finite mono audio")
                if int(sample_rate) <= 0:
                    raise ValueError("clone() must return a positive sample rate")
                path = (self._speaker_output_dir(output_dir, str(sample.speaker_id))
                        / self._cloned_filename(sample, sample.index))
                sf.write(str(path), np.clip(waveform, -1.0, 1.0), int(sample_rate))
                self._record_synthesis_timing(path, elapsed)
        finally:
            self._flush_synthesis_timings()
