"""Metric-specific loading, scoring and resource lifetimes."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoreInput:
    reference: Path
    generated: Path
    text: str
    language: str | None


def create_scorer(name, device, logger):
    if name == 'emotion':
        from .emotion import EmotionScorer
        return EmotionScorer(device, logger)
    if name == 'mcd':
        from .mcd import MCDScorer
        return MCDScorer(device, logger)
    if name == 'wer':
        from .wer import WERScorer
        return WERScorer(device, logger)
    if name in ('sim', 'sva'):
        from .speaker import SpeakerScorer
        return SpeakerScorer(device, logger)
    from .auxiliary import AuxiliaryScorer
    return AuxiliaryScorer(name, device, logger)
