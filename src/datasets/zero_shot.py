"""Manifest-only zero-shot data access; no training features or model downloads."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hydra.utils import to_absolute_path
from .manifest_utils import load_dataset_manifest, resolve_hf_dataset_root


@dataclass(frozen=True)
class ZeroShotSample:
    speaker_id: str
    index: int
    prompt_path: Optional[Path]
    prompt_text: str
    prompt_language: Optional[str]
    target_path: Optional[Path]
    target_text: str
    target_language: Optional[str]
    extra: dict

    @property
    def target_stub(self):
        return self.target_path.stem if self.target_path else f'sample_{self.index}'


class ZeroShotDataset:
    def __init__(self, config, dataset_config, logger):
        self.config, self.dataset_config, self.logger = config, dataset_config, logger
        root = Path(to_absolute_path(str(dataset_config.root_path)))
        if dataset_config.get('use_hf_dataset', True) and dataset_config.get('hf_config_name'):
            root = resolve_hf_dataset_root(dataset_config.get('hf_dataset_id', 'Nanboy/RVCBench'),
                                          dataset_config.hf_config_name,
                                          revision=dataset_config.get('hf_revision'), logger=logger)
        self._dataset_root = root.resolve()
        frame = load_dataset_manifest(self._dataset_root, dataset_name=dataset_config.get('name', root.name),
                                      manifest_filename=dataset_config.get('manifest_filename'))
        selected = dataset_config.get('speaker_id')
        if selected is not None:
            frame = frame[frame.speaker_id.astype(str) == str(selected)]
        self._zero_shot_samples = []
        for idx, raw in enumerate(frame.to_dict(orient='records')):
            row = {k: (None if isinstance(v, float) and v != v else v) for k, v in raw.items()}
            def resolve(value):
                if not value:
                    return None
                p = Path(str(value))
                if p.is_absolute():
                    return p
                candidates = [self._dataset_root / p, self._dataset_root.parent / p]
                return next((p.resolve() for p in candidates if p.is_file()), candidates[0].resolve())
            self._zero_shot_samples.append(ZeroShotSample(
                str(row['speaker_id']), idx, resolve(row.get('prompt_file_name')),
                row.get('prompt_text') or '', row.get('prompt_language'), resolve(row.get('target_file_name')),
                row.get('target_text') or '', row.get('target_language'), row))
        if not self._zero_shot_samples:
            raise ValueError(f'No samples found for speaker={selected!r} in {root}')

    def get_zero_shot_samples(self, *, speaker_id=None, max_samples=None):
        if max_samples is not None and int(max_samples) <= 0:
            raise ValueError('max_samples must be positive')
        samples = [s for s in self._zero_shot_samples if speaker_id is None or s.speaker_id == str(speaker_id)]
        return samples[:int(max_samples)] if max_samples is not None else samples

    def iter_zero_shot_samples(self, **kwargs):
        return iter(self.get_zero_shot_samples(**kwargs))
