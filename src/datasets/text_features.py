"""Validated cached text features with explicit legacy RNG fallback."""
from pathlib import Path

import torch


def load_text_feature(path, phone_count, policy='strict_cached', logger=None):
    if policy not in ('strict_cached', 'legacy_random'):
        raise ValueError('text_feature_policy must be strict_cached or legacy_random')
    path = Path(path)
    try:
        feature = torch.load(path, map_location='cpu', weights_only=True)
        if (not isinstance(feature, torch.Tensor) or feature.shape != (1024, phone_count) or
                not feature.is_floating_point() or not torch.isfinite(feature).all()):
            raise ValueError(f'Expected a finite floating tensor with shape (1024, {phone_count})')
        return feature
    except Exception as exc:
        if policy == 'strict_cached':
            raise ValueError(f'Invalid or missing text feature cache {path}: {exc}. '
                             'Prepare matching features or explicitly select legacy_random.') from exc
        if logger:
            logger.warning('Explicit legacy_random text feature fallback for %s: %s', path, exc)
        return torch.randn(1024, phone_count)
