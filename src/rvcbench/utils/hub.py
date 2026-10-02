"""Shared fixed-revision resolution for model loaders and provenance."""
import re


def resolve_hub_revision(repo_id, revision=None):
    """Preserve a full commit pin; resolve mutable references only online.

    A pin identifies a version, not the presence or integrity of its weights.
    Model/download loaders remain responsible for finding that version.
    """
    if isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision):
        return revision
    from huggingface_hub import HfApi, constants
    if constants.HF_HUB_OFFLINE:
        raise ValueError(f'Offline model resolution requires a full 40-character commit revision '
                         f'for {repo_id}; received {revision!r}. Resolve and pin it online first.')
    return HfApi().model_info(repo_id, revision=revision).sha


