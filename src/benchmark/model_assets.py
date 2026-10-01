"""Resolve model references separately from orchestration."""
from pathlib import Path

from omegaconf import OmegaConf
from .artifacts import digest, file_hash


def resolve_model_assets(conf, logger=None):
    resolved = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    model = str(conf.vc.model).lower()
    assets, unresolved, cache = {}, {}, {}
    suffixes = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.onnx', '.json', '.yaml', '.yml',
                '.txt', '.tiktoken', '.model', '.vocab', '.py'}
    fields = {'checkpoint', 'checkpoint_path', 'model_path', 'model_dir', 'models_dir', 'checkpoint_dir',
              'config_path', 'hubert_checkpoint', 'hubert_tokenizer', 'vocoder_path',
              'ckpt_file', 'vocab_file', 'vocab_path', 'speaker_file_path', 'base_speaker_dir',
              'vocoder_local_path', 'spt_config_path', 'spt_checkpoint_path', 'codec_path'}
    for key, value in resolved.adversary.items():
        if not isinstance(value, str) or not value or not (key in fields or key.endswith(('_checkpoint_path', '_config_path'))):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute() and not path.exists() and resolved.adversary.get('code_path'):
            upstream_candidate = Path(resolved.adversary.code_path).expanduser() / path
            if upstream_candidate.exists():
                path = upstream_candidate
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = [p for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in suffixes]
        else:
            unresolved[key] = value
            continue
        if not files:
            unresolved[key] = value
            continue
        if logger:
            logger.info('Hashing %s: %d asset files (%.2f GiB)', key, len(files),
                        sum(f.stat().st_size for f in files) / 1024**3)
        hashes = {}
        for file in files:
            identity = str(file.resolve())
            if identity not in cache:
                cache[identity] = file_hash(file)
            hashes[str(file.relative_to(path)) if path.is_dir() else file.name] = cache[identity]
        assets[key] = {'configured_path': value, 'files': hashes}
        if logger:
            logger.info('Verified %s asset hashes', key)
    if model in ('qwen3_tts', 'qwentts') and 'checkpoint_path' in unresolved:
        from huggingface_hub import HfApi
        repo_id = unresolved.pop('checkpoint_path')
        revision = HfApi().model_info(repo_id, revision=resolved.adversary.get('revision')).sha
        OmegaConf.update(resolved, 'adversary.revision', revision, force_add=True)
        assets['checkpoint_path'] = {'repo_id': repo_id, 'revision': revision}
    reference = {'assets': assets, 'unresolved_references': unresolved,
                 'coverage': 'configured_local_assets_and_supported_hub_resolvers',
                 'note': 'Implicit upstream downloads and service-side models require additional provenance.'}
    return resolved, reference, digest(reference)
