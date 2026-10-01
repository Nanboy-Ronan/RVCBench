"""Resolve model references separately from orchestration."""
from pathlib import Path

from omegaconf import OmegaConf
from .artifacts import digest, file_hash


def resolve_model_assets(conf, logger=None):
    resolved = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    model = str(conf.vc.model).lower()
    assets, unresolved, cache = {}, {}, {}
    suffixes = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.onnx', '.json', '.yaml', '.yml',
                '.txt', '.tiktoken', '.model', '.vocab', '.py', '.t7', '.jsonl', '.fst', '.jinja'}
    fields = {'checkpoint', 'checkpoint_path', 'model_path', 'model_dir', 'models_dir', 'checkpoint_dir',
              'config_path', 'pretrained_dir', 'hubert_checkpoint', 'hubert_tokenizer', 'vocoder_path',
              'ckpt_file', 'vocab_file', 'vocab_path', 'speaker_file_path', 'base_speaker_dir',
              'vocoder_local_path', 'spt_config_path', 'spt_checkpoint_path', 'codec_path',
              'audio_tokenizer_path', 'scene_prompt_path', 'ckpt_dir', 'frontend_dir', 'reference_asr_model',
              'cosyvoice_path', 'codec_encoder_path', 'codec_decoder_path',
              'facodec_encoder_path', 'facodec_decoder_path'}
    references = {key: value for key, value in resolved.adversary.items()
                  if isinstance(value, str) and value and (key in fields or key.endswith(('_checkpoint_path', '_config_path')))}
    if model == 'moss_ttsd' and resolved.adversary.get('use_prompt_transcript', False):
        references.pop('reference_asr_model', None)
    if model == 'ozspeech':
        missing_codecs = [name for name in ('encoder', 'decoder')
                          if not (resolved.adversary.get('codec_' + name + '_path') or
                                  resolved.adversary.get('facodec_' + name + '_path'))]
        if missing_codecs:
            from huggingface_hub import HfApi, hf_hub_download
            repo_id = 'amphion/naturalspeech3_facodec'
            revision = HfApi().model_info(repo_id, revision=resolved.adversary.get('codec_revision')).sha
            for name in missing_codecs:
                key = 'codec_' + name + '_path'
                path = hf_hub_download(repo_id=repo_id, revision=revision,
                                       filename='ns3_facodec_' + name + '.bin')
                references[key] = path
                OmegaConf.update(resolved, 'adversary.' + key, path, force_add=True)
            assets['ozspeech.codec_hub'] = {'repo_id': repo_id, 'revision': revision}
        if resolved.adversary.get('code_path'):
            references['ozspeech.lexicon'] = str(Path(resolved.adversary.code_path).expanduser() /
                                                'zact/lexicon/librispeech-lexicon.txt')
    if model == 'playdiffusion':
        from src.models.playdiffusion.generator import PRESET_FILES, PlayDiffusionGeneratorConfig
        filenames = {key: resolved.adversary.get(key, getattr(PlayDiffusionGeneratorConfig, key))
                     for key in PRESET_FILES}
        preset = resolved.adversary.get('preset_dir')
        if not preset:
            from huggingface_hub import HfApi, snapshot_download
            repo_id = resolved.adversary.get('hf_repo_id', 'PlayHT/inpainter')
            revision = HfApi().model_info(repo_id, revision=resolved.adversary.get('hf_revision')).sha
            preset = snapshot_download(repo_id=repo_id, revision=revision,
                cache_dir=resolved.adversary.get('cache_dir'), allow_patterns=list(filenames.values()))
            OmegaConf.update(resolved, 'adversary.preset_dir', preset, force_add=True)
            OmegaConf.update(resolved, 'adversary.hf_revision', revision, force_add=True)
            assets['playdiffusion.hub'] = {'repo_id': repo_id, 'revision': revision}
        for key, filename in filenames.items():
            references['playdiffusion.' + key] = str(Path(preset).expanduser() / filename)
    if model in ('glm_tts', 'glmtts') and resolved.adversary.get('code_path'):
        upstream = Path(resolved.adversary.code_path).expanduser()
        references['glmtts.configs'] = str(upstream / 'configs')
    if model == 'styletts2' and resolved.adversary.get('config_path'):
        configuration = OmegaConf.load(resolved.adversary.config_path)
        for name in ('ASR_config', 'ASR_path', 'F0_path', 'PLBERT_dir'):
            value = configuration.get(name)
            if value:
                dependency = Path(value).expanduser()
                if not dependency.is_absolute():
                    dependency = Path(resolved.adversary.code_path).expanduser() / dependency
                references['styletts2.' + name] = str(dependency)
    for key, value in references.items():
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
