"""Resolve model references separately from orchestration."""
from pathlib import Path
import re

from omegaconf import OmegaConf
from .artifacts import digest, file_hash


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


def resolve_model_assets(conf, logger=None):
    resolved = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    model = str(conf.vc.model).lower()
    assets, unresolved, cache = {}, {}, {}
    suffixes = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.onnx', '.json', '.yaml', '.yml',
                '.txt', '.tiktoken', '.model', '.vocab', '.py', '.t7', '.jsonl', '.fst', '.jinja'}
    fields = {'checkpoint', 'checkpoint_path', 'generator_checkpoint', 'model_path', 'model_dir', 'models_dir', 'checkpoint_dir',
              'config_path', 'pretrained_dir', 'text_tokens_path', 'text_tokenizer_path', 'hubert_checkpoint', 'hubert_tokenizer', 'vocoder_path',
              'ckpt_file', 'vocab_file', 'vocab_path', 'speaker_file_path', 'base_speaker_dir',
              'vocoder_local_path', 'spt_config_path', 'spt_checkpoint_path', 'codec_path',
              'audio_tokenizer_path', 'scene_prompt_path', 'ckpt_dir', 'frontend_dir', 'reference_asr_model',
              'cosyvoice_path', 'codec_encoder_path', 'codec_decoder_path',
              'facodec_encoder_path', 'facodec_decoder_path'}
    references = {key: value for key, value in resolved.adversary.items()
                  if isinstance(value, str) and value and (key in fields or key.endswith(('_checkpoint_path', '_config_path')))}
    if model == 'moss_ttsd' and resolved.adversary.get('use_prompt_transcript', False):
        references.pop('reference_asr_model', None)
    if model == 'f5_tts':
        from importlib.resources import files
        from huggingface_hub import hf_hub_download, snapshot_download
        preset = str(resolved.adversary.get('model', 'F5TTS_v1_Base'))
        package_root = files('f5_tts')
        config_path = Path(str(package_root.joinpath(f'configs/{preset}.yaml')))
        preset_config = OmegaConf.load(config_path)
        mel = preset_config.model.mel_spec.mel_spec_type
        references['f5_tts.package_config'] = str(config_path)
        if not resolved.adversary.get('vocab_file'):
            vocab = str(package_root.joinpath('infer/examples/vocab.txt'))
            OmegaConf.update(resolved, 'adversary.vocab_file', vocab, force_add=True)
            references['vocab_file'] = vocab
        if not resolved.adversary.get('ckpt_file'):
            repo = 'SWivid/E2-TTS' if preset == 'E2TTS_Base' else 'SWivid/F5-TTS'
            step = 1200000 if preset in ('F5TTS_Base', 'E2TTS_Base') else 1250000
            filename = f'{preset}/model_{step}.safetensors'
            if preset == 'F5TTS_Base' and mel == 'bigvgan':
                filename = 'F5TTS_Base_bigvgan/model_1250000.pt'
            revision = resolve_hub_revision(repo, resolved.adversary.get('revision'))
            checkpoint = hf_hub_download(repo, filename=filename, revision=revision,
                                         cache_dir=resolved.adversary.get('hf_cache_dir'))
            OmegaConf.update(resolved, 'adversary.ckpt_file', checkpoint, force_add=True)
            OmegaConf.update(resolved, 'adversary.revision', revision, force_add=True)
            references['ckpt_file'] = checkpoint
            assets['f5_tts.hub'] = {'repo_id': repo, 'revision': revision, 'filename': filename}
        if not resolved.adversary.get('vocoder_local_path'):
            repos = {'vocos': 'charactr/vocos-mel-24khz',
                     'bigvgan': 'nvidia/bigvgan_v2_24khz_100band_256x'}
            if mel not in repos:
                raise ValueError(f'F5-TTS vocoder requires explicit local assets: {mel}')
            repo = repos[mel]
            revision = resolve_hub_revision(repo, resolved.adversary.get('vocoder_revision'))
            vocoder = snapshot_download(repo, revision=revision,
                cache_dir=resolved.adversary.get('hf_cache_dir'),
                allow_patterns=['config.yaml', 'pytorch_model.bin'] if mel == 'vocos' else None)
            OmegaConf.update(resolved, 'adversary.vocoder_local_path', vocoder, force_add=True)
            OmegaConf.update(resolved, 'adversary.vocoder_revision', revision, force_add=True)
            references['vocoder_local_path'] = vocoder
            assets['f5_tts.vocoder_hub'] = {'repo_id': repo, 'revision': revision}
    if model == 'ozspeech':
        missing_codecs = [name for name in ('encoder', 'decoder')
                          if not (resolved.adversary.get('codec_' + name + '_path') or
                                  resolved.adversary.get('facodec_' + name + '_path'))]
        if missing_codecs:
            from huggingface_hub import hf_hub_download
            repo_id = 'amphion/naturalspeech3_facodec'
            revision = resolve_hub_revision(repo_id, resolved.adversary.get('codec_revision'))
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
            from huggingface_hub import snapshot_download
            repo_id = resolved.adversary.get('hf_repo_id', 'PlayHT/inpainter')
            revision = resolve_hub_revision(repo_id, resolved.adversary.get('hf_revision'))
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
    if model == 'vall_e' and resolved.adversary.get('implementation') == 'amphion':
        references['valle.base_configs'] = str(Path(resolved.adversary.code_path).expanduser() / 'config')
        import torch
        references['valle.encodec_cache'] = str(Path(torch.hub.get_dir()) / 'checkpoints' /
                                               'encodec_24khz-d7cc33bc.th')
    if model == 'bark_voice_clone':
        import torch
        references['bark.encodec_cache'] = str(Path(torch.hub.get_dir()) / 'checkpoints' /
                                              'encodec_24khz-d7cc33bc.th')
    if model == 'styletts2' and resolved.adversary.get('config_path'):
        configuration = OmegaConf.load(resolved.adversary.config_path)
        for name in ('ASR_config', 'ASR_path', 'F0_path', 'PLBERT_dir'):
            value = configuration.get(name)
            if value:
                dependency = Path(value).expanduser()
                if not dependency.is_absolute():
                    dependency = Path(resolved.adversary.code_path).expanduser() / dependency
                references['styletts2.' + name] = str(dependency)
    if model in ('qwen3_tts', 'qwentts') and references.get('checkpoint_path'):
        checkpoint = references['checkpoint_path']
        if not Path(checkpoint).expanduser().exists():
            from huggingface_hub import snapshot_download
            revision = resolve_hub_revision(checkpoint, resolved.adversary.get('revision'))
            # The upstream wrapper forwards revision to the model but not its
            # processor. A local snapshot binds both to the same commit.
            snapshot = snapshot_download(repo_id=checkpoint, revision=revision,
                allow_patterns=['*' + suffix for suffix in sorted(suffixes)])
            OmegaConf.update(resolved, 'adversary.revision', revision, force_add=True)
            OmegaConf.update(resolved, 'adversary.checkpoint_path', snapshot, force_add=True)
            references['checkpoint_path'] = snapshot
            assets['qwen3_tts.hub'] = {'repo_id': checkpoint, 'revision': revision}
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
    reference = {'assets': assets, 'unresolved_references': unresolved,
                 'coverage': 'configured_local_assets_and_supported_hub_resolvers',
                 'note': 'Implicit upstream downloads and service-side models require additional provenance.'}
    return resolved, reference, digest(reference)
