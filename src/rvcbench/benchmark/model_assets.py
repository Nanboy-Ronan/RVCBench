"""Resolve model references separately from orchestration."""
from pathlib import Path

from omegaconf import OmegaConf
from .artifacts import digest, file_hash
from rvcbench.utils.hub import resolve_hub_revision


def _maskgct_espeak_resources(runtime_python):
    import json
    import subprocess
    from rvcbench.models.maskgct.generator import build_worker_env
    code = """import json
from phonemizer.backend.espeak.wrapper import EspeakWrapper
wrapper = EspeakWrapper()
print(json.dumps({'version': list(wrapper.version), 'library_path': str(wrapper.library_path),
                  'data_path': str(wrapper.data_path)}))
"""
    try:
        result = subprocess.run([str(runtime_python), '-c', code], check=True,
            capture_output=True, text=True, timeout=30, env=build_worker_env(runtime_python))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'MaskGCT eSpeak resource probe failed: {(exc.stderr or str(exc))[-2000:]}') from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('MaskGCT eSpeak resource probe timed out after 30 seconds') from exc
    resources = json.loads(result.stdout)
    if not Path(resources['library_path']).is_file() or not Path(resources['data_path']).is_dir():
        raise FileNotFoundError('MaskGCT eSpeak library or data path is missing')
    return resources


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
    if model == 'openvoice':
        capture = resolved.adversary.get('capture_english_text_resources', False)
        if not isinstance(capture, bool):
            raise ValueError('OpenVoice capture_english_text_resources must be boolean')
        if capture:
            from rvcbench.models.openvoice.text_resources import english_text_resources
            code_path = resolved.adversary.get('melo_code_path') or 'checkpoints/MeloTTS'
            for name, path in english_text_resources(code_path).items():
                references[f'openvoice.english_resource.{name}'] = str(path)
    if model == 'openvoice' and resolved.adversary.get('text_models') is not None:
        from huggingface_hub import snapshot_download
        from rvcbench.models.openvoice.text_models import text_files
        models = resolved.adversary.text_models
        if not OmegaConf.is_dict(models):
            raise ValueError('OpenVoice text_models must map upstream repository IDs to assets')
        for repo, entry in models.items():
            if not isinstance(repo, str) or not repo or not OmegaConf.is_dict(entry):
                raise ValueError('OpenVoice text_models requires repository keys and asset mappings')
            if not entry.get('path'):
                load_model = entry.get('load_model', False)
                if not isinstance(load_model, bool):
                    raise ValueError('OpenVoice text load_model must be boolean')
                revision = resolve_hub_revision(repo, entry.get('revision'))
                patterns = ['config.json', 'tokenizer_config.json', 'vocab.txt', 'tokenizer.json',
                            'special_tokens_map.json', 'added_tokens.json']
                if load_model:
                    patterns += ['model.safetensors', 'pytorch_model.bin']
                entry.path = snapshot_download(repo_id=repo, revision=revision,
                    allow_patterns=patterns, cache_dir=resolved.adversary.get('hf_cache_dir'),
                    local_files_only=bool(resolved.adversary.get('local_files_only', False)))
                entry.revision = revision
                assets[f'openvoice.text.{repo}.hub'] = {'repo_id': repo, 'revision': revision}
            for path in text_files(entry):
                references[f'openvoice.text.{repo}.{path.name}'] = str(path)
    if model == 'openvoice' and resolved.adversary.get('melo_models') is not None:
        from huggingface_hub import snapshot_download
        from rvcbench.models.openvoice.assets import melo_files
        models = resolved.adversary.melo_models
        if not OmegaConf.is_dict(models):
            raise ValueError('OpenVoice melo_models must be a mapping of language to assets')
        for language, entry in models.items():
            if not isinstance(language, str) or not language or language != language.upper() or not OmegaConf.is_dict(entry):
                raise ValueError('OpenVoice melo_models requires uppercase language keys and asset mappings')
            local_config, local_checkpoint = entry.get('config_path'), entry.get('checkpoint_path')
            if bool(local_config) != bool(local_checkpoint):
                raise ValueError(f'OpenVoice Melo {language} requires both local config and checkpoint paths')
            if not local_config:
                repo = entry.get('repo_id')
                if not isinstance(repo, str) or not repo:
                    raise ValueError(f'OpenVoice Melo {language} requires repo_id or explicit local paths')
                revision = resolve_hub_revision(repo, entry.get('revision'))
                snapshot = Path(snapshot_download(repo_id=repo, revision=revision,
                    allow_patterns=['config.json', 'checkpoint.pth'],
                    cache_dir=resolved.adversary.get('hf_cache_dir'),
                    local_files_only=bool(resolved.adversary.get('local_files_only', False))))
                entry.config_path, entry.checkpoint_path = str(snapshot / 'config.json'), str(snapshot / 'checkpoint.pth')
                entry.revision = revision
                assets[f'openvoice.melo.{language}.hub'] = {'repo_id': repo, 'revision': revision}
            for key, path in melo_files(entry).items():
                entry[key] = str(path)
                references[f'openvoice.melo.{language}.{key}'] = str(path)
    if model == 'cosyvoice':
        from rvcbench.models.cosyvoice.assets import model_directory, checkpoint_files
        configured = resolved.adversary.get('model_dir')
        if not configured:
            raise ValueError("CosyVoice requires an explicit local model_dir")
        variant = resolved.adversary.get('variant', 'cosyvoice2')
        directory = model_directory(configured, variant)
        checkpoint_files(directory, variant, resolved.adversary.get('zero_shot_spk_id', ''))
        OmegaConf.update(resolved, 'adversary.model_dir', str(directory), force_add=True)
        references['model_dir'] = str(directory)
    if model == 'sparktts':
        from rvcbench.models.sparktts.assets import model_directory, checkpoint_files
        directory = model_directory(resolved.adversary.get('code_path', 'checkpoints/Spark-TTS'),
            resolved.adversary.get('model_dir', 'pretrained_models/Spark-TTS-0.5B'))
        checkpoint_files(directory)
        OmegaConf.update(resolved, 'adversary.model_dir', str(directory), force_add=True)
        references['model_dir'] = str(directory)
    if model == 'zipvoice':
        from huggingface_hub import snapshot_download
        from rvcbench.models.zipvoice.assets import (MODEL_REPO, MODEL_DIRS, VOCODER_REPO,
                                               VOCODER_FILES, model_files, vocoder_files)
        model_dir = resolved.adversary.get('model_dir')
        checkpoint_name = str(resolved.adversary.get('checkpoint_name', 'model.pt'))
        options = {'cache_dir': resolved.adversary.get('hf_cache_dir'),
                   'local_files_only': bool(resolved.adversary.get('local_files_only', False))}
        if not model_dir:
            model_name = str(resolved.adversary.get('model_name', 'zipvoice')).strip().lower()
            if model_name not in MODEL_DIRS:
                raise ValueError(f'Unsupported ZipVoice Hub model: {model_name}')
            subdir = MODEL_DIRS[model_name]
            revision = resolve_hub_revision(MODEL_REPO, resolved.adversary.get('revision'))
            snapshot = snapshot_download(repo_id=MODEL_REPO, revision=revision,
                allow_patterns=[f'{subdir}/{name}' for name in
                                (checkpoint_name, 'model.json', 'tokens.txt')], **options)
            model_dir = str(Path(snapshot) / subdir)
            OmegaConf.update(resolved, 'adversary.revision', revision, force_add=True)
            assets['zipvoice.hub'] = {'repo_id': MODEL_REPO, 'revision': revision,
                                     'model_name': model_name, 'checkpoint_name': checkpoint_name}
        model_files(model_dir, checkpoint_name)
        model_dir = str(Path(model_dir).expanduser().absolute())
        OmegaConf.update(resolved, 'adversary.model_dir', model_dir, force_add=True)
        references['model_dir'] = model_dir
        vocoder = resolved.adversary.get('vocoder_path')
        if not vocoder:
            revision = resolve_hub_revision(VOCODER_REPO, resolved.adversary.get('vocoder_revision'))
            vocoder = snapshot_download(repo_id=VOCODER_REPO, revision=revision,
                                       allow_patterns=list(VOCODER_FILES), **options)
            OmegaConf.update(resolved, 'adversary.vocoder_revision', revision, force_add=True)
            assets['zipvoice.vocoder_hub'] = {'repo_id': VOCODER_REPO, 'revision': revision}
        vocoder_files(vocoder)
        vocoder = str(Path(vocoder).expanduser().absolute())
        OmegaConf.update(resolved, 'adversary.vocoder_path', vocoder, force_add=True)
        references['vocoder_path'] = vocoder
    if model == 'xtts':
        from huggingface_hub import snapshot_download
        from rvcbench.models.xtts.assets import SNAPSHOT_FILES, checkpoint_files
        checkpoint = str(resolved.adversary.get('checkpoint') or 'coqui/XTTS-v2')
        directory = Path(checkpoint).expanduser()
        if not directory.exists():
            revision = resolve_hub_revision(checkpoint, resolved.adversary.get('revision'))
            directory = Path(snapshot_download(repo_id=checkpoint, revision=revision,
                cache_dir=resolved.adversary.get('cache_dir'),
                local_files_only=bool(resolved.adversary.get('local_files_only', False)),
                allow_patterns=list(SNAPSHOT_FILES)))
            OmegaConf.update(resolved, 'adversary.revision', revision, force_add=True)
            assets['xtts.hub'] = {'repo_id': checkpoint, 'revision': revision}
        directory = directory.absolute()
        paths = checkpoint_files(directory, **{key: resolved.adversary.get(key)
            for key in ('config_path', 'checkpoint_path', 'vocab_path', 'speaker_file_path')})
        OmegaConf.update(resolved, 'adversary.checkpoint', str(directory), force_add=True)
        references['checkpoint'] = str(directory)
        for key, path in paths.items():
            OmegaConf.update(resolved, 'adversary.' + key, str(path.absolute()), force_add=True)
            references[key] = str(path.absolute())
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
    if model == 'maskgct':
        from rvcbench.models.maskgct.assets import (CHECKPOINT_FILES, checkpoint_files,
                                                SEMANTIC_REPO, SEMANTIC_FILES, semantic_files)
        directory = resolved.adversary.get('checkpoint_dir')
        if not directory:
            from huggingface_hub import snapshot_download
            repo = resolved.adversary.get('repo_id', 'amphion/MaskGCT')
            revision = resolve_hub_revision(repo, resolved.adversary.get('revision'))
            directory = snapshot_download(repo, revision=revision,
                allow_patterns=list(CHECKPOINT_FILES),
                cache_dir=resolved.adversary.get('hf_cache_dir'))
            OmegaConf.update(resolved, 'adversary.revision', revision, force_add=True)
            assets['maskgct.hub'] = {'repo_id': repo, 'revision': revision}
        paths = checkpoint_files(directory)
        OmegaConf.update(resolved, 'adversary.checkpoint_dir', str(Path(directory).expanduser().absolute()), force_add=True)
        for filename, path in zip(CHECKPOINT_FILES, paths):
            references['maskgct.' + filename] = str(path)
        semantic_dir = resolved.adversary.get('semantic_model_path')
        if not semantic_dir:
            from huggingface_hub import snapshot_download
            revision = resolve_hub_revision(SEMANTIC_REPO, resolved.adversary.get('semantic_revision'))
            semantic_dir = snapshot_download(SEMANTIC_REPO, revision=revision,
                allow_patterns=list(SEMANTIC_FILES), cache_dir=resolved.adversary.get('hf_cache_dir'))
            OmegaConf.update(resolved, 'adversary.semantic_revision', revision, force_add=True)
            assets['maskgct.semantic_hub'] = {'repo_id': SEMANTIC_REPO, 'revision': revision}
        paths = semantic_files(semantic_dir)
        OmegaConf.update(resolved, 'adversary.semantic_model_path', str(Path(semantic_dir).expanduser().absolute()), force_add=True)
        for filename, path in zip(SEMANTIC_FILES, paths):
            references['maskgct.semantic.' + filename] = str(path)
        if resolved.adversary.get('code_path'):
            references['maskgct.semantic_stats'] = str(Path(resolved.adversary.code_path).expanduser() /
                'models/tts/maskgct/ckpt/wav2vec2bert_stats.pt')
            references['maskgct.g2p_resources'] = str(Path(resolved.adversary.code_path).expanduser() /
                'models/tts/maskgct/g2p')
            import sys
            espeak = _maskgct_espeak_resources(resolved.adversary.get('runtime_python') or sys.executable)
            assets['maskgct.espeak_runtime'] = espeak
            references['maskgct.espeak_library'] = espeak['library_path']
            references['maskgct.espeak_data'] = espeak['data_path']
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
        from rvcbench.models.playdiffusion.generator import PRESET_FILES, PlayDiffusionGeneratorConfig
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
            files = [p for p in sorted(path.rglob('*')) if p.is_file() and
                     (key == 'maskgct.espeak_data' or key.startswith('openvoice.english_resource.') or p.suffix in suffixes)]
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
    if model == 'maskgct' and 'maskgct.espeak_runtime' in assets:
        expected = dict(assets['maskgct.espeak_runtime'])
        expected['library_sha256'] = next(iter(assets['maskgct.espeak_library']['files'].values()))
        expected['data_sha256'] = digest(assets['maskgct.espeak_data']['files'])
        expected['data_file_count'] = len(assets['maskgct.espeak_data']['files'])
        OmegaConf.update(resolved, 'adversary.espeak_runtime', expected, force_add=True)
    reference = {'assets': assets, 'unresolved_references': unresolved,
                 'coverage': 'configured_local_assets_and_supported_hub_resolvers',
                 'note': 'Implicit upstream downloads and service-side models require additional provenance.'}
    return resolved, reference, digest(reference)
