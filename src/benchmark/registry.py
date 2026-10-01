"""Lazy adapter registry, independent of workflow orchestration."""
import importlib
from typing import Dict

_ADVERSARY_REGISTRY: Dict[str, Dict[str, str]] = {
    "ots": {
        "smoke": "src.adversary.smoke:SmokeAdversary",
        "bertvits2": "src.adversary.bertvit2_ots:BertVits2ZeroShotAdversary",
        "ozspeech": "src.adversary.ozspeech_ots:OzSpeechZeroShotAdversary",
        "higgs_audio": "src.adversary.higgs_audio_ots:HiggsAudioZeroShotAdversary",
        "cosyvoice": "src.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cozyvoice": "src.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cozyvoice2": "src.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cosyvoice2": "src.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "sparktts": "src.adversary.sparktts_ots:SparkTTSZeroShotAdversary",
        "vall_e": "src.adversary.vall_e_ots:VallEZeroShotAdversary",
        "styletts2": "src.adversary.styletts2_ots:StyleTTS2ZeroShotAdversary",
        "glowtts": "src.adversary.glowtts_ots:GlowTTSZeroShotAdversary",
        "glm_tts": "src.adversary.glmtts_ots:GLMTTSZeroShotAdversary",
        "glmtts": "src.adversary.glmtts_ots:GLMTTSZeroShotAdversary",
        "kimi_audio": "src.adversary.kimi_audio_ots:KimiAudioZeroShotAdversary",
        "moss_ttsd": "src.adversary.moss_ttsd_ots:MossTTSDZeroShotAdversary",
        "playdiffusion": "src.adversary.playdiffusion_ots:PlayDiffusionZeroShotAdversary",
        "bark_voice_clone": "src.adversary.bark_voice_clone_ots:BarkVoiceCloneZeroShotAdversary",
        "fishspeech": "src.adversary.fishspeech_ots:FishSpeechZeroShotAdversary",
        "fishspeech_s2": "src.adversary.fishspeech_s2_ots:FishSpeechS2ZeroShotAdversary",
        "qwen3_omni": "src.adversary.qwen3_omni_ots:Qwen3OmniZeroShotAdversary",
        "mgm_omni": "src.adversary.mgm_omni_ots:MGMOmniZeroShotAdversary",
        "vibevoice": "src.adversary.vibevoice_ots:VibeVoiceZeroShotAdversary",
        "f5_tts": "src.adversary.f5_tts_ots:F5TTSZeroShotAdversary",
        "qwen3_tts": "src.adversary.qwen3_tts_ots:Qwen3TTSZeroShotAdversary",
        "qwentts": "src.adversary.qwen3_tts_ots:Qwen3TTSZeroShotAdversary",
        "fireredtts2": "src.adversary.fireredtts2_ots:FireRedTTS2ZeroShotAdversary",
        "firered_tts2": "src.adversary.fireredtts2_ots:FireRedTTS2ZeroShotAdversary",
        "voxcpm": "src.adversary.voxcpm_ots:VoxCPMZeroShotAdversary",
        "maskgct": "src.adversary.maskgct_ots:MaskGCTZeroShotAdversary",
        "openvoice": "src.adversary.openvoice_ots:OpenVoiceZeroShotAdversary",
        "xtts": "src.adversary.xtts_ots:XttsZeroShotAdversary",
        "xtts_v2": "src.adversary.xtts_ots:XttsZeroShotAdversary",
        "index_tts": "src.adversary.index_tts_ots:IndexTTSZeroShotAdversary",
        "indextts": "src.adversary.index_tts_ots:IndexTTSZeroShotAdversary",
        "zipvoice": "src.adversary.zipvoice_ots:ZipVoiceZeroShotAdversary",
        "dots_tts": "src.adversary.dots_tts_ots:DotsTTSZeroShotAdversary",
        "dotstts": "src.adversary.dots_tts_ots:DotsTTSZeroShotAdversary",
        "zonos2": "src.adversary.zonos2_ots:Zonos2ZeroShotAdversary",
        "zonos_2": "src.adversary.zonos2_ots:Zonos2ZeroShotAdversary",
        "moss_tts": "src.adversary.moss_tts_ots:MossTTSZeroShotAdversary",
        "moss_tts_local": "src.adversary.moss_tts_ots:MossTTSZeroShotAdversary",
        "fish_audio_s2": "src.adversary.fish_audio_s2_server_ots:FishAudioS2ServerZeroShotAdversary",
        "fish_speech_s2": "src.adversary.fish_audio_s2_server_ots:FishAudioS2ServerZeroShotAdversary",
        "fishaudio_s2": "src.adversary.fish_audio_s2_server_ots:FishAudioS2ServerZeroShotAdversary",
        "higgs_tts_3": "src.adversary.openai_speech_server_ots:OpenAISpeechServerZeroShotAdversary",
        "higgs_audio_v3": "src.adversary.openai_speech_server_ots:OpenAISpeechServerZeroShotAdversary",
    },
    "finetune": {
        "bertvits2": "src.adversary.bertvits2_finetune:BertVits2FinetuneAdversary",
    },
}

def select_adversary(conf, dataset_conf, device, logger):
    mode = str(conf.vc.mode).lower()
    model = str(conf.vc.get("model", "bertvits2")).lower()

    mode_registry = _ADVERSARY_REGISTRY.get(mode)
    if mode_registry is None:
        raise ValueError(f"Unsupported vc.mode '{conf.vc.mode}'. Expected one of: {', '.join(_ADVERSARY_REGISTRY)}")

    target = mode_registry.get(model)
    if target is None:
        raise ValueError(
            f"Unsupported adversary model '{model}' for mode '{mode}'. Available models: {', '.join(mode_registry.keys())}"
        )

    module_path, class_name = target.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import adversary module '{module_path}' for model '{model}'."
        ) from exc

    try:
        adversary_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Adversary class '{class_name}' not found in module '{module_path}'."
        ) from exc

    return adversary_cls(conf, dataset_conf, device, logger)
