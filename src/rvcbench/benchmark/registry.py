"""Lazy adapter registry, independent of workflow orchestration."""
import importlib
from typing import Dict

_ADVERSARY_REGISTRY: Dict[str, Dict[str, str]] = {
    "ots": {
        "smoke": "rvcbench.adversary.smoke:SmokeAdversary",
        "bertvits2": "rvcbench.adversary.bertvit2_ots:BertVits2ZeroShotAdversary",
        "ozspeech": "rvcbench.adversary.ozspeech_ots:OzSpeechZeroShotAdversary",
        "higgs_audio": "rvcbench.adversary.higgs_audio_ots:HiggsAudioZeroShotAdversary",
        "cosyvoice": "rvcbench.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cozyvoice": "rvcbench.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cozyvoice2": "rvcbench.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "cosyvoice2": "rvcbench.adversary.cosyvoice_ots:CosyVoiceZeroShotAdversary",
        "sparktts": "rvcbench.adversary.sparktts_ots:SparkTTSZeroShotAdversary",
        "vall_e": "rvcbench.adversary.vall_e_ots:VallEZeroShotAdversary",
        "styletts2": "rvcbench.adversary.styletts2_ots:StyleTTS2ZeroShotAdversary",
        "glowtts": "rvcbench.adversary.glowtts_ots:GlowTTSZeroShotAdversary",
        "glm_tts": "rvcbench.adversary.glmtts_ots:GLMTTSZeroShotAdversary",
        "glmtts": "rvcbench.adversary.glmtts_ots:GLMTTSZeroShotAdversary",
        "kimi_audio": "rvcbench.adversary.kimi_audio_ots:KimiAudioZeroShotAdversary",
        "moss_ttsd": "rvcbench.adversary.moss_ttsd_ots:MossTTSDZeroShotAdversary",
        "playdiffusion": "rvcbench.adversary.playdiffusion_ots:PlayDiffusionZeroShotAdversary",
        "bark_voice_clone": "rvcbench.adversary.bark_voice_clone_ots:BarkVoiceCloneZeroShotAdversary",
        "fishspeech": "rvcbench.adversary.fishspeech_ots:FishSpeechZeroShotAdversary",
        "fishspeech_s2": "rvcbench.adversary.fishspeech_s2_ots:FishSpeechS2ZeroShotAdversary",
        "qwen3_omni": "rvcbench.adversary.qwen3_omni_ots:Qwen3OmniZeroShotAdversary",
        "mgm_omni": "rvcbench.adversary.mgm_omni_ots:MGMOmniZeroShotAdversary",
        "vibevoice": "rvcbench.adversary.vibevoice_ots:VibeVoiceZeroShotAdversary",
        "f5_tts": "rvcbench.adversary.f5_tts_ots:F5TTSZeroShotAdversary",
        "qwen3_tts": "rvcbench.adversary.qwen3_tts_ots:Qwen3TTSZeroShotAdversary",
        "qwentts": "rvcbench.adversary.qwen3_tts_ots:Qwen3TTSZeroShotAdversary",
        "fireredtts2": "rvcbench.adversary.fireredtts2_ots:FireRedTTS2ZeroShotAdversary",
        "firered_tts2": "rvcbench.adversary.fireredtts2_ots:FireRedTTS2ZeroShotAdversary",
        "voxcpm": "rvcbench.adversary.voxcpm_ots:VoxCPMZeroShotAdversary",
        "maskgct": "rvcbench.adversary.maskgct_ots:MaskGCTZeroShotAdversary",
        "openvoice": "rvcbench.adversary.openvoice_ots:OpenVoiceZeroShotAdversary",
        "xtts": "rvcbench.adversary.xtts_ots:XttsZeroShotAdversary",
        "xtts_v2": "rvcbench.adversary.xtts_ots:XttsZeroShotAdversary",
        "index_tts": "rvcbench.adversary.index_tts_ots:IndexTTSZeroShotAdversary",
        "indextts": "rvcbench.adversary.index_tts_ots:IndexTTSZeroShotAdversary",
        "zipvoice": "rvcbench.adversary.zipvoice_ots:ZipVoiceZeroShotAdversary",
        "dots_tts": "rvcbench.adversary.dots_tts_ots:DotsTTSZeroShotAdversary",
        "dotstts": "rvcbench.adversary.dots_tts_ots:DotsTTSZeroShotAdversary",
        "zonos2": "rvcbench.adversary.zonos2_ots:Zonos2ZeroShotAdversary",
        "zonos_2": "rvcbench.adversary.zonos2_ots:Zonos2ZeroShotAdversary",
        "moss_tts": "rvcbench.adversary.moss_tts_ots:MossTTSZeroShotAdversary",
        "moss_tts_local": "rvcbench.adversary.moss_tts_ots:MossTTSZeroShotAdversary",
        "fish_audio_s2": "rvcbench.adversary.fish_audio_s2_server_ots:FishAudioS2ServerZeroShotAdversary",
        "fish_speech_s2": "rvcbench.adversary.fish_audio_s2_server_ots:FishAudioS2ServerZeroShotAdversary",
        "fishaudio_s2": "rvcbench.adversary.fish_audio_s2_server_ots:FishAudioS2ServerZeroShotAdversary",
        "higgs_tts_3": "rvcbench.adversary.openai_speech_server_ots:OpenAISpeechServerZeroShotAdversary",
        "higgs_audio_v3": "rvcbench.adversary.openai_speech_server_ots:OpenAISpeechServerZeroShotAdversary",
    },
    "finetune": {
        "bertvits2": "rvcbench.adversary.bertvits2_finetune:BertVits2FinetuneAdversary",
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
