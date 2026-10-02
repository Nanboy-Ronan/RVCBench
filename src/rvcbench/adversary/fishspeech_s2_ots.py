"""Serial native S2 runtime without a background initialization thread."""
from contextlib import nullcontext
import importlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .fishspeech_ots import FishSpeechZeroShotAdversary


class SerialSemanticQueue:
    """Execute native generation on the requesting thread; errors propagate."""
    def __init__(self, model, decode, native):
        self.model, self.decode, self.native = model, decode, native

    def put(self, item):
        if item is None:
            return
        for chunk in self.native.generate_long(
                model=self.model, decode_one_token=self.decode, **item.request):
            item.response_queue.put(self.native.WrappedGenerateResponse(
                status='success', response=chunk))


def load_shards(path, loader):
    path = Path(path)
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    state = {}
    for name in sorted(set(index['weight_map'].values())):
        shard_path = (path / name).resolve()
        if not shard_path.is_relative_to(path.resolve()):
            raise ValueError('S2 shard path escapes checkpoint directory')
        shard = loader(str(shard_path), device='cpu')
        if state.keys() & shard.keys():
            raise ValueError('S2 checkpoint has duplicate tensor keys across shards')
        state.update(shard)
    if set(state) != set(index['weight_map']):
        raise ValueError('S2 checkpoint tensors differ from its shard index')
    return state


def load_codec_state(codec, state):
    state = dict(state)
    for key in state.keys() - codec.state_dict().keys():
        parent, name = key.rsplit('.', 1) if '.' in key else ('', key)
        module = codec.get_submodule(parent)
        if name not in {'freqs_cis', 'causal_mask'} or name not in module._non_persistent_buffers_set:
            raise ValueError(f'Unexpected S2 codec weight: {key}')
        cached, rebuilt = state[key], getattr(module, name)
        if (rebuilt is None or cached.dtype != rebuilt.dtype or cached.ndim != rebuilt.ndim
                or any(a > b for a, b in zip(cached.shape, rebuilt.shape))
                or not torch.equal(cached, rebuilt[tuple(slice(0, n) for n in cached.shape)])):
            raise ValueError(f'S2 codec cached buffer differs from native reconstruction: {key}')
        del state[key]
    codec.load_state_dict(state, strict=True, assign=True)


class FishSpeechS2ZeroShotAdversary(FishSpeechZeroShotAdversary):
    def _scope(self):
        return torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()

    def _ensure_model(self):
        if self._tts_engine is not None:
            return
        if self.compile:
            raise ValueError('Measured serial S2 runtime requires compile=false')
        if self.half:
            raise ValueError('Released S2 runtime requires bfloat16; set half=false')
        if self.decoder_config_name != 'modded_dac_vq':
            raise ValueError('Released S2 requires its native modded_dac_vq codec config')
        native = importlib.import_module('fish_speech.models.text2semantic.inference')
        llama = importlib.import_module('fish_speech.models.text2semantic.llama')
        tokenizer = importlib.import_module('fish_speech.tokenizer')
        engine = importlib.import_module('fish_speech.inference_engine')
        schema = importlib.import_module('fish_speech.utils.schema')
        for module in (native, llama, tokenizer, engine, schema):
            if not Path(module.__file__).resolve().is_relative_to(self.code_path):
                raise ImportError('S2 runtime imported outside configured code_path')
        from hydra.utils import instantiate
        from safetensors.torch import load_file
        self._semantic_model = self._decoder_model = None
        try:
            with self._scope():
                config = llama.BaseModelArgs.from_pretrained(str(self.llama_checkpoint_path))
                tokens = tokenizer.FishTokenizer.from_pretrained(self.llama_checkpoint_path)
                config.semantic_begin_id, config.semantic_end_id = tokens.semantic_begin_id, tokens.semantic_end_id
                model = llama.DualARTransformer(config)
                model.tokenizer = tokens
                state = llama._remap_fish_qwen3_omni_keys(load_shards(self.llama_checkpoint_path, load_file))
                model.load_state_dict(state, strict=True, assign=True)
                del state
                model = model.to(device=self.device, dtype=torch.bfloat16).eval()
                self._semantic_model = model
                with torch.device(self.device):
                    model.setup_caches(max_batch_size=1, max_seq_len=config.max_seq_len, dtype=torch.bfloat16)
                model.fixed_temperature = torch.tensor(.7, device=self.device)
                model.fixed_top_p = torch.tensor(.7, device=self.device)
                model.fixed_repetition_penalty = torch.tensor(1.5, device=self.device)
                model._cache_setup_done = False
                codec_config = OmegaConf.load(self.code_path / 'fish_speech/configs/modded_dac_vq.yaml')
                codec = instantiate(codec_config)
                state = torch.load(self.decoder_checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
                state = state.get('state_dict', state)
                if any(key.startswith('generator.') for key in state):
                    if not all(key.startswith('generator.') for key in state):
                        raise ValueError('S2 codec contains unrelated non-generator weights')
                    state = {key.removeprefix('generator.'): value for key, value in state.items()}
                load_codec_state(codec, state)
                del state
                codec = codec.to(self.device).eval()
                self._decoder_model = codec
                self._tts_engine = engine.TTSInferenceEngine(
                    llama_queue=SerialSemanticQueue(model, native.decode_one_token_ar, native),
                    decoder_model=codec, precision=torch.bfloat16, compile=False)
                self._ServeTTSRequest, self._ServeReferenceAudio = schema.ServeTTSRequest, schema.ServeReferenceAudio
        except BaseException:
            self.close()
            raise

    def _build_request(self, text, reference_path, reference_text=None, sample_index=0):
        if not str(text or '').strip() or not str(reference_text or '').strip():
            raise ValueError('S2 requires actual target and reference transcripts')
        return super()._build_request(text, reference_path, reference_text, sample_index)

    def _run_inference(self, request):
        with self._scope():
            return super()._run_inference(request)

    def attack(self, *, output_path, dataset, protected_audio_path=None):
        for sample in dataset.get_zero_shot_samples(max_samples=self.max_samples):
            if not str(sample.target_text or '').strip() or not str(sample.prompt_text or '').strip():
                raise ValueError('S2 requires actual target and reference transcripts')
        return super().attack(output_path=output_path, dataset=dataset,
                              protected_audio_path=protected_audio_path)

    def close(self):
        self._tts_engine = self._semantic_model = self._decoder_model = None
        super().close()
