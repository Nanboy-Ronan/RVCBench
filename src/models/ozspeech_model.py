from src.models.model import BaseModel
from pathlib import  Path
from hydra.utils import to_absolute_path
import sys
import importlib
import os
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf


class OZSpeechWrapper(BaseModel):
    def __init__(self, config, model_config, logger, **kwargs):
        super().__init__(config, model_config, logger)
        self._imports_loaded = False
        self.code_path = Path(to_absolute_path(model_config.code_path)).resolve()
        self.max_wav_value = kwargs.get("max_wav_value", 16000)
        ckpt_key = "checkpoint_path" if "checkpoint_path" in self.model_config else "ckpt_path"
        cfg_key = "config_path" if "config_path" in self.model_config else "cfg_path"
        if ckpt_key not in self.model_config or cfg_key not in self.model_config:
            raise ValueError(
                "OZSpeech adversary config must provide 'checkpoint_path'/'ckpt_path' and 'config_path'/'cfg_path'.")

        self.checkpoint_path = Path(to_absolute_path(self.model_config[ckpt_key])).resolve()
        self.config_path = Path(to_absolute_path(self.model_config[cfg_key])).resolve()
        self.load_model()

    def load_model(self):
        self._ensure_imports()
        cfg = OmegaConf.load(self.config_path,)# additionally add ZACT module
        self.model = self._ZACT_cls.from_pretrained(cfg=cfg,
                                          ckpt_path=self.checkpoint_path,
                                          device=self.device,
                                          training_mode=False)
        self.temperature = float(self.model_config.get("model_temperature", 0.01)) # todo add temperature
        self.model.train()

    def _ensure_imports(self):
        if self._imports_loaded:
            return

        if not self.code_path.exists():
            raise FileNotFoundError(f"OZSpeech code path not found: {self.code_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"OZSpeech checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"OZSpeech config not found: {self.config_path}")

        prev_cwd = Path.cwd()
        try:
            if str(self.code_path) not in sys.path:
                sys.path.insert(0, str(self.code_path))
            os.chdir(self.code_path)
            synth_module = importlib.import_module("synthesize")

            zact_module = importlib.import_module('zact')
            self._ZACT_cls = getattr(zact_module, 'ZACT')

        finally:
            os.chdir(prev_cwd)

        self._synthesize_fn = synth_module.synthesize
        self.fa_encoder, self.fa_decoder = synth_module.get_codec(self.checkpoint_path.parent)

        if self.device.find("cuda")!=-1:
            index = 0 if self.device.index is None else self.device.index
            self._device_str = f"cuda:{index}"
        else:
            self._device_str = "cpu"

        self._imports_loaded = True

    def to(self, device):
        self.model.to(device)
        self.fa_encoder.to(device)
        self.fa_decoder.to(device)

    def generate(self, input, **kwargs):
        gen_wav= []
        for idx, in_text in enumerate(input.text_raw):
            breakpoint()
            output = self.model.synthesize(
                text=in_text,
                acoustic_prompt=input.audio_raw[idx][:,:input.wav_len[idx]],
                codec_encoder=self.fa_encoder,
                codec_decoder=self.fa_decoder,
                temperature=self.temperature,
            )
            breakpoint()

            gen_wav.append(torch.from_numpy(output['synth_wav']))
            # path1= os.path.join('/home/xenial/projects/rpp-xli135/xenial/AudioWatermarkBench/results/tmp',  f'raw_{idx}.wav')
            # path2= os.path.join('/home/xenial/projects/rpp-xli135/xenial/AudioWatermarkBench/results/tmp',  f'gen_{idx}.wav')
            # SR = 16000
            # sf.write(
            #     file=path1,
            #     data=input.wav[idx].squeeze().detach().cpu().numpy(),
            #     samplerate=SR,
            #     format='WAV'
            # )
            #
            # sf.write(
            #     file=path2,
            #     data=output['synth_wav'],
            #     samplerate=SR,
            #     format='WAV'
            # )
        breakpoint()
        gen_wav = pad_sequence(gen_wav)
        gen_wav = gen_wav / self.max_wav_value
        gen_wav = gen_wav.unsqueeze(0)
        return gen_wav #pad_sequence(gen_wav).unsqueeze(1).transpose(0, 2)
