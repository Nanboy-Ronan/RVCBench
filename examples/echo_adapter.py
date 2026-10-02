"""Minimal RVCBench adapter: returns the reference audio instead of synthesizing speech.

Use it to check an installation end to end, or as a starting point for your own model:

    PYTHONPATH=examples rvcbench run --config-name ots_vc/clean/libritts/custom_ots \
        vc.adapter=echo_adapter:EchoAdapter vc.model=echo run_name=echo_on_libritts \
        +vc.generate_only=true adversary.max_samples=5 device=cpu
"""
import soundfile as sf

from rvcbench import VoiceCloningAdapter


class EchoAdapter(VoiceCloningAdapter):
    def load(self):
        # Load model weights here. `self.config` holds the `adversary` block of the run
        # config and `self.device` the requested device.
        self.gain = float(self.config.get("gain", 1.0))

    def clone(self, *, text, reference_audio, reference_text, language):
        # Replace this with your model: synthesize `text` in the voice of `reference_audio`.
        waveform, sample_rate = sf.read(str(reference_audio), dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform * self.gain, sample_rate
