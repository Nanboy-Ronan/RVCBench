"""Model wrappers are imported on demand to isolate runtime dependencies."""
def __getattr__(name):
    if name == 'BertVits2Wrapper':
        from .bertvits2_model import BertVits2Wrapper
        return BertVits2Wrapper
    if name == 'OZSpeechWrapper':
        from .ozspeech_model import OZSpeechWrapper
        return OZSpeechWrapper
    raise AttributeError(name)
