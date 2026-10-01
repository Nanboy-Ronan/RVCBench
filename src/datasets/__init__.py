"""Load training datasets only when requested."""
def __getattr__(name):
    if name in {'AllSpeakerData', 'TextAudioSpeakerDataset'}:
        from . import data_utils
        return getattr(data_utils, name)
    if name == 'ZeroShotDataset':
        from .zero_shot import ZeroShotDataset
        return ZeroShotDataset
    raise AttributeError(name)
