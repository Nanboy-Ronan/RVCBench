import numpy as np
import pytest
import soundfile as sf

from rvcbench.evaluation.audio_io import audio_info, load_audio, save_audio


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    return (0.5 * rng.uniform(-1, 1, size=(2400, 2))).astype('float32')


@pytest.mark.parametrize('subtype', ['PCM_16', 'PCM_24', 'FLOAT'])
@pytest.mark.parametrize('channels', [1, 2])
def test_load_matches_torchaudio(tmp_path, signal, subtype, channels):
    torchaudio = pytest.importorskip('torchaudio')
    path = tmp_path / 'a.wav'
    sf.write(path, signal[:, :channels], 16000, subtype=subtype)
    try:
        expected, expected_rate = torchaudio.load(str(path))
    except (ImportError, RuntimeError):  # torchaudio >= 2.9 needs torchcodec for file input
        pytest.skip('torchaudio.load is unavailable')
    waveform, rate = load_audio(path)
    assert rate == expected_rate and waveform.dtype == expected.dtype and waveform.shape == expected.shape
    assert (waveform == expected).all()


def test_partial_read_info_and_float_round_trip(tmp_path, signal):
    path = tmp_path / 'a.wav'
    sf.write(path, signal, 24000, subtype='FLOAT')
    assert audio_info(path) == (24000, 2400)
    head, _ = load_audio(path, frame_offset=0, num_frames=1000)
    assert head.shape == (2, 1000)
    save_audio(tmp_path / 'b.wav', head, 24000)
    assert sf.info(str(tmp_path / 'b.wav')).subtype == 'FLOAT'
    again, rate = load_audio(tmp_path / 'b.wav')
    assert rate == 24000 and (again == head).all()
