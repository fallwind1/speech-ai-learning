from pathlib import Path
import torch

from src.audio_utils import load_audio,to_mono,resample_audio,save_audio

def test_audio_pipeline():
    input_file = "data/audio/test.mp3"

    waveform,sr = load_audio(input_file)

    print(
        "original:",
        waveform.shape,
        sr
    )

    mono = to_mono(waveform)

    print(
        "mono:",
        mono.shape
    )

    resampled = resample_audio(waveform,sr,16000)

    print("resampled:",resampled.shape)

    output_file = "data/audio/test_16k.wav"

    save_audio(output_file,resampled,16000)

    wav2,sr2 = load_audio(output_file)

    print(
        "reload:",
        wav2.shape,
        sr2
    )

    assert sr2 == 16000
    assert wav2.shape[1] == resampled.shape[1]

def test_stereo_to_mono():
    stereo = torch.tensor([
        [0.2,0.4,0.6],
        [0.6,0.4,0.2]
    ])

    mono = to_mono(stereo)

    expected = torch.tensor([
        [0.4,0.4,0.4]
    ])

    assert mono.shape == expected.shape
    assert torch.allclose(mono,expected)

    print("Stereo to mono: PASS")

def test_invalid_Path():
    try:
        load_audio("data/audio/not_exist.wav")

    except FileNotFoundError:
        print("Missing file error: PASS")
        return

    raise AssertionError(
        "Expected FileNotFoundError"
    )

if __name__ == "__main__":
    test_audio_pipeline()
    test_stereo_to_mono()
    test_invalid_Path()