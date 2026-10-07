import torch

from psychic.data.preprocessing import (
    pad_or_crop,
    preprocess_audio_file,
    preprocess_waveform,
)


def test_pad_or_crop_is_deterministic() -> None:
    short_waveform = torch.tensor([1.0, 2.0, 3.0])
    long_waveform = torch.arange(9, dtype=torch.float32)

    padded = pad_or_crop(short_waveform, target_num_samples=7)
    cropped = pad_or_crop(long_waveform, target_num_samples=5)

    assert torch.equal(
        padded,
        torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]),
    )
    assert torch.equal(cropped, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))


def test_preprocess_waveform_converts_to_mono_float32() -> None:
    waveform = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=torch.float64,
    )

    fixed = preprocess_waveform(
        waveform,
        source_sample_rate_hz=4,
        target_sample_rate_hz=4,
        target_num_samples=4,
    )

    assert fixed.dtype == torch.float32
    assert fixed.shape == (4,)
    assert torch.equal(fixed, torch.tensor([3.0, 4.0, 5.0, 6.0]))


def test_preprocess_audio_file_returns_fixed_shape_and_source_facts(
    tmp_path,
    write_wav_file,
) -> None:
    sample_path = tmp_path / "sample.wav"
    source_waveform = torch.linspace(-0.5, 0.5, steps=1000).unsqueeze(0)
    write_wav_file(sample_path, source_waveform, 8000)
    waveform, duration_sec = preprocess_audio_file(
        sample_path,
        target_sample_rate_hz=8000,
        target_num_samples=2000,
    )

    assert waveform.dtype == torch.float32
    assert waveform.shape == (2000,)
    assert duration_sec == 0.125
    assert torch.isfinite(waveform).all()
