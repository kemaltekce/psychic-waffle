"""Log-mel features and optional training-only audio augmentations."""

import torch
from torchaudio.transforms import MelSpectrogram

from psychic.data.preprocessing import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SAMPLE_RATE_HZ,
)

LOG_FLOOR = 1e-10
STD_FLOOR = 1e-8


def build_log_mel_config() -> dict:
    """Return the feature contract; window sizes are measured in samples."""
    return {
        "name": "log_mel_v1",
        "mel_spectrogram": {
            "sample_rate": DEFAULT_SAMPLE_RATE_HZ,
            "n_fft": 1024,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 64,
            "f_min": 0.0,
            "f_max": 8000.0,
            "power": 2.0,
            "pad": 0,
            "center": True,
            "pad_mode": "reflect",
            "normalized": False,
            "norm": "slaney",
            "mel_scale": "slaney",
        },
        "window": "hann_periodic",
        "log": "natural",
        "log_floor": LOG_FLOOR,
        "normalization": "per_clip_mean_population_std",
        "std_floor": STD_FLOOR,
    }


# Reuse the CPU filter bank/window outside the model so .to(device) leaves
# the STFT on CPU, including when the CNN runs on MPS.
_MEL_TRANSFORM = MelSpectrogram(**build_log_mel_config()["mel_spectrogram"])


@torch.no_grad()
def time_shift(
    waveforms: torch.Tensor, max_shift_ms: float = 50.0
) -> torch.Tensor:
    """Independently shift CPU float32 `[batch, 48000]` clips left or right.

    Sample uniformly within +/- max_shift_ms at 16 kHz. Preserve length by
    zero padding the exposed edge and discarding the opposite edge; speech
    there may be truncated. Never wrap audio or modify the input. Randomness
    follows the CPU PyTorch seed. Zero disables shifting without drawing RNG.
    """
    assert waveforms.device.type == "cpu"
    assert waveforms.dtype == torch.float32
    assert waveforms.ndim == 2 and waveforms.shape[1] == DEFAULT_NUM_SAMPLES
    assert torch.isfinite(waveforms).all().item()
    assert (
        0 <= max_shift_ms < 1000 * DEFAULT_NUM_SAMPLES / DEFAULT_SAMPLE_RATE_HZ
    )
    max_shift_samples = int(max_shift_ms * DEFAULT_SAMPLE_RATE_HZ / 1000)
    if max_shift_samples == 0:
        return waveforms

    shifts = torch.randint(
        -max_shift_samples, max_shift_samples + 1, (len(waveforms),)
    )
    shifted = torch.zeros_like(waveforms)
    # Each target is a row view into shifted; slice assignment fills that row.
    for source, target, shift in zip(
        waveforms, shifted, shifts.tolist(), strict=True
    ):
        if shift > 0:
            target[shift:] = source[:-shift]
        elif shift < 0:
            target[:shift] = source[-shift:]
        else:
            target.copy_(source)
    assert shifted.shape == waveforms.shape
    return shifted


@torch.no_grad()
def time_mask(
    features: torch.Tensor,
    max_frames: int = 10,
    probability: float = 0.5,
) -> torch.Tensor:
    """Mask one interval per selected CPU float32 `[batch, 1, mel, time]` clip.

    After normalization, fill all mel bins in a random 1..max_frames interval
    with zero (the original clip mean). Each clip is selected independently
    with probability; frames are 10 ms apart in the current log-mel features.
    Preserve shape and input contents. Randomness follows the CPU PyTorch
    seed; zero probability or max_frames disables masking without RNG draws.
    """
    assert features.device.type == "cpu"
    assert features.dtype == torch.float32
    assert features.ndim == 4 and features.shape[1] == 1
    assert features.shape[2] > 0 and features.shape[3] > 0
    assert torch.isfinite(features).all().item()
    assert isinstance(max_frames, int) and 0 <= max_frames <= features.shape[3]
    assert 0 <= probability <= 1
    if max_frames == 0 or probability == 0:
        return features

    masked = features.clone()
    selected = torch.rand(len(features)) < probability
    for index in selected.nonzero().flatten().tolist():
        width = torch.randint(1, max_frames + 1, ()).item()
        start = torch.randint(0, features.shape[3] - width + 1, ()).item()
        masked[index, :, :, start : start + width] = 0
    assert masked.shape == features.shape
    return masked


@torch.no_grad()
def waveforms_to_log_mel(waveforms: torch.Tensor) -> torch.Tensor:
    """Convert CPU float32 `[batch, 48000]` to `[batch, 1, 64, 301]`.

    Natural-log power is standardized independently per clip, identically
    for train/eval/inference. Silence stays finite. The STFT stays on CPU
    so accelerator training does not require complex-number MPS kernels.
    """
    assert waveforms.device.type == "cpu", "transform waveforms on CPU"
    assert waveforms.dtype == torch.float32
    assert waveforms.ndim == 2 and waveforms.shape[1] == DEFAULT_NUM_SAMPLES, (
        "expected [batch, 48000] waveforms"
    )
    features = _MEL_TRANSFORM(waveforms).clamp_min(LOG_FLOOR).log()
    std, mean = torch.std_mean(
        features, dim=(-2, -1), correction=0, keepdim=True
    )
    features = ((features - mean) / std.clamp_min(STD_FLOOR)).unsqueeze(1)
    assert features.shape == (waveforms.shape[0], 1, 64, 301)
    assert torch.isfinite(features).all().item()
    return features
