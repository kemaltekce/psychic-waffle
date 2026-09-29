"""Deterministic log-mel features shared by CNN training and inference."""

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
