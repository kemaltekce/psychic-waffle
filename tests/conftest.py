import struct
import wave
from collections.abc import Callable
from pathlib import Path

import pytest
import torch


@pytest.fixture
def write_wav_file() -> Callable[[Path, torch.Tensor, int], None]:
    def write(
        sample_path: Path,
        waveform: torch.Tensor,
        sample_rate_hz: int,
    ) -> None:
        assert waveform.ndim == 2, "test waveform must be [channels, frames]"
        assert sample_rate_hz > 0, "test sample rate must be positive"

        sample_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (
            waveform.clamp(-1.0, 1.0)
            .mul(32767)
            .round()
            .to(torch.int16)
        )
        frames = pcm.transpose(0, 1).contiguous().view(-1).tolist()

        with wave.open(str(sample_path), "wb") as wav_file:
            wav_file.setnchannels(waveform.shape[0])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate_hz)
            wav_file.writeframes(struct.pack(f"<{len(frames)}h", *frames))

    return write
