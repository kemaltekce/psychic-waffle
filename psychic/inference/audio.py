"""Bounded microphone buffering, independent of model and device loading."""

from collections import deque
from math import ceil
from time import monotonic

import numpy as np

from psychic.data.preprocessing import DEFAULT_DURATION_SEC

BLOCK_DURATION_SEC = 0.02


class AudioBuffer:
    """Keep the latest three seconds of mono float32 microphone blocks.

    One capture callback appends fixed-size blocks; the main thread snapshots
    them. Copy the deque before NumPy concatenation so capture can continue
    while the main thread assembles its independent waveform.
    """

    def __init__(self, sample_rate_hz: int) -> None:
        assert sample_rate_hz > 0
        self.window_frames = round(sample_rate_hz * DEFAULT_DURATION_SEC)
        self.block_frames = round(sample_rate_hz * BLOCK_DURATION_SEC)
        assert self.block_frames > 0
        self.blocks: deque[np.ndarray] = deque(
            maxlen=ceil(self.window_frames / self.block_frames)
        )
        self.last_received_sec = monotonic()
        self.error: str | None = None

    def append(self, samples: np.ndarray) -> None:
        """Copy one mono block; the audio driver may reuse its input memory."""
        assert samples.shape == (self.block_frames,)
        assert samples.dtype == np.float32
        self.blocks.append(samples.copy())
        self.last_received_sec = monotonic()

    def snapshot(self) -> np.ndarray | None:
        """Return the newest complete window, or None during initial fill."""
        blocks = self.blocks.copy()
        if len(blocks) < blocks.maxlen:
            return None
        waveform = np.concatenate(blocks)[-self.window_frames :]
        assert waveform.shape == (self.window_frames,)
        assert waveform.dtype == np.float32
        return waveform
