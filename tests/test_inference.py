"""Check streaming transformations without a microphone or device mocks."""

import numpy as np
import pytest
import torch

from psychic.data.preprocessing import preprocess_waveform
from psychic.inference.audio import AudioBuffer
from psychic.inference.live import (
    SMOOTHING_ALPHA,
    VOLUME_THRESHOLD_DBFS,
    LivePredictor,
    format_display,
    is_audible,
    latest_checkpoint,
)
from psychic.labels import EMOTION_LABELS
from psychic.training.model import CNN


@pytest.mark.parametrize("sample_rate_hz", [16_000, 44_100, 48_000, 12_345])
def test_buffer_keeps_latest_window_and_owns_its_samples(sample_rate_hz):
    buffer = AudioBuffer(sample_rate_hz)
    generator = np.random.default_rng(42)
    history = []
    assert buffer.snapshot() is None
    for _ in range(3 * buffer.blocks.maxlen):
        block = generator.normal(size=buffer.block_frames).astype(np.float32)
        history.append(block.copy())
        buffer.append(block)
        block.fill(0)  # The microphone driver reuses its input storage.
        snapshot = buffer.snapshot()
        if len(history) < buffer.blocks.maxlen:
            assert snapshot is None
            continue
        expected = np.concatenate(history)[-buffer.window_frames :]
        np.testing.assert_array_equal(snapshot, expected)
        snapshot.fill(np.nan)
        np.testing.assert_array_equal(buffer.snapshot(), expected)
    assert len(buffer.blocks) == buffer.blocks.maxlen


def test_volume_gate_requires_at_least_a_tenth_of_the_blocks():
    threshold = 10 ** (VOLUME_THRESHOLD_DBFS / 20)
    waveform = torch.full((48_000,), threshold / 2)
    assert not is_audible(waveform)
    waveform[:320] = threshold * 2
    assert not is_audible(waveform)

    blocks = waveform.reshape(150, 320)
    blocks[::10] = threshold * 2  # Exactly 15 nonconsecutive active blocks.
    assert is_audible(waveform)
    blocks[0] = threshold / 2  # 14 active blocks.
    assert not is_audible(waveform)
    blocks[1:4:2] = threshold * 2  # 16 active blocks.
    assert is_audible(waveform)
    waveform[0] = float("nan")
    with pytest.raises(AssertionError, match="finite"):
        is_audible(waveform)


def test_predictor_resamples_smooths_and_resets_after_silence():
    model = CNN(conv_out_channels=(4, 8)).eval()
    predictor = LivePredictor(model)
    sample_rate_hz = 48_000
    time_sec = np.arange(3 * sample_rate_hz) / sample_rate_hz
    windows = [
        (0.1 * np.sin(2 * np.pi * frequency * time_sec)).astype(np.float32)
        for frequency in (220, 880)
    ]
    rng_state = torch.get_rng_state()
    with torch.inference_mode():
        expected = []
        for window in windows:
            waveform = preprocess_waveform(
                torch.from_numpy(window).unsqueeze(0), sample_rate_hz
            )
            features = model.preprocess_data(waveform.unsqueeze(0))
            expected.append(model(features).softmax(dim=1)[0])

    first = predictor.predict(windows[0], sample_rate_hz)
    torch.testing.assert_close(first, expected[0])
    second = predictor.predict(windows[1], sample_rate_hz)
    torch.testing.assert_close(
        second,
        SMOOTHING_ALPHA * expected[1] + (1 - SMOOTHING_ALPHA) * expected[0],
    )
    quiet = predictor.predict(np.zeros_like(windows[0]), sample_rate_hz)
    assert torch.count_nonzero(quiet) == 0
    assert predictor.smoothed is None
    restarted = predictor.predict(windows[1], sample_rate_hz)
    torch.testing.assert_close(restarted, expected[1])
    assert not restarted.requires_grad
    assert torch.equal(rng_state, torch.get_rng_state())
    assert all(not module.training for module in model.modules())
    model.train()
    with pytest.raises(AssertionError):
        predictor.predict(windows[0], sample_rate_hz)


def test_latest_checkpoint_uses_run_timestamp_and_never_skips_bad_file(
    tmp_path,
):
    with pytest.raises(RuntimeError, match="No checkpoint"):
        latest_checkpoint(tmp_path)
    older = tmp_path / "2026-09-30_120000_000000_cnn" / "checkpoint.pt"
    newer = tmp_path / "2026-10-01_120000_000000_cnn" / "checkpoint.pt"
    unfinished = tmp_path / "2026-10-02_120000_000000_cnn"
    for path in (newer, older):
        path.parent.mkdir()
        path.write_bytes(b"invalid checkpoint")
    unfinished.mkdir()
    assert latest_checkpoint(tmp_path) == newer


def test_display_preserves_label_order_and_shows_zero_for_quiet_window():
    display = format_display(torch.zeros(8), "Listening... but no speech")
    rows = display.splitlines()[2:10]
    assert [row.split()[0] for row in rows] == list(EMOTION_LABELS)
    assert all(row.endswith("0.00") and "█" not in row for row in rows)
    assert "Window: 3.0s" in display
