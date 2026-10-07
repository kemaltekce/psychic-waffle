"""CPU emotion inference and a small live microphone terminal display."""

import sys
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import torch

from psychic.data.preprocessing import (
    DEFAULT_DURATION_SEC,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SAMPLE_RATE_HZ,
    preprocess_waveform,
    validate_preprocessed_waveform,
)
from psychic.inference.audio import BLOCK_DURATION_SEC, AudioBuffer
from psychic.inference.model import load_model
from psychic.labels import EMOTION_LABELS

UPDATE_INTERVAL_SEC = 0.25
VOLUME_THRESHOLD_DBFS = -40.0
SMOOTHING_ALPHA = 0.3
CAPTURE_TIMEOUT_SEC = 2.0
BAR_WIDTH = 20


def latest_checkpoint(models_dir: Path = Path("models")) -> Path:
    """Select the newest timestamped training run containing a checkpoint.

    Run names sort chronologically. Do not fall back to an older checkpoint
    if the selected file later fails compatibility checks.
    """
    checkpoints = [
        path for path in models_dir.glob("*/checkpoint.pt") if path.is_file()
    ]
    if not checkpoints:
        raise RuntimeError(
            f"No checkpoint found in {models_dir}. Run uv run psy train first."
        )
    return max(checkpoints, key=lambda path: path.parent.name)


def is_audible(waveform: torch.Tensor) -> bool:
    """Require at least a tenth of 20 ms blocks above the RMS threshold.

    Input is a finite float32 mono tensor of three seconds at 16 kHz. At least
    15 of its 150 blocks must qualify; they need not be consecutive. dBFS is
    relative to amplitude 1.0. This measures volume, not speech presence.
    """
    validate_preprocessed_waveform(waveform)
    block_frames = round(DEFAULT_SAMPLE_RATE_HZ * BLOCK_DURATION_SEC)
    assert DEFAULT_NUM_SAMPLES % block_frames == 0
    rms = waveform.reshape(-1, block_frames).square().mean(dim=1).sqrt()
    # ponytail: noise also opens this gate; use VAD if that becomes a problem.
    active_blocks = (rms > 10 ** (VOLUME_THRESHOLD_DBFS / 20)).sum().item()
    return active_blocks >= rms.numel() / 10


class LivePredictor:
    """Reuse an eval-mode CPU model and retain only the previous EMA scores."""

    def __init__(self, model: torch.nn.Module) -> None:
        assert all(not module.training for module in model.modules())
        assert all(p.device.type == "cpu" for p in model.parameters())
        self.model = model
        self.smoothed: torch.Tensor | None = None

    @torch.inference_mode()
    def predict(self, audio: np.ndarray, sample_rate_hz: int) -> torch.Tensor:
        """Resample one full mono window and return eight smoothed CPU scores.

        Windows below the gate skip the CNN, clear EMA, and return exact zeros.
        Active scores sum to one. Augmentation and gradients stay disabled.
        """
        assert sample_rate_hz > 0
        assert audio.dtype == np.float32
        assert audio.shape == (round(sample_rate_hz * DEFAULT_DURATION_SEC),)
        assert all(not module.training for module in self.model.modules())
        waveform = preprocess_waveform(
            torch.from_numpy(audio).unsqueeze(0), sample_rate_hz
        )
        if not is_audible(waveform):
            self.smoothed = None
            return torch.zeros(len(EMOTION_LABELS))

        features = self.model.preprocess_data(
            waveform.unsqueeze(0), augment=False
        )
        logits = self.model(features)
        assert logits.shape == (1, len(EMOTION_LABELS))
        assert logits.dtype == torch.float32 and logits.device.type == "cpu"
        assert torch.isfinite(logits).all().item()
        probabilities = logits.softmax(dim=1)[0]
        if self.smoothed is None:
            self.smoothed = probabilities
        else:
            self.smoothed = (
                SMOOTHING_ALPHA * probabilities
                + (1 - SMOOTHING_ALPHA) * self.smoothed
            )
        assert torch.isclose(self.smoothed.sum(), torch.tensor(1.0)).item()
        return self.smoothed


def format_display(probabilities: torch.Tensor, status: str) -> str:
    """Format eight fixed-order bars; zeros represent inactive prediction."""
    assert probabilities.shape == (len(EMOTION_LABELS),)
    assert torch.isfinite(probabilities).all().item()
    assert ((probabilities >= 0) & (probabilities <= 1)).all().item()
    lines = [status, ""]
    for label, probability in zip(
        EMOTION_LABELS, probabilities.tolist(), strict=True
    ):
        filled = round(probability * BAR_WIDTH)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        lines.append(f"{label:<9} {bar}  {probability:.2f}")
    lines.extend(
        ["", f"Window: {DEFAULT_DURATION_SEC:.1f}s", "Ctrl+C to stop"]
    )
    return "\n".join(lines) + "\n"


def run_inference() -> None:
    """Load once, capture the default microphone, and redraw until Ctrl+C.

    Requires an interactive terminal and a compatible newest checkpoint.
    Capture errors stop the command; stream and terminal state are restored
    on interruption or failure. Audio stays in memory and is never saved.
    """
    if not sys.stdout.isatty():
        raise RuntimeError("Live inference requires an interactive terminal.")

    # Initialize PortAudio only for microphone inference.
    import sounddevice as sd

    checkpoint = latest_checkpoint()
    try:
        model, _ = load_model(checkpoint, device="cpu")
    except Exception as error:
        raise RuntimeError(
            f"Cannot load newest checkpoint {checkpoint}: {error}"
        ) from error
    predictor = LivePredictor(model)
    print(f"Checkpoint: {checkpoint}", flush=True)

    try:
        device = sd.query_devices(kind="input")
        sample_rate_hz = round(device["default_samplerate"])
        buffer = AudioBuffer(sample_rate_hz)

        def callback(indata, frames, time_info, status):
            """Copy audio and pass capture failures to the main loop."""
            if status:
                buffer.error = str(status)
                raise sd.CallbackAbort
            buffer.append(indata[:, 0])

        with sd.InputStream(
            samplerate=sample_rate_hz,
            channels=1,
            dtype="float32",
            blocksize=buffer.block_frames,
            callback=callback,
        ) as stream:
            _display_loop(stream, buffer, predictor)
    except sd.PortAudioError as error:
        raise RuntimeError(
            f"Cannot capture the default microphone: {error}. "
            "Check the system input device and microphone permission."
        ) from error


def _display_loop(stream, buffer, predictor) -> None:
    """Predict the newest window at most four times per second; never queue."""
    sample_rate_hz = round(stream.samplerate)
    zeros = torch.zeros(len(EMOTION_LABELS))
    try:
        sys.stdout.write("\033[?1049h\033[?25l")
        while True:
            started_sec = monotonic()
            if buffer.error:
                raise RuntimeError(
                    f"Microphone capture failed: {buffer.error}"
                )
            if not stream.active:
                raise RuntimeError("Microphone stream stopped unexpectedly.")
            if started_sec - buffer.last_received_sec > CAPTURE_TIMEOUT_SEC:
                raise RuntimeError("Microphone stopped delivering audio.")

            audio = buffer.snapshot()
            probabilities = zeros
            status = "Listening... filling 3-second buffer"
            if audio is not None:
                probabilities = predictor.predict(audio, sample_rate_hz)
                status = (
                    "Listening... but no speech"
                    if predictor.smoothed is None
                    else "Listening... analysing..."
                )
            sys.stdout.write(
                "\033[H\033[J" + format_display(probabilities, status)
            )
            sys.stdout.flush()
            sleep(max(0, UPDATE_INTERVAL_SEC - (monotonic() - started_sec)))
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
