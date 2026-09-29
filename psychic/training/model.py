"""The small four-block CNN used for the first rebuilt baseline."""

import torch
from torch import nn

from psychic.labels import EMOTION_LABELS
from psychic.training.preprocessing import (
    build_log_mel_config,
    waveforms_to_log_mel,
)

CURRENT_MODEL = "cnn"


class CNN(nn.Module):
    """Map float32 `[batch, 1, mel, time]` inputs to `[batch, output_dim]`.

    All convolutions use the same square kernel, stride 1, and padding 1.
    On MPS, feature dimensions must be divisible by the pooling dimensions.
    """

    def __init__(
        self,
        conv_out_channels: tuple[int, ...] = (16, 32, 64, 128),
        dropout_p: float = 0.1,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        output_dim: int = 8,
        avg_pool_dim: tuple[int, int] = (3, 5),
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        assert conv_out_channels, "at least one convolution block is required"
        self.init_args = {
            "conv_out_channels": list(conv_out_channels),
            "dropout_p": dropout_p,
            "hidden_dim1": hidden_dim1,
            "hidden_dim2": hidden_dim2,
            "output_dim": output_dim,
            "avg_pool_dim": list(avg_pool_dim),
            "kernel_size": kernel_size,
        }
        layers = []
        in_channels = 1
        for index, out_channels in enumerate(conv_out_channels):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
            if index < len(conv_out_channels) - 1:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(
            *layers,
            nn.Dropout2d(dropout_p),
            nn.AdaptiveAvgPool2d(tuple(avg_pool_dim)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                conv_out_channels[-1] * avg_pool_dim[0] * avg_pool_dim[1],
                hidden_dim1,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim2, output_dim),
        )

    def preprocess_data(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Convert CPU `[batch, 48000]` waveforms to log-mel CNN inputs."""
        return waveforms_to_log_mel(waveforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits; callers choose train/eval mode explicitly."""
        assert x.ndim == 4 and x.shape[1] == 1, (
            "expected [batch, 1, mel, time]"
        )
        x = self.features(x)
        return self.classifier(x)

    def build_model_config(self) -> dict:
        """Record architecture, labels, and preprocessing for reloading."""
        assert self.init_args["output_dim"] == len(EMOTION_LABELS), (
            "saved emotion models must output one logit per label"
        )
        return {
            "name": "cnn",
            "version": 1,
            "input": "log_mel_spectrogram",
            "num_classes": self.init_args["output_dim"],
            "labels": list(EMOTION_LABELS),
            "init_args": dict(self.init_args),
            "preprocessing": build_log_mel_config(),
        }


# Both training and loading select architectures here.
MODELS: dict[str, type[nn.Module]] = {"cnn": CNN}
