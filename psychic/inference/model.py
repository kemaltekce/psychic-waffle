"""Restore a saved model using its recorded architecture and preprocessing."""

from pathlib import Path

import torch

from psychic.data.preprocessing import build_preprocess_config
from psychic.training.model import MODELS


def load_model(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict]:
    """Load saved weights into an eval-mode model and return its run config.

    Load on CPU first for portability. Reject incompatible architecture,
    emotion ordering, waveform/feature contracts, or state dicts. These
    checkpoints support evaluation/inference, not optimizer-state resumption.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    model_config = config["model"]
    assert config["waveform_preprocessing"] == build_preprocess_config(), (
        "unsupported waveform preprocessing"
    )
    model = MODELS[model_config["name"]](**model_config["init_args"])
    assert model_config == model.build_model_config(), (
        "model config mismatch: architecture, labels, or preprocessing"
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert all(
        torch.isfinite(value).all().item()
        for value in model.state_dict().values()
    ), "checkpoint tensors must be finite"
    model.to(device)
    model.eval()
    return model, config
