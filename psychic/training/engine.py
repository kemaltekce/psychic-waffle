"""Train on cached waveforms and select by validation macro F1."""

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from psychic.data.cache import load_waveform_cache
from psychic.data.preprocessing import DEFAULT_WAVEFORM_CACHE_DIR
from psychic.labels import EMOTION_LABELS
from psychic.training.metrics import classification_metrics
from psychic.training.model import CURRENT_MODEL, MODELS

logger = logging.getLogger(__name__)


def select_device(name: str = "auto") -> torch.device:
    """Select MPS, CUDA, then CPU, or require the explicitly named device."""
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if name == "auto":
        name = next(name for name, supported in available.items() if supported)
    if name not in available or not available[name]:
        raise ValueError(f"device is not available: {name}")
    return torch.device(name)


def build_class_weights(labels: list[int]) -> torch.Tensor:
    """Balance observed training classes; return CPU float32 label weights.

    Use N / (observed classes * class count). Absent training classes keep
    weight 1 so tiny cache slices do not hide validation errors for them.
    """
    counts = torch.bincount(
        torch.tensor(labels), minlength=len(EMOTION_LABELS)
    )
    present = counts > 0
    weights = torch.ones(len(EMOTION_LABELS), dtype=torch.float32)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    """Enter train mode, optimize one pass, and report online batch metrics."""
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    all_labels, all_predictions = [], []
    for waveforms, labels in loader:
        features = model.preprocess_data(waveforms).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        targets = labels.to(device)
        loss = nn.functional.cross_entropy(
            logits, targets, weight=class_weights
        )
        assert torch.isfinite(loss).item(), "training loss must be finite"
        loss.backward()
        optimizer.step()
        batch_weight = (
            labels.numel()
            if class_weights is None
            else class_weights[targets].sum().item()
        )
        total_loss += loss.item() * batch_weight
        total_weight += batch_weight
        all_labels.append(labels)
        all_predictions.append(logits.detach().argmax(dim=1).cpu())
    return _epoch_metrics(
        total_loss, total_weight, all_labels, all_predictions
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    """Enter and leave the model in eval mode; score without changing weights.

    Loss uses the sum of target weights, including the final partial batch
    (or sample count when unweighted). Accuracy
    and macro F1 are computed over the entire split, not averaged per batch.
    """
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    all_labels, all_predictions = [], []
    for waveforms, labels in loader:
        features = model.preprocess_data(waveforms).to(device)
        logits = model(features)
        targets = labels.to(device)
        loss = nn.functional.cross_entropy(
            logits, targets, weight=class_weights
        )
        assert torch.isfinite(loss).item(), "evaluation loss must be finite"
        batch_weight = (
            labels.numel()
            if class_weights is None
            else class_weights[targets].sum().item()
        )
        total_loss += loss.item() * batch_weight
        total_weight += batch_weight
        all_labels.append(labels)
        all_predictions.append(logits.argmax(dim=1).cpu())
    return _epoch_metrics(
        total_loss, total_weight, all_labels, all_predictions
    )


def _epoch_metrics(
    total_loss: float,
    total_weight: float,
    labels: list[torch.Tensor],
    predictions: list[torch.Tensor],
) -> dict[str, float]:
    """Aggregate a nonempty epoch using target weights for loss averaging."""
    labels = torch.cat(labels)
    metrics = classification_metrics(labels, torch.cat(predictions))
    metrics["loss"] = total_loss / total_weight
    return metrics


def train(
    cache_dir: str | Path = DEFAULT_WAVEFORM_CACHE_DIR,
    models_dir: str | Path = "models",
    *,
    model_name: str = CURRENT_MODEL,
    model_kwargs: dict | None = None,
    epochs: int = 100,
    early_stopping_patience: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seed: int = 456,
    device: str = "auto",
) -> Path:
    """Train/validate from v1 cache splits and return a new model run folder.

    Save the first epoch with each strictly higher validation macro F1;
    ties retain the earlier checkpoint. Stop after early_stopping_patience
    epochs without improvement. Class weights come only from training labels
    and are used for training and validation loss. Test tensors are never
    loaded.
    Seed initialization, dropout, and train shuffling; accelerator results
    are not guaranteed bit-for-bit reproducible across devices or versions.
    Choose a class from MODELS with model_name and pass its constructor args
    via model_kwargs. It must define preprocess_data and build_model_config;
    preprocessing accepts CPU waveforms and forward returns emotion logits.
    """
    assert epochs > 0, "epochs must be positive"
    assert early_stopping_patience > 0, (
        "early_stopping_patience must be positive"
    )
    assert math.isfinite(learning_rate) and learning_rate > 0, (
        "learning_rate must be finite and positive"
    )
    selected_device = select_device(device)
    datasets, waveform_config, splits = load_waveform_cache(cache_dir)
    class_weights = build_class_weights(
        [label for _, label in datasets["train"].samples]
    ).to(selected_device)
    torch.manual_seed(seed)
    model = MODELS[model_name](**(model_kwargs or {})).to(selected_device)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(datasets["val"], batch_size=batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    config = {
        "seed": seed,
        "device": str(selected_device),
        "torch_version": str(torch.__version__),
        "data": {
            "cache_dir": str(Path(cache_dir).resolve()),
            "cache_id": waveform_config["cache_id"],
            "split_id": splits["split_id"],
            "splits": splits,
        },
        "waveform_preprocessing": waveform_config,
        "model": model.build_model_config(),
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "adamw",
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "loss": "cross_entropy",
            "class_weights": class_weights.cpu().tolist(),
            "early_stopping_patience": early_stopping_patience,
        },
    }
    run_dir = Path(models_dir) / (
        f"{datetime.now():%Y-%m-%d_%H%M%S_%f}_{model_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", config)
    logger.info(
        "Training %s on %s: train=%s val=%s; run=%s",
        model_name,
        selected_device,
        len(datasets["train"]),
        len(datasets["val"]),
        run_dir,
    )
    history = []
    best_f1 = -1.0
    metrics = {}
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, selected_device, class_weights
        )
        val_metrics = evaluate(
            model, val_loader, selected_device, class_weights
        )
        epoch_metrics = {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append({"epoch": epoch, **epoch_metrics})
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            checkpoint = {
                "config": config,
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            }
            # Preserve the previous best until the new write completes.
            temporary_path = run_dir / "checkpoint.tmp"
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(run_dir / "checkpoint.pt")
            metrics = {
                "best_epoch": epoch,
                "best_metric": "val_macro_f1",
                **epoch_metrics,
            }
        _write_json(run_dir / "metrics.json", {**metrics, "history": history})
        logger.info(
            "Epoch %s/%s | train loss=%.4f acc=%.4f F1=%.4f | "
            "val loss=%.4f acc=%.4f F1=%.4f | best=%s",
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            metrics["best_epoch"],
        )
        if epoch - metrics["best_epoch"] >= early_stopping_patience:
            logger.info(
                "Early stopping: validation macro F1 has not improved for %s "
                "epochs; best epoch=%s",
                early_stopping_patience,
                metrics["best_epoch"],
            )
            break
    logger.info(
        "Saved best checkpoint: %s (epoch=%s, val macro F1=%.4f)",
        run_dir / "checkpoint.pt",
        metrics["best_epoch"],
        best_f1,
    )
    return run_dir


def _write_json(path: Path, value: dict) -> None:
    """Replace a JSON sidecar only after serializing finite metrics/config."""
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
