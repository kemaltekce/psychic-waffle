"""Whole-split classification metrics in the canonical label order."""

import torch

from psychic.labels import EMOTION_LABELS


def classification_metrics(
    labels: torch.Tensor, predictions: torch.Tensor
) -> dict[str, float]:
    """Return accuracy and macro F1 for nonempty CPU int64 class-id vectors.

    All eight emotions contribute equally to macro F1. Classes with no true
    or predicted samples receive F1 zero, including on tiny development sets.
    """
    assert labels.dtype == predictions.dtype == torch.int64
    assert labels.ndim == 1 and labels.numel() > 0
    assert predictions.shape == labels.shape
    num_classes = len(EMOTION_LABELS)
    assert ((labels >= 0) & (labels < num_classes)).all().item()
    assert ((predictions >= 0) & (predictions < num_classes)).all().item()
    matrix = torch.bincount(
        labels * num_classes + predictions, minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    true_positives = matrix.diag()
    denominator = matrix.sum(dim=0) + matrix.sum(dim=1)
    f1 = 2 * true_positives.double() / denominator.clamp_min(1)
    return {
        "accuracy": true_positives.sum().item() / labels.numel(),
        "macro_f1": f1.mean().item(),
    }
