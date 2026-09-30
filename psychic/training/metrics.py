"""Whole-split classification metrics in the canonical label order."""

import torch

from psychic.labels import EMOTION_LABELS


def classification_metrics(
    labels: torch.Tensor, predictions: torch.Tensor
) -> dict:
    """Return scalar metrics, confusion counts, and per-emotion scores.

    Inputs are nonempty CPU int64 class-id vectors. Matrix rows are true
    labels and columns are predictions, both in canonical emotion order.
    Support is the number of true samples, without class weighting.
    All eight emotions contribute equally to macro F1. Classes with no true
    or predicted samples receive F1 zero. Undefined precision/recall are zero.
    """
    assert labels.device.type == predictions.device.type == "cpu"
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
    predicted_counts = matrix.sum(dim=0)
    support = matrix.sum(dim=1)
    precision = true_positives.double() / predicted_counts.clamp_min(1)
    recall = true_positives.double() / support.clamp_min(1)
    denominator = predicted_counts + support
    f1 = 2 * true_positives.double() / denominator.clamp_min(1)
    assert matrix.sum().item() == labels.numel()
    return {
        "accuracy": true_positives.sum().item() / labels.numel(),
        "macro_f1": f1.mean().item(),
        "labels": list(EMOTION_LABELS),
        "confusion_matrix": matrix.tolist(),
        "per_emotion": {
            emotion: {
                "precision": precision[index].item(),
                "recall": recall[index].item(),
                "f1": f1[index].item(),
                "support": support[index].item(),
            }
            for index, emotion in enumerate(EMOTION_LABELS)
        },
    }


def format_classification_report(metrics: dict) -> str:
    """Format classification_metrics output as labeled counts and scores."""
    labels = metrics["labels"]
    lines = [
        "Confusion matrix (rows=true, columns=predicted; counts):",
        f"{'':>10}" + "".join(f"{label:>10}" for label in labels),
    ]
    for label, row in zip(labels, metrics["confusion_matrix"], strict=True):
        lines.append(f"{label:>10}" + "".join(f"{count:>10}" for count in row))
    lines.extend(
        [
            "",
            "Precision: when it predicts this emotion, how often is it right?",
            "Recall: how many actual examples of this emotion does it find?",
            "",
            f"{'Emotion':>10} {'Precision':>10} {'Recall':>10} "
            f"{'F1':>10} {'Support':>10}",
        ]
    )
    for label in labels:
        scores = metrics["per_emotion"][label]
        lines.append(
            f"{label:>10} {scores['precision']:>10.4f} "
            f"{scores['recall']:>10.4f} {scores['f1']:>10.4f} "
            f"{scores['support']:>10}"
        )
    return "\n".join(lines)
