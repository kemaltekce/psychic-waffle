import logging
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("psychic")


class CNN(nn.Module):
    """
    Convolutional Neural Network for classifying spectrograms into the
    8 RAVDESS emotions.
    """

    def __init__(
        self,
        conv1_out_channels: int = 16,
        conv2_out_channels: int = 32,
        conv3_out_channels: int = 64,
        conv4_out_channels: int = 128,
        avg_pool_dim: Tuple[int, int] = (4, 4),
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        output_dim: int = 8,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(
                in_channels=1,
                out_channels=conv1_out_channels,
                kernel_size=4,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(conv1_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer 2
            nn.Conv2d(
                in_channels=conv1_out_channels,
                out_channels=conv2_out_channels,
                kernel_size=4,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(conv2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer 3
            nn.Conv2d(
                in_channels=conv2_out_channels,
                out_channels=conv3_out_channels,
                kernel_size=4,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(conv3_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer 4
            nn.Conv2d(
                in_channels=conv3_out_channels,
                out_channels=conv4_out_channels,
                kernel_size=4,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(conv4_out_channels),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p),
            # adaptive pooling to reduce highly dense connection to neural
            # network. main logic should be learnt in conv layers not in the
            # dense layers
            nn.AdaptiveAvgPool2d(avg_pool_dim),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                conv4_out_channels * avg_pool_dim[0] * avg_pool_dim[1],
                hidden_dim1,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(
                hidden_dim1,
                hidden_dim2,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO add assert of shape of x doesn't match. 4d expected because of
        # batch and channel

        x = self.features(x)
        x = self.classifier(x)
        return x


def inspect_model(model: nn.Module):
    """
    Print a short summary of a model's parameter counts and approximate
    memory footprint.

    Args:
        model: PyTorch module to inspect.
    """
    logger.info("Inspecting model")
    # TODO add assert
    # number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # approximate memory size
    param_size_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    buffer_size_bytes = sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    model_size = (param_size_bytes + buffer_size_bytes) / 1024**2
    logger.debug(
        "Model specs - "
        f"Total params= {total_params:_} \\ "
        f"Trainable params: {trainable_params:_} \\ "
        f"Model size (MB): {model_size:.2f}"
    )


def model_capacity_check(model: nn.Module, train_dataset_size: int) -> None:
    """
    Print a rough sanity check for model size relative to the train split.

    Args:
        model: Neural network whose parameter count should be inspected.
        train_dataset_size: Number of samples in the training split used
            to estimate model capacity relative to available data.
    """
    logger.info("Checking model capacity")
    # TODO assert if model exists
    total_params = sum(p.numel() for p in model.parameters())
    params_per_train_sample = total_params / train_dataset_size
    if params_per_train_sample > 1_000:
        logger.warning(
            "Capacity warning: this CNN is probably large for the "
            "RAVDESS train split and may overfit."
        )
    else:
        logger.debug("Capacity check: model size looks reasonable.")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Evaluate a model on a dataloader and return average loss and accuracy.

    Args:
        model: Neural network to evaluate.
        dataloader: Batches of `(data, meta)` samples to score.
        loss_fn: Loss function used to compute batch loss from predictions
            and labels.

    Returns:
        A `(loss, accuracy, labels, predictions)` tuple with dataset-level
        average loss, classification accuracy, and all labels/predictions.
    """
    # TODO add asserts
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        for data, meta in dataloader:
            labels = meta["emotion"]
            prediction = model(data)
            total += labels.size(0)
            probability = torch.softmax(prediction, dim=1)
            probability_pred = probability.argmax(dim=1)
            all_labels.append(labels.detach())
            all_predictions.append(probability_pred.detach())
            correct += (probability_pred == labels).sum().item()
            loss = loss_fn(prediction, labels)
            # corss entropy loss is mean value. multiply by size to
            # calculate unskewed average with total later
            total_loss += loss.item() * labels.size(0)
        acc = correct / total
        loss = total_loss / total
    model.train()
    # TODO add assert e.g. model in train mode
    return (
        loss,
        acc,
        torch.cat(all_labels),
        torch.cat(all_predictions),
    )


def calculate_f1_score(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Calculate macro F1 score for multiclass classification.

    Args:
        labels: Ground-truth class ids for each evaluated sample.
        predictions: Predicted class ids for each evaluated sample.
        num_classes: Number of classes included in the task.

    Returns:
        Macro-averaged F1 score across all classes.
    """
    # TODO add asserts
    f1_scores = []
    for class_idx in range(num_classes):
        true_positive = (
            ((predictions == class_idx) & (labels == class_idx)).sum().item()
        )
        false_positive = (
            ((predictions == class_idx) & (labels != class_idx)).sum().item()
        )
        false_negative = (
            ((predictions != class_idx) & (labels == class_idx)).sum().item()
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )
        f1_denominator = precision + recall
        f1_scores.append(
            2 * precision * recall / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

    # TODO add asserts

    return sum(f1_scores) / num_classes


def calculate_confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
    label_mapper: Dict[int, str],
    log_matrix: bool = True,
) -> torch.Tensor:
    """
    Build a confusion matrix with rows=true labels and cols=predictions.

    Args:
        labels: Ground-truth class ids for each evaluated sample.
        predictions: Predicted class ids for each evaluated sample.
        num_classes: Number of classes used to size the square matrix.
        label_mapper: Mapping from class id to human-readable emotion
            label used for row names.
        log_matrix: print matrix if true
    """
    logger.info("Creating confusion matrix")
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for label, prediction in zip(labels, predictions):
        matrix[label.long(), prediction.long()] += 1

    if log_matrix:
        labels_matrix = [
            f"{label_mapper[idx]} ({idx})" for idx in range(matrix.size(1))
        ]
        header = " " * 19 + " ".join(
            f"{f'({idx})':>4}" for idx in range(matrix.size(1))
        )
        lines = ["Confusion matrix with rows=true and cols=pred:", header]
        for idx, row in enumerate(matrix):
            values = " ".join(f"{value.item():>4}" for value in row)
            lines.append(f"{labels_matrix[idx]:>18} {values}")
        logger.debug("\n".join(lines))
    return matrix
