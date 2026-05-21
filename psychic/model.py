import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from psychic.dataset import ID_EMOTION_MAPPER

logger = logging.getLogger("psychic")

MODELS_DIR = Path("models")
TO_PREDICT_DIR = Path("to_predict")
MODEL_FILE_PATTERN = "*.pt"
SUPPORTED_AUDIO_EXTENSIONS = [".wav"]
CURRENT_MODEL = "CNN"


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


def get_model() -> nn.Module:
    """
    Load currently defined model in CURRENT_MODEL.
    """
    model_class = globals()[CURRENT_MODEL]
    return model_class()


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
            "Capacity warning: this model is probably too large for the "
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


def transfrom_waveform_to_spectogram(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    duration_sec: float = 3.0,
    n_mels: int = 64,
    n_fft: int = 1024,
    win_length: int = 400,
    hop_length: int = 160,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert waveform to spectrogram

    Parameters
    ----------
    sample_rate:
        Sampling rate used for the input waveform.
    duration_sec:
        Target waveform duration before feature extraction.
    n_mels:
        Number of mel frequency bins in the output spectrogram.
    n_fft:
        FFT size used to compute each spectrogram frame.
    win_length:
        Window size, in samples, for each analysis frame.
    hop_length:
        Step size, in samples, between consecutive frames.
    eps:
        Small constant used to avoid division by zero during normalization.
    """
    # TODO add assert for every arg. eg no negative value
    # TODO add assert waveform not empty and size

    target_num_samples = int(sample_rate * duration_sec)

    # pad and trim symatrically
    # TODO log if padding or trimming happens. log how long waveform is and
    # what it is trimmed or padded to.
    if waveform.numel() < target_num_samples:
        pad_amount = target_num_samples - waveform.numel()
        left_pad = pad_amount // 2
        right_pad = pad_amount - left_pad
        waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
    else:
        trim_amount = waveform.numel() - target_num_samples
        left_trim = trim_amount // 2
        right_trim = left_trim + target_num_samples
        waveform = waveform[left_trim:right_trim]

    # Convert the waveform into a compact time-frequency representation.
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform.numpy(),
        sr=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    # Log scaling compresses large energy differences and is standard for
    # audio models.
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    spectrogram = torch.tensor(log_mel_spectrogram, dtype=torch.float32)
    # Standardize each spectrogram so training sees a consistent value
    # range.
    spectrogram = (spectrogram - spectrogram.mean()) / (
        spectrogram.std() + eps
    )

    # TODO add asserts regarding final spectogram

    return spectrogram


def save_model(model: nn.Module, models_dir: Path = MODELS_DIR) -> Path:
    """
    Save a model state dict into the models folder using a timestamp name.
    """
    logger.info("Saving model")

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{datetime.now():%Y-%m-%d-%H%M%S}.pt"
    torch.save(model.state_dict(), model_path)

    logger.debug("Saved model to %s", model_path)
    return model_path


def load_latest_model(models_dir: Path = MODELS_DIR) -> nn.Module:
    """
    Load the latest saved model from the models folder.
    """
    logger.info("Loading latest model")
    model_paths = sorted(models_dir.glob(MODEL_FILE_PATTERN))
    if not model_paths:
        raise FileNotFoundError(
            "No saved models found in the models folder. Train and save a "
            "model first."
        )

    model_path = model_paths[-1]

    model = get_model()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    logger.debug("Loaded latest model from %s", model_path)
    return model


def predict_folder(
    model: nn.Module,
    folder_path: Path = TO_PREDICT_DIR,
) -> dict[str, str]:
    """
    Predict emotions for all supported audio files in the prediction folder.
    """
    # TODO add asserts

    logger.info("Predicting data from folder")

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Prediction folder {folder_path} does not exist. Please create "
            "it and add audio files to score."
        )

    audio_files = sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise FileNotFoundError(
            f"No supported audio files found in {folder_path}. Add files with "
            f"extensions {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    predictions: dict[str, str] = {}

    model.eval()
    with torch.no_grad():
        for audio_file in audio_files:
            waveform_np, _ = librosa.load(
                audio_file,
                sr=16_000,
                mono=True,
            )
            waveform = torch.tensor(waveform_np, dtype=torch.float32)
            features = transfrom_waveform_to_spectogram(waveform)
            features = features.unsqueeze(0).unsqueeze(0)
            logits = model(features)
            prediction_id = int(
                torch.softmax(logits, dim=1).argmax(dim=1).item()
            )
            prediction_label = ID_EMOTION_MAPPER[prediction_id]
            predictions[audio_file.name] = prediction_label
            logger.debug(
                "Prediction for %s: %s",
                audio_file.name,
                prediction_label,
            )

    return predictions
