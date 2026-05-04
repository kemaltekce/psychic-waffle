# TODO maybe replace librosa with torchaudio if librosa is too slow
import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict, Any, Tuple, List, TypedDict
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Sample(TypedDict):
    path: str
    modality: int
    vocal_channel: int
    emotion: int
    intensity: int
    statement: int
    repetition: int
    actor: int
    duration_sec: float


class RavdessAudioDataset(Dataset):
    """
    Dataset for Ravdess Speech Emotion data. But only for audio-only and speech
    data. Emotions are mapped from 01-08 to 0-7 for easier use with models.

    ==================
    Dataset parameters
    ==================

    root_dir:
        Root folder of the RAVDESS dataset (actor subfolders expected).
        Default: "data/Ravdess_Audio_Speech_Actors_01-24"

    sample_rate:
        Target sampling rate for loaded audio (default 16000 Hz).

    transform:
        Optional preprocessing function applied to waveform after loading

    =================================
    RAVDESS Dataset Metadata Encoding
    =================================

    Each audio filename follows the format:

        XX-XX-XX-XX-XX-XX-XX.wav

    and encodes structured emotional speech metadata.

    FILENAME PARTS (in order):
    --------------------------
    0. modality        → recording modality
    1. vocal_channel   → speech vs song
    2. emotion         → emotional state
    3. intensity       → emotional intensity
    4. statement       → spoken sentence ID
    5. repetition      → repetition count
    6. actor           → speaker ID

    ENCODING MAPS:
    --------------

    MODALITY:
        01 → full-AV
        02 → video-only
        03 → audio-only

    VOCAL CHANNEL:
        01 → speech
        02 → song

    EMOTION:
        01 → neutral
        02 → calm
        03 → happy
        04 → sad
        05 → angry
        06 → fearful
        07 → disgust
        08 → surprised

    INTENSITY:
        01 → normal
        02 → strong
        (Note: neutral emotion has no strong intensity variant)

    STATEMENTS:
        01 → "Kids are talking by the door"
        02 → "Dogs are sitting by the door"

    REPETITION:
        01 → 1st repetition
        02 → 2nd repetition

    ACTOR:
        01–24 → speaker ID
        odd numbers → male actors
        even numbers → female actors
    """

    # loader only for speech and audio-only data
    IDENTIFIER = "03-01"
    EXTENSION = ".wav"

    def __init__(
        self,
        root_dir: str = "data/Ravdess_Audio_Speech_Actors_01-24",
        sample_rate: int = 16_000,
        transform: Optional[Callable] = None,
        samples: Optional[List[Sample]] = None,
    ) -> None:

        # TODO assert root dir should be string not path
        # TODO assert if file path doesn't exist
        # TODO assert if samples not none or not list
        # TODO assert if sample_rate doesn't make sense

        self.transform = transform
        self.sample_rate = sample_rate
        self.root_dir = root_dir

        if samples is None:
            self.samples = self._load_samples()
        else:
            self.samples = list(samples)

    def _load_samples(self) -> list[Sample]:
        samples = []

        for actor_dir in os.listdir(self.root_dir):
            actor_path = os.path.join(self.root_dir, actor_dir)
            for file in os.listdir(actor_path):
                if file.startswith(self.IDENTIFIER) and file.endswith(
                    self.EXTENSION
                ):
                    full_path = os.path.join(actor_path, file)
                    metadata = file.replace(self.EXTENSION, "").split("-")
                    sample = {
                        "path": full_path,
                        "modality": int(metadata[0]),
                        "vocal_channel": int(metadata[1]),
                        # map emotions to 0-7 for easier use with model
                        "emotion": int(metadata[2]) - 1,
                        "intensity": int(metadata[3]),
                        "statement": int(metadata[4]),
                        "repetition": int(metadata[5]),
                        "actor": int(metadata[6]),
                        "duration_sec": round(
                            librosa.get_duration(path=full_path), 2
                        ),
                    }
                    samples.append(sample)

        # TODO assert if samples is empty

        return samples

    def subset_actors(self, actor_ids: List[int]) -> "RavdessAudioDataset":
        """Return a dataset view containing only the requested actors."""

        # TODO assert if empty list of actor_ids

        actor_ids_set = set(actor_ids)
        samples = [
            sample
            for sample in self.samples
            if sample["actor"] in actor_ids_set
        ]

        # TODO assert if samples empty

        return RavdessAudioDataset(
            root_dir=self.root_dir,
            sample_rate=self.sample_rate,
            transform=self.transform,
            samples=samples,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Sample]:
        sample = self.samples[idx]

        # define type to get rid of warning
        waveform_np, _ = librosa.load(
            sample["path"], sr=self.sample_rate, mono=True
        )

        waveform = torch.tensor(waveform_np, dtype=torch.float32)

        if self.transform:
            waveform = self.transform(waveform)

        # add dimension to have channel information present for CNN forward
        # waveform shape (height, width) -> (channel, height, width)
        waveform = waveform.unsqueeze(0)
        return waveform, sample.copy()


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        layer1_output_dim = 10

        self.layer1 = nn.Linear(input_dim, layer1_output_dim)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(layer1_output_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


class CNN(nn.Module):
    """
    Convolutional Neural Network for classifying spectrograms into the
    8 RAVDESS emotions.
    """

    def __init__(
        self,
        conv1_out_channels: int = 16,
        conv2_out_channels: int = 32,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        output_dim: int = 8,
    ) -> None:
        super().__init__()
        # the image size is 64*301 and we have two pooling layers
        flattened_dim = conv2_out_channels * 16 * 75

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv1_out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=conv1_out_channels,
                out_channels=conv2_out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO add assert of shape of x doesn't match. 4d expected because of
        # batch and channel

        x = self.features(x)
        x = self.classifier(x)
        return x


def transform(
    sample_rate: int = 16_000,
    duration_sec: float = 3.0,
    n_mels: int = 64,
    n_fft: int = 1024,
    win_length: int = 400,
    hop_length: int = 160,
    eps: float = 1e-8,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a preprocessing function for audio data/wavefroms.

    The returned callable converts a 1D waveform tensor into a normalized
    log-mel spectrogram by enforcing a fixed duration, scaling the waveform,
    extracting mel features, and standardizing the result.

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
    target_num_samples = int(sample_rate * duration_sec)

    def apply(waveform: torch.Tensor) -> torch.Tensor:

        # Force a fixed-length input so every sample produces the same shape.
        if waveform.numel() < target_num_samples:
            pad_amount = target_num_samples - waveform.numel()
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            trim_amount = waveform.numel() - target_num_samples
            left_trim = trim_amount // 2
            right_trim = left_trim + target_num_samples
            waveform = waveform[left_trim:right_trim]

        # TODO normalizing wavefrom could remove important information.
        # Scale amplitudes into a stable range without changing silence.
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

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

        return spectrogram

    return apply


def plot_spectrogram(data: torch.Tensor, meta: Dict[str, Any]) -> None:
    """Plot a normalized log-mel spectrogram with metadata."""

    plt.figure(figsize=(10, 4))
    plt.imshow(data.numpy(), aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="Normalized log-mel value")
    plt.title(f"Emotion: {meta['emotion']}, Actor: {meta['actor']}")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.show()


def inspect_model(model: nn.Module):
    # number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

    # approximate memory size
    param_size_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    buffer_size_bytes = sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    print("Model size (MB):", (param_size_bytes + buffer_size_bytes) / 1024**2)


def run():
    # define seed for reproducebility
    torch.manual_seed(456)
    g = torch.Generator()
    g.manual_seed(456)

    # load data
    dataset = RavdessAudioDataset(transform=transform())
    train_dataset = dataset.subset_actors(list(range(1, 19)))
    val_dataset = dataset.subset_actors(list(range(19, 22)))
    # test_dataset = dataset.subset_actors(list(range(22, 25)))
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, generator=g
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, generator=g
    )
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=16, shuffle=False, generator=g
    # )

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)
    # data, meta = next(iter(dataloader))
    # plot_spectrogram(data[0][0], {k: v[0] for k, v in meta.items()})

    # modeling
    model = CNN()
    inspect_model(model)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 10

    # keep history of model training
    model_history = {
        "loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total = 0
        correct = 0

        # train model
        for data, meta in train_dataloader:
            labels = meta["emotion"]

            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_fn(prediction, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            # softmax unnecessary for code but here for debugging purposes
            probability = torch.softmax(prediction, dim=1)
            probability_pred = probability.argmax(dim=1)
            correct += (probability_pred == labels).sum().item()
            # corss entropy loss is mean value. multiply by size to calculate
            # unskewed average with total later
            total_loss += loss.item() * labels.size(0)

        train_acc = correct / total
        avg_train_loss = total_loss / total

        # validation check
        model.eval()
        with torch.no_grad():
            val_total = 0
            val_correct = 0
            val_total_loss = 0
            for val_data, val_meta in val_dataloader:
                val_labels = val_meta["emotion"]
                val_prediction = model(val_data)
                val_total += val_labels.size(0)
                val_probability = torch.softmax(val_prediction, dim=1)
                val_probability_pred = val_probability.argmax(dim=1)
                val_correct += (
                    (val_probability_pred == val_labels).sum().item()
                )
                val_loss = loss_fn(val_prediction, val_labels)
                # corss entropy loss is mean value. multiply by size to
                # calculate unskewed average with total later
                val_total_loss += val_loss.item() * val_labels.size(0)
            val_acc = val_correct / val_total
            avg_val_loss = val_total_loss / val_total

        # save model history
        model_history["loss"].append(avg_train_loss)
        model_history["train_acc"].append(train_acc)
        model_history["val_loss"].append(avg_val_loss)
        model_history["val_acc"].append(val_acc)

        # if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} / "
            f"Train Acc = {train_acc:.4f} / "
            f"Val Loss = {avg_val_loss:.4f} / "
            f"Val Acc = {val_acc:.4f} / "
        )

    __import__("ipdb").set_trace()
    # TODO keep track of best epoch run and save model weights and at the end pick best model
    # TODO stop training if val doesn't improve for x runs
    # TODO add final test
    # TODO additional metrics to acc. recall, precision, f1, confusion metrix or per class accuracy
    # TODO plot model history
    # TODO plot confusion metrix
