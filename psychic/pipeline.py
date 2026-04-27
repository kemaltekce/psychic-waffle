import librosa
import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class RavdessAudioDataset(Dataset):
    """
    ==================
    Dataset parameters
    ==================

    root_dir:
        Root folder of the RAVDESS dataset (actor subfolders expected).
        Default: "data/Ravdess_Audio_Speech_Actors_01-24"

    identifier:
        File prefix used to filter audio only speech samples (e.g. "03-01").

    extension:
        Audio file extension to include (default ".wav").

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

    def __init__(
        self,
        root_dir: str | Path = "data/Ravdess_Audio_Speech_Actors_01-24",
        identifier="03-01",
        extension=".wav",
        sample_rate: int = 16_000,
        transform: Optional[Callable] = None,
    ) -> None:

        # TODO assert if file path doesn't exist

        self.samples = []
        self.transform = transform
        self.sample_rate = sample_rate

        for actor_dir in os.listdir(root_dir):
            actor_path = os.path.join(root_dir, actor_dir)
            for file in os.listdir(actor_path):
                if file.startswith(identifier) & file.endswith(extension):
                    full_path = os.path.join(actor_path, file)
                    metadata = file.replace(extension, "").split("-")
                    sample = {
                        "path": full_path,
                        "modality": int(metadata[0]),
                        "vocal_channel": int(metadata[1]),
                        "emotion": int(metadata[2]),
                        "intensity": int(metadata[3]),
                        "statement": int(metadata[4]),
                        "repetition": int(metadata[5]),
                        "actor": int(metadata[6]),
                        "duration_sec": round(
                            librosa.get_duration(path=full_path), 2
                        ),
                    }
                    self.samples.append(sample)

        # TODO assert if samples is empty

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]

        # define type to get rid of warning
        waveform_np: np.ndarray
        waveform_np, _ = librosa.load(
            sample["path"], sr=self.sample_rate, mono=True
        )

        waveform: torch.Tensor = torch.tensor(waveform_np, dtype=torch.float32)

        if self.transform:
            waveform = self.transform(waveform)

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


def run():
    torch.manual_seed(456)
    X = torch.randn(10, 1)
    y = X @ torch.tensor([[2.0]]) + torch.tensor(1.0) + torch.rand(10, 1) * 0.1

    model = NeuralNetwork(1, 1)
    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    epochs = 100

    for epoch in range(1, epochs + 1):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            print(
                "target, pred: \n", torch.stack([y[:, 0], y_hat[:, 0]], dim=1)
            )
