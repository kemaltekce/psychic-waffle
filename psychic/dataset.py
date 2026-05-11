import librosa  # TODO maybe use torchaudio for speed
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict, Any, Tuple, List, TypedDict
import os
from collections import Counter
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("psychic")


ID_EMOTION_MAPPER = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}


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


def transform(
    sample_rate: int = 16_000,
    duration_sec: float = 3.0,
    n_mels: int = 64,
    n_fft: int = 1024,
    win_length: int = 400,
    hop_length: int = 160,
    eps: float = 1e-8,
    max_shift_ms: float = 50.0,
    noise_std_range: Tuple[float, float] = (0.0, 0.002),
    gain_range: Tuple[float, float] = (1.0, 1.0),
    num_time_masks: int = 0,
    max_time_mask_pct: float = 0.01,
    num_freq_masks: int = 0,
    max_freq_mask_bins: int = 1,
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
    max_shift_ms:
        Maximum random temporal shift, in milliseconds, to wavefrom
    noise_std_range:
        Range to add Gaussian noise standard deviation to waveform
    gain_range:
        Inclusive range to sample waveform amplitude gain.
    num_time_masks:
        Number of time masks to apply to the spectrogram.
    max_time_mask_pct:
        Maximum width of each time mask as a fraction of total time frames.
    num_freq_masks:
        Number of frequency masks to apply to the spectrogram.
    max_freq_mask_bins:
        Maximum width of each frequency mask, measured in mel bins.
    """
    # TODO add assert for every arg. eg no negative value

    target_num_samples = int(sample_rate * duration_sec)
    max_shift_samples = int(sample_rate * max_shift_ms / 1000.0)

    def apply(waveform: torch.Tensor) -> torch.Tensor:

        # TODO add assert waveform not empty and size

        # augment waveform data. time shift, noise and gain jitter
        # Randomly time shift speech
        shift = int(
            torch.randint(
                -max_shift_samples,
                max_shift_samples + 1,
                (1,),
            ).item()
        )
        if shift > 0:
            waveform = torch.nn.functional.pad(waveform[:-shift], (shift, 0))
        elif shift < 0:
            waveform = torch.nn.functional.pad(waveform[-shift:], (0, -shift))

        # add noise
        noise_low, noise_high = noise_std_range
        noise_std = torch.empty(1).uniform_(noise_low, noise_high).item()
        waveform = waveform + torch.randn_like(waveform) * noise_std

        # add gain gitter
        gain_low, gain_high = gain_range
        gain = torch.empty(1).uniform_(gain_low, gain_high).item()
        waveform = waveform * gain

        # randomly trimming and padding probably best for real life situation
        # randomly trim and pad data.
        if waveform.numel() < target_num_samples:
            pad_amount = target_num_samples - waveform.numel()
            left_pad = int(torch.randint(0, pad_amount + 1, (1,)).item())
            right_pad = pad_amount - left_pad
            waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
        else:
            trim_amount = waveform.numel() - target_num_samples
            left_trim = torch.randint(0, trim_amount + 1, (1,)).item()
            right_trim = left_trim + target_num_samples
            waveform = waveform[left_trim:right_trim]
        # pad at end and trim symatrically
        # if waveform.numel() < target_num_samples:
        #     pad_amount = target_num_samples - waveform.numel()
        #     waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        # else:
        #     trim_amount = waveform.numel() - target_num_samples
        #     left_trim = trim_amount // 2
        #     right_trim = left_trim + target_num_samples
        #     waveform = waveform[left_trim:right_trim]
        #
        # # TODO normalizing wavefrom could remove important information.
        # # Scale amplitudes into a stable range without changing silence.
        # peak = waveform.abs().max()
        # if peak > 0:
        #     waveform = waveform / peak

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

        # augment spectogram. time and frequency mask
        # time mask
        # TODO assert if spectrogram too small. max time mask frames
        # should be greater than 1 and smaller than spectrogram
        max_time_mask_frames = int(spectrogram.size(1) * max_time_mask_pct)
        for _ in range(num_time_masks):
            mask_width = torch.randint(
                0, max_time_mask_frames + 1, (1,)
            ).item()
            if mask_width == 0:
                continue
            start = torch.randint(
                0, int(spectrogram.size(1) - mask_width + 1), (1,)
            ).item()
            spectrogram[:, start : start + mask_width] = 0.0

        # freq mask
        # TODO assert max_freq_mask_bins should be smaller than
        # spectogram.size(0)
        for _ in range(num_freq_masks):
            mask_height = torch.randint(0, max_freq_mask_bins + 1, (1,)).item()
            if mask_height == 0:
                continue
            start = torch.randint(
                0, int(spectrogram.size(0) - mask_height + 1), (1,)
            ).item()
            spectrogram[start : start + mask_height, :] = 0.0

        # plot_spectrogram(spectrogram, {"emotion": 0, "actor": 0})

        # TODO add asserts regarding final spectogram

        return spectrogram

    return apply


def plot_spectrogram(data: torch.Tensor, meta: Dict[str, Any]) -> None:
    """
    Plot a normalized log-mel spectrogram and annotate it with sample
    metadata.

    Args:
        data: Two-dimensional spectrogram tensor to display.
        meta: Metadata dictionary for the sample, expected to include at
            least the `emotion` and `actor` fields for the plot title.
    """
    # TODO add assert. tensor size and meta keys

    plt.figure(figsize=(10, 4))
    plt.imshow(data.numpy(), aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="Normalized log-mel value")
    plt.title(f"Emotion: {meta['emotion']}, Actor: {meta['actor']}")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.show()


def dataset_summary(dataset: RavdessAudioDataset) -> None:
    """
    Print dataset size and class balance for a RAVDESS split.

    Args:
        dataset: Dataset split whose sample count and class frequencies
            should be printed.
    """
    # TODO add assert check size of dataset
    class_counts = Counter(sample["emotion"] for sample in dataset.samples)
    logger.info("Summarizing dataset")
    logger.debug(f"Dataset size: {len(dataset)}")
    class_balance_str = "Class balance:"
    for emotion_idx in range(8):
        count = class_counts.get(emotion_idx, 0)
        ratio = count / len(dataset)
        class_balance_str += (
            f"\n    Class {emotion_idx}: {count} samples ({ratio:.2%})"
        )
    logger.debug(class_balance_str)


def build_class_weights(dataset: RavdessAudioDataset) -> torch.Tensor:
    """
    Create inverse-frequency class weights from a dataset split.

    Args:
        dataset: Dataset split used for training.
    """
    logger.info("Calculating class weights")
    counts = Counter(sample["emotion"] for sample in dataset.samples)
    num_classes = len(counts.keys())
    total_samples = len(dataset)
    weights = []
    for class_idx in range(num_classes):
        class_count = counts.get(class_idx, 0)
        if class_count == 0:
            weights.append(0.0)
        else:
            weights.append(total_samples / (num_classes * class_count))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    logger.debug(
        f"Using the following class weights: {class_weights.tolist()}"
    )
    return class_weights
