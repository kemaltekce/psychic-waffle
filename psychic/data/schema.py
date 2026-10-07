"""Common sample records produced by dataset-specific loaders."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from psychic.labels import EMOTION_TO_ID

HASH_LENGTH = 16


@dataclass(frozen=True)
class AudioSample:
    """Raw audio sample metadata shared across datasets.

    The sample points to an original audio file and contains only metadata
    known before deterministic preprocessing. Tensor paths, fixed waveform
    shape, and cache-specific fields belong to the preprocessed manifest.
    """

    sample_id: str
    dataset: str
    sample_path: Path
    emotion: str
    emotion_id: int
    speaker_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # `frozen=True` does not protect values inside a mutable dict, so copy
        # and wrap metadata to keep loader output stable after creation.
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


def build_sample_id(
    dataset: str,
    sample_path: str | Path,
    dataset_dir: str | Path,
) -> str:
    """Build a stable sample id from a dataset id and source-relative path."""
    assert dataset.strip(), "dataset must not be empty"
    assert dataset == dataset.lower(), "dataset must be lowercase"

    sample_path = Path(sample_path)
    dataset_dir = Path(dataset_dir)
    assert sample_path.is_relative_to(
        dataset_dir
    ), "sample_path must be inside dataset_dir"
    relative_sample_path = sample_path.relative_to(dataset_dir)

    # Use the sample path relative to the dataset dir so ids stay stable when
    # the project folder moves to another machine.
    identity = f"{dataset}:{relative_sample_path.as_posix()}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return f"{dataset}_{digest[:HASH_LENGTH]}"


def validate_audio_sample(sample: AudioSample) -> None:
    """Assert the raw audio sample contract.

    The sample file must exist, the emotion label and id must match the shared
    project taxonomy, and ids must be non-empty strings. Dataset-specific
    fields may live in `metadata`, but metadata keys must be strings so they
    can later be written into JSON manifests.
    """
    assert isinstance(sample, AudioSample), "sample must be an AudioSample"
    assert sample.sample_id.strip(), "sample_id must not be empty"
    assert sample.dataset.strip(), "dataset must not be empty"
    assert (
        sample.dataset == sample.dataset.lower()
    ), "dataset must be lowercase"
    assert isinstance(sample.sample_path, Path), "sample_path must be a Path"
    assert sample.sample_path.is_file(), "sample_path must be an existing file"
    assert sample.emotion in EMOTION_TO_ID, "emotion must be known"
    assert (
        sample.emotion_id == EMOTION_TO_ID[sample.emotion]
    ), "emotion_id must match emotion"
    assert sample.speaker_id.strip(), "speaker_id must not be empty"
    assert isinstance(sample.metadata, Mapping), "metadata must be a mapping"
    assert all(
        isinstance(key, str) and key.strip() for key in sample.metadata
    ), "metadata keys must be non-empty strings"
