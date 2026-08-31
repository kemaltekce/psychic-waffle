"""Stable sample ids for dataset records."""

import hashlib
from pathlib import Path

HASH_LENGTH = 16


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
