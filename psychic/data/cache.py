"""Read waveform caches without depending on the original audio files."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from psychic.data.preprocessing import (
    build_preprocess_config,
    validate_preprocessed_waveform,
)
from psychic.labels import EMOTION_TO_ID


class CachedWaveforms(Dataset):
    """Load validated CPU float32 `[48000]` waveforms and integer labels."""

    def __init__(self, samples: list[tuple[Path, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        tensor_path, emotion_id = self.samples[index]
        waveform = torch.load(
            tensor_path, map_location="cpu", weights_only=True
        )
        validate_preprocessed_waveform(waveform)
        return waveform, emotion_id


def load_waveform_cache(
    cache_dir: str | Path,
) -> tuple[dict[str, CachedWaveforms], dict, dict]:
    """Read v1 metadata and return datasets, waveform config, and splits.

    Reject incompatible caches, invalid labels/paths, incomplete partitions,
    and speaker leakage before training. Tensor contents are checked lazily
    when read; original audio paths are provenance only.
    """
    cache_dir = Path(cache_dir).resolve()
    config = json.loads(
        (cache_dir / "preprocess_config.json").read_text(encoding="utf-8")
    )
    assert config == build_preprocess_config(), (
        "unsupported waveform cache contract; run psy preprocess"
    )
    records = [
        json.loads(line)
        for line in (cache_dir / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records, "cache manifest must not be empty"
    samples = {}
    tensor_paths = set()
    for record in records:
        sample_id = record["sample_id"]
        assert isinstance(sample_id, str) and sample_id.strip()
        assert sample_id not in samples, "manifest sample ids must be unique"
        emotion_id = record["emotion_id"]
        assert type(emotion_id) is int
        assert record["emotion"] in EMOTION_TO_ID, "unknown emotion"
        assert EMOTION_TO_ID[record["emotion"]] == emotion_id, (
            "emotion_id must match emotion"
        )
        speaker_id = record["speaker_id"]
        assert isinstance(speaker_id, str) and speaker_id.strip()
        relative_path = Path(record["tensor_path"])
        tensor_path = (cache_dir / relative_path).resolve()
        assert not relative_path.is_absolute()
        assert tensor_path.is_relative_to(cache_dir), (
            "tensor_path must stay inside cache_dir"
        )
        assert tensor_path.is_file(), f"missing cached tensor: {tensor_path}"
        assert tensor_path not in tensor_paths, "tensor paths must be unique"
        tensor_paths.add(tensor_path)
        samples[sample_id] = (tensor_path, emotion_id, speaker_id)

    splits = json.loads(
        (cache_dir / "splits.json").read_text(encoding="utf-8")
    )
    assert isinstance(splits["split_id"], str) and splits["split_id"].strip()
    assert splits["strategy"] == "speaker_disjoint"
    seen_ids, seen_speakers = set(), set()
    datasets = {}
    for name in ("train", "val", "test"):
        sample_ids = splits[name]
        assert isinstance(sample_ids, list) and sample_ids, (
            f"{name} split must be a nonempty list"
        )
        assert all(isinstance(sample_id, str) for sample_id in sample_ids)
        assert len(sample_ids) == len(set(sample_ids)), "duplicate split ids"
        assert set(sample_ids) <= samples.keys(), "unknown split sample id"
        assert seen_ids.isdisjoint(sample_ids), "split sample ids overlap"
        speakers = {samples[sample_id][2] for sample_id in sample_ids}
        declared_speakers = splits[f"{name}_speakers"]
        assert isinstance(declared_speakers, list)
        assert len(declared_speakers) == len(set(declared_speakers))
        assert set(declared_speakers) == speakers, "speaker metadata mismatch"
        assert seen_speakers.isdisjoint(speakers), "split speakers overlap"
        seen_ids.update(sample_ids)
        seen_speakers.update(speakers)
        datasets[name] = CachedWaveforms(
            [
                (samples[sample_id][0], samples[sample_id][1])
                for sample_id in sample_ids
            ]
        )
    assert seen_ids == samples.keys(), "splits must cover the entire manifest"
    return datasets, config, splits
