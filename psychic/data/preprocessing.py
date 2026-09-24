"""Deterministic waveform preprocessing for reusable audio caches."""

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import librosa
import torch
import torchaudio.functional as audio_functional

from psychic.data.schema import AudioSample, validate_audio_sample
from psychic.labels import EMOTION_LABELS

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE_HZ = 16_000
DEFAULT_DURATION_SEC = 3.0
DEFAULT_NUM_SAMPLES = int(DEFAULT_SAMPLE_RATE_HZ * DEFAULT_DURATION_SEC)
DEFAULT_PREPROCESSING_ID = "waveform_16khz_3s_v1"
DEFAULT_PREPROCESSED_ROOT = Path("data/preprocessed")
DEFAULT_WAVEFORM_CACHE_DIR = (
    DEFAULT_PREPROCESSED_ROOT / DEFAULT_PREPROCESSING_ID
)
PAD_CROP_HOW = "center"
PREPROCESS_CONFIG_FILENAME = "preprocess_config.json"
MANIFEST_FILENAME = "manifest.jsonl"
SPLITS_FILENAME = "splits.json"
TENSOR_DIRNAME = "tensors"


def build_preprocess_config() -> dict[str, object]:
    """Return the fixed waveform preprocessing contract for JSON output."""
    return {
        "cache_id": DEFAULT_PREPROCESSING_ID,
        "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
        "duration_sec": DEFAULT_DURATION_SEC,
        "channels": 1,
        "dtype": "float32",
        "shape": [DEFAULT_NUM_SAMPLES],
        "pad_crop": PAD_CROP_HOW,
        "labels": list(EMOTION_LABELS),
    }


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert a loaded waveform tensor shaped `[channels, frames]` to 1D.

    Returns a contiguous `float32` tensor shaped `[frames]`.
    """
    assert isinstance(waveform, torch.Tensor), "waveform must be a tensor"
    assert waveform.ndim == 2, "waveform must have shape [channels, frames]"
    assert waveform.shape[0] > 0, "waveform must have at least one channel"
    assert waveform.shape[1] > 0, "waveform must have at least one frame"

    waveform = waveform.to(dtype=torch.float32)
    if waveform.shape[0] == 1:
        mono = waveform[0]
    else:
        mono = waveform.mean(dim=0)

    assert mono.ndim == 1, "mono waveform must be 1D"
    assert mono.numel() == waveform.shape[1], (
        "mono waveform must preserve frame count"
    )
    return mono.contiguous()


def resample_waveform(
    waveform: torch.Tensor,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> torch.Tensor:
    """Resample a 1D waveform when the source rate differs from the target."""
    assert isinstance(waveform, torch.Tensor), "waveform must be a tensor"
    assert waveform.ndim == 1, "waveform must be 1D before resampling"
    assert waveform.numel() > 0, "waveform must not be empty"
    assert source_sample_rate_hz > 0, "source sample rate must be positive"
    assert target_sample_rate_hz > 0, "target sample rate must be positive"

    waveform = waveform.to(dtype=torch.float32)
    if source_sample_rate_hz == target_sample_rate_hz:
        return waveform.contiguous()

    resampled = audio_functional.resample(
        waveform,
        orig_freq=source_sample_rate_hz,
        new_freq=target_sample_rate_hz,
    )
    assert resampled.ndim == 1, "resampled waveform must stay 1D"
    assert resampled.numel() > 0, "resampled waveform must not be empty"
    return resampled.to(dtype=torch.float32).contiguous()


def pad_or_crop(
    waveform: torch.Tensor,
    target_num_samples: int,
    how: str = PAD_CROP_HOW,
) -> torch.Tensor:
    """Deterministically pad or crop a 1D waveform."""
    assert how == "center", (
        "pad/crop methods like 'left' and 'right' must be implemented first"
    )
    assert isinstance(waveform, torch.Tensor), "waveform must be a tensor"
    assert waveform.ndim == 1, "waveform must be 1D"
    assert waveform.numel() > 0, "waveform must not be empty"
    assert target_num_samples > 0, "target_num_samples must be positive"

    waveform = waveform.to(dtype=torch.float32)
    sample_count = waveform.numel()
    if sample_count == target_num_samples:
        return waveform.contiguous()

    if sample_count < target_num_samples:
        pad_count = target_num_samples - sample_count
        left_pad = pad_count // 2
        right_pad = pad_count - left_pad
        padded = torch.nn.functional.pad(waveform, (left_pad, right_pad))
        assert padded.numel() == target_num_samples, (
            "padded waveform must match target length"
        )
        return padded.to(dtype=torch.float32).contiguous()

    crop_count = sample_count - target_num_samples
    left_crop = crop_count // 2
    right_crop = left_crop + target_num_samples
    cropped = waveform[left_crop:right_crop]
    assert cropped.numel() == target_num_samples, (
        "cropped waveform must match target length"
    )
    return cropped.to(dtype=torch.float32).contiguous()


def validate_preprocessed_waveform(
    waveform: torch.Tensor,
    target_num_samples: int = DEFAULT_NUM_SAMPLES,
) -> None:
    """Assert the fixed waveform tensor contract before saving or training."""
    assert isinstance(waveform, torch.Tensor), "waveform must be a tensor"
    assert waveform.dtype == torch.float32, "waveform must be float32"
    assert tuple(waveform.shape) == (target_num_samples,), (
        "waveform shape must match preprocessing config"
    )
    assert waveform.ndim == 1, "waveform cache tensors must be 1D"
    assert torch.isfinite(waveform).all().item(), (
        "waveform values must be finite"
    )


def preprocess_waveform(
    waveform: torch.Tensor,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    target_num_samples: int = DEFAULT_NUM_SAMPLES,
) -> torch.Tensor:
    """Apply deterministic mono, resample, and center pad/crop processing."""
    mono = convert_to_mono(waveform)
    resampled = resample_waveform(
        mono,
        source_sample_rate_hz=source_sample_rate_hz,
        target_sample_rate_hz=target_sample_rate_hz,
    )
    fixed = pad_or_crop(resampled, target_num_samples)
    validate_preprocessed_waveform(fixed, target_num_samples)
    return fixed


def preprocess_audio_file(
    sample_path: str | Path,
    target_sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    target_num_samples: int = DEFAULT_NUM_SAMPLES,
) -> tuple[torch.Tensor, float]:
    """Load an audio file and return its deterministic fixed waveform."""
    sample_path = Path(sample_path)
    assert sample_path.is_file(), "sample_path must be an existing file"

    audio, source_sample_rate_hz = librosa.load(
        sample_path,
        sr=None,
        mono=False,
    )
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    assert source_sample_rate_hz > 0, "source sample rate must be positive"
    assert waveform.ndim == 2, "loaded waveform must be [channels, frames]"
    assert waveform.shape[1] > 0, "loaded waveform must not be empty"

    source_duration_sec = waveform.shape[1] / source_sample_rate_hz
    fixed = preprocess_waveform(
        waveform,
        source_sample_rate_hz=source_sample_rate_hz,
        target_sample_rate_hz=target_sample_rate_hz,
        target_num_samples=target_num_samples,
    )
    return fixed, source_duration_sec


def preprocess_cache_waveforms(
    samples: Sequence[AudioSample],
    splits: dict[str, object],
    cache_dir: str | Path = DEFAULT_WAVEFORM_CACHE_DIR,
) -> None:
    """Rebuild the deterministic waveform cache for validated samples."""
    cache_dir = Path(cache_dir)
    validate_samples_and_splits(samples, splits)

    logger.info("Preprocessing and writing waveform cache to %s", cache_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)
    tensor_dir = cache_dir / TENSOR_DIRNAME
    tensor_dir.mkdir(parents=True)

    write_preprocess_config(cache_dir)

    manifest_records = []
    for sample in samples:
        waveform, duration_sec = preprocess_audio_file(sample.sample_path)
        validate_preprocessed_waveform(waveform)

        tensor_path = tensor_dir / f"{sample.sample_id}.pt"
        torch.save(waveform, tensor_path)
        manifest_records.append(
            build_manifest_record(
                sample=sample,
                tensor_path=tensor_path,
                cache_dir=cache_dir,
                duration_sec=duration_sec,
            )
        )

    write_manifest(cache_dir, manifest_records)
    write_splits(cache_dir, splits)

    logger.info(
        "Wrote %s waveform tensors: train=%s val=%s test=%s",
        len(samples),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )


def validate_samples_and_splits(
    samples: Sequence[AudioSample],
    splits: dict[str, object],
) -> None:
    """Assert cache inputs before deleting generated files."""
    assert samples, "samples must not be empty"
    for sample in samples:
        validate_audio_sample(sample)

    sample_ids = [sample.sample_id for sample in samples]
    assert len(sample_ids) == len(set(sample_ids)), "sample ids must be unique"

    split_sample_ids = []
    for split_name in ("train", "val", "test"):
        split_ids = splits.get(split_name)
        assert isinstance(split_ids, list), (
            f"{split_name} split must be a list"
        )
        assert all(isinstance(sample_id, str) for sample_id in split_ids), (
            f"{split_name} split ids must be strings"
        )
        assert len(split_ids) == len(set(split_ids)), (
            f"{split_name} split ids must be unique"
        )
        split_sample_ids.extend(split_ids)

    assert set(split_sample_ids) == set(sample_ids), (
        "split sample ids must match loaded sample ids"
    )
    assert len(split_sample_ids) == len(sample_ids), (
        "split sample ids must not overlap"
    )


def write_preprocess_config(cache_dir: Path) -> None:
    """Write the preprocessing contract."""
    (cache_dir / PREPROCESS_CONFIG_FILENAME).write_text(
        json.dumps(build_preprocess_config(), indent=2) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    cache_dir: Path,
    manifest_records: Sequence[dict[str, object]],
) -> None:
    """Write one compact JSON record per preprocessed sample."""
    (cache_dir / MANIFEST_FILENAME).write_text(
        "\n".join(
            json.dumps(record, separators=(",", ":"))
            for record in manifest_records
        )
        + "\n",
        encoding="utf-8",
    )


def write_splits(cache_dir: Path, splits: dict[str, object]) -> None:
    """Write speaker-disjoint split assignments."""
    (cache_dir / SPLITS_FILENAME).write_text(
        json.dumps(splits, indent=2) + "\n",
        encoding="utf-8",
    )


def build_manifest_record(
    sample: AudioSample,
    tensor_path: Path,
    cache_dir: Path,
    duration_sec: float,
) -> dict[str, object]:
    """Build one manifest row for a saved waveform tensor."""
    assert tensor_path.is_relative_to(cache_dir), (
        "tensor_path must be inside cache_dir"
    )
    assert duration_sec > 0, "duration_sec must be positive"

    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "sample_path": sample.sample_path.as_posix(),
        "tensor_path": tensor_path.relative_to(cache_dir).as_posix(),
        "emotion": sample.emotion,
        "emotion_id": sample.emotion_id,
        "speaker_id": sample.speaker_id,
        "duration_sec": round(duration_sec, 6),
        "metadata": dict(sample.metadata),
    }
