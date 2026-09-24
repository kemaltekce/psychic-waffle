import json
from pathlib import Path

import torch

from psychic.data.preprocessing import preprocess_cache_waveforms
from psychic.data.ravdess import load_ravdess_samples
from psychic.data.splits import build_speaker_disjoint_splits


def write_ravdess_wav(
    dataset_dir: Path,
    actor: int,
    write_wav_file,
    ravdess_emotion_id: int = 5,
) -> Path:
    actor_dir = dataset_dir / f"Actor_{actor:02d}"
    actor_dir.mkdir(parents=True, exist_ok=True)
    sample_path = actor_dir / (
        f"03-01-{ravdess_emotion_id:02d}-01-01-01-{actor:02d}.wav"
    )
    waveform = torch.linspace(-0.25, 0.25, steps=400).unsqueeze(0)
    write_wav_file(sample_path, waveform, 8000)
    return sample_path


def test_preprocess_cache_waveforms_writes_expected_artifacts(
    tmp_path,
    write_wav_file,
) -> None:
    dataset_dir = tmp_path / "ravdess"
    for actor in range(1, 4):
        write_ravdess_wav(
            dataset_dir,
            actor=actor,
            write_wav_file=write_wav_file,
        )

    samples = load_ravdess_samples(dataset_dir)
    splits = build_speaker_disjoint_splits(samples)
    cache_dir = tmp_path / "cache"

    preprocess_cache_waveforms(samples, splits, cache_dir=cache_dir)

    config_path = cache_dir / "preprocess_config.json"
    manifest_path = cache_dir / "manifest.jsonl"
    splits_path = cache_dir / "splits.json"
    assert config_path.is_file()
    assert manifest_path.is_file()
    assert splits_path.is_file()

    written_config = json.loads(config_path.read_text())
    assert written_config["cache_id"] == "waveform_16khz_3s_v1"
    assert written_config["shape"] == [48000]
    assert written_config["pad_crop"] == "center"

    manifest_records = [
        json.loads(line) for line in manifest_path.read_text().splitlines()
    ]
    assert len(manifest_records) == 3
    first_record = manifest_records[0]
    assert first_record["sample_id"] == samples[0].sample_id
    assert first_record["dataset"] == "ravdess"
    assert first_record["emotion"] == "angry"
    assert first_record["emotion_id"] == 4
    assert first_record["speaker_id"] == "ravdess_actor_01"
    assert first_record["tensor_path"].startswith("tensors/")
    assert first_record["duration_sec"] == 0.05
    assert first_record["metadata"]["actor"] == 1

    tensor_path = cache_dir / first_record["tensor_path"]
    tensor = torch.load(tensor_path, weights_only=True)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (48000,)

    splits = json.loads(splits_path.read_text())
    assert splits["split_id"] == "speaker_disjoint_v1"
    assert splits["strategy"] == "speaker_disjoint"
    assert splits["train_speakers"] == ["ravdess_actor_01"]
    assert splits["val_speakers"] == ["ravdess_actor_02"]
    assert splits["test_speakers"] == ["ravdess_actor_03"]
