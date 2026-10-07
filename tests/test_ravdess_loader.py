from pathlib import Path

import pytest

from psychic.data.ravdess import (
    DATASET_ID,
    build_ravdess_sample,
    load_ravdess_samples,
    parse_ravdess_filename,
)


def write_ravdess_file(
    dataset_dir: Path,
    actor: int = 1,
    ravdess_emotion_id: int = 5,
    vocal_channel: int = 1,
) -> Path:
    actor_dir = dataset_dir / f"Actor_{actor:02d}"
    actor_dir.mkdir(parents=True, exist_ok=True)
    sample_path = actor_dir / (
        "03-"
        f"{vocal_channel:02d}-"
        f"{ravdess_emotion_id:02d}-"
        f"01-01-01-{actor:02d}.wav"
    )
    sample_path.write_bytes(b"fake wav data")
    return sample_path


def test_parse_ravdess_filename_extracts_metadata() -> None:
    metadata = parse_ravdess_filename(
        Path("03-01-05-02-01-02-13.wav")
    )

    assert metadata == {
        "modality": 3,
        "vocal_channel": 1,
        "ravdess_emotion_id": 5,
        "intensity": 2,
        "statement": 1,
        "repetition": 2,
        "actor": 13,
    }


def test_load_ravdess_samples_discovers_speech_audio_files(tmp_path) -> None:
    valid_path = write_ravdess_file(tmp_path, actor=1, ravdess_emotion_id=5)
    write_ravdess_file(
        tmp_path,
        actor=1,
        ravdess_emotion_id=5,
        vocal_channel=2,
    )

    samples = load_ravdess_samples(tmp_path)

    assert len(samples) == 1, "loader should ignore non-speech audio files"
    sample = samples[0]
    assert sample.dataset == DATASET_ID
    assert sample.sample_path == valid_path
    assert sample.emotion == "angry"
    assert sample.emotion_id == 4
    assert sample.speaker_id == "ravdess_actor_01"
    assert sample.sample_id.startswith("ravdess_")
    assert sample.metadata["ravdess_emotion_id"] == 5
    assert sample.metadata["actor"] == 1


def test_load_ravdess_samples_is_deterministic(tmp_path) -> None:
    second_path = write_ravdess_file(tmp_path, actor=2, ravdess_emotion_id=4)
    first_path = write_ravdess_file(tmp_path, actor=1, ravdess_emotion_id=5)

    samples = load_ravdess_samples(tmp_path)

    assert [sample.sample_path for sample in samples] == [
        first_path,
        second_path,
    ], "loader should return samples in deterministic path order"


def test_load_ravdess_samples_requires_existing_root(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="RAVDESS dataset not found"):
        load_ravdess_samples(tmp_path / "missing")


def test_load_ravdess_samples_requires_matching_actor_folder(
    tmp_path,
) -> None:
    actor_dir = tmp_path / "Actor_02"
    actor_dir.mkdir()
    sample_path = actor_dir / "03-01-05-01-01-01-01.wav"
    sample_path.write_bytes(b"fake wav data")

    with pytest.raises(AssertionError, match="actor folder must match"):
        build_ravdess_sample(sample_path, tmp_path)


def test_parse_ravdess_filename_rejects_bad_field_count() -> None:
    with pytest.raises(AssertionError, match="seven metadata fields"):
        parse_ravdess_filename(Path("03-01-05.wav"))
