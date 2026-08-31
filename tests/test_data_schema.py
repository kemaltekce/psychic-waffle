from dataclasses import FrozenInstanceError

import pytest

from psychic.data.schema import AudioSample, validate_audio_sample


def make_sample(tmp_path) -> AudioSample:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    return AudioSample(
        sample_id="ravdess_7f3a91c8d4e2b105",
        dataset="ravdess",
        sample_path=audio_path,
        emotion="angry",
        emotion_id=4,
        speaker_id="ravdess_actor_01",
        metadata={"actor": 1, "statement": 1},
    )


def test_valid_audio_sample_passes_validation(tmp_path) -> None:
    sample = make_sample(tmp_path)

    validate_audio_sample(sample)


def test_audio_sample_is_immutable(tmp_path) -> None:
    sample = make_sample(tmp_path)

    with pytest.raises(FrozenInstanceError):
        sample.emotion = "sad"

    with pytest.raises(TypeError):
        sample.metadata["actor"] = 2


def test_audio_sample_metadata_is_copied(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")
    metadata = {"actor": 1}

    sample = AudioSample(
        sample_id="ravdess_7f3a91c8d4e2b105",
        dataset="ravdess",
        sample_path=audio_path,
        emotion="angry",
        emotion_id=4,
        speaker_id="ravdess_actor_01",
        metadata=metadata,
    )
    metadata["actor"] = 2

    assert (
        sample.metadata["actor"] == 1
    ), "metadata must not change when the original dict changes"


def test_validation_rejects_emotion_id_mismatch(tmp_path) -> None:
    sample = make_sample(tmp_path)
    invalid_sample = AudioSample(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        sample_path=sample.sample_path,
        emotion=sample.emotion,
        emotion_id=3,
        speaker_id=sample.speaker_id,
        metadata=sample.metadata,
    )

    with pytest.raises(AssertionError, match="emotion_id must match emotion"):
        validate_audio_sample(invalid_sample)
