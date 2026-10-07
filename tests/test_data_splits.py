from pathlib import Path

from psychic.data.schema import AudioSample
from psychic.data.splits import build_speaker_disjoint_splits


def make_sample(tmp_path: Path, sample_index: int) -> AudioSample:
    sample_path = tmp_path / f"sample-{sample_index}.wav"
    sample_path.write_bytes(b"")

    return AudioSample(
        sample_id=f"ravdess_{sample_index:016x}",
        dataset="ravdess",
        sample_path=sample_path,
        emotion="angry",
        emotion_id=4,
        speaker_id=f"ravdess_actor_{sample_index:02d}",
        metadata={"actor": sample_index},
    )


def test_speaker_disjoint_splits_have_no_speaker_overlap(tmp_path) -> None:
    samples = [
        make_sample(tmp_path, sample_index)
        for sample_index in range(1, 5)
    ]

    splits = build_speaker_disjoint_splits(samples)

    train_speakers = set(splits["train_speakers"])
    val_speakers = set(splits["val_speakers"])
    test_speakers = set(splits["test_speakers"])
    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)
    assert splits["train"] == [
        "ravdess_0000000000000001",
        "ravdess_0000000000000002",
    ]
    assert splits["val"] == ["ravdess_0000000000000003"]
    assert splits["test"] == ["ravdess_0000000000000004"]
