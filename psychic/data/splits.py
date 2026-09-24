"""Speaker-disjoint split generation for dataset sample records."""

from collections.abc import Sequence

from psychic.data.schema import AudioSample

VAL_RATIO = 0.125
TEST_RATIO = 0.125


def build_speaker_disjoint_splits(
    samples: Sequence[AudioSample],
) -> dict[str, object]:
    """Build deterministic train/val/test splits with no speaker overlap.

    Speakers are sorted by id, then assigned contiguously to train,
    validation, and test sets. This preserves the old RAVDESS 1-18, 19-21,
    22-24 actor split when all 24 project speaker ids are present.
    """
    assert samples, "samples must not be empty"

    sample_ids = [sample.sample_id for sample in samples]
    assert len(sample_ids) == len(set(sample_ids)), "sample ids must be unique"

    speakers = sorted({sample.speaker_id for sample in samples})
    assert len(speakers) >= 3, (
        "at least three speakers are needed for train/val/test splits"
    )

    val_count = max(1, int(len(speakers) * VAL_RATIO + 0.5))
    test_count = max(1, int(len(speakers) * TEST_RATIO + 0.5))
    train_count = len(speakers) - val_count - test_count
    assert train_count > 0, (
        "split ratios must leave at least one train speaker"
    )

    train_speakers = speakers[:train_count]
    val_speakers = speakers[train_count : train_count + val_count]
    test_speakers = speakers[train_count + val_count :]

    train_speaker_set = set(train_speakers)
    val_speaker_set = set(val_speakers)
    test_speaker_set = set(test_speakers)
    assert train_speaker_set.isdisjoint(val_speaker_set), (
        "train and validation speakers must be disjoint"
    )
    assert train_speaker_set.isdisjoint(test_speaker_set), (
        "train and test speakers must be disjoint"
    )
    assert val_speaker_set.isdisjoint(test_speaker_set), (
        "validation and test speakers must be disjoint"
    )

    return {
        "split_id": "speaker_disjoint_v1",
        "strategy": "speaker_disjoint",
        "train_speakers": train_speakers,
        "val_speakers": val_speakers,
        "test_speakers": test_speakers,
        "train": _sample_ids_for_speakers(samples, train_speakers),
        "val": _sample_ids_for_speakers(samples, val_speakers),
        "test": _sample_ids_for_speakers(samples, test_speakers),
    }


def _sample_ids_for_speakers(
    samples: Sequence[AudioSample],
    speakers: Sequence[str],
) -> list[str]:
    speaker_ids = list(speakers)
    assert len(speaker_ids) == len(set(speaker_ids)), (
        "split speaker ids must be unique"
    )
    speaker_set = set(speaker_ids)
    return [
        sample.sample_id
        for sample in samples
        if sample.speaker_id in speaker_set
    ]
