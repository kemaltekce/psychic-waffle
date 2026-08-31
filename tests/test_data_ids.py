from pathlib import Path

import pytest

from psychic.data.ids import HASH_LENGTH, build_sample_id


def test_build_sample_id_is_stable_across_root_locations() -> None:
    first_id = build_sample_id(
        "ravdess",
        Path("/tmp/first/Actor_01/sample.wav"),
        Path("/tmp/first"),
    )
    second_id = build_sample_id(
        "ravdess",
        Path("/tmp/second/Actor_01/sample.wav"),
        Path("/tmp/second"),
    )

    assert first_id == second_id, "ids should use paths relative to root"
    assert first_id.startswith("ravdess_"), "id should include dataset prefix"
    assert len(first_id) == len("ravdess_") + HASH_LENGTH


def test_build_sample_id_requires_sample_path_inside_dataset_dir() -> None:
    with pytest.raises(AssertionError, match="inside dataset_dir"):
        build_sample_id(
            "ravdess",
            Path("/tmp/other/sample.wav"),
            Path("/tmp/root"),
        )
