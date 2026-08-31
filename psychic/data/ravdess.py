"""RAVDESS dataset discovery and metadata parsing.

RAVDESS speech audio files encode metadata in file names:

    modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav

FILENAME PARTS (in order):
--------------------------
    0. modality      -> recording modality
    1. vocal_channel -> speech vs song
    2. emotion       -> emotional state
    3. intensity     -> emotional intensity
    4. statement     -> spoken sentence ID
    5. repetition    -> repetition count
    6. actor         -> speaker ID

ENCODING MAPS:
--------------
    MODALITY:
        01 -> full-AV
        02 -> video-only
        03 -> audio-only

    VOCAL CHANNEL:
        01 -> speech
        02 -> song

    EMOTION:
        01 -> neutral
        02 -> calm
        03 -> happy
        04 -> sad
        05 -> angry
        06 -> fearful
        07 -> disgust
        08 -> surprised

    INTENSITY:
        01 -> normal
        02 -> strong
        Neutral emotion has no strong intensity variant.

    STATEMENTS:
        01 -> "Kids are talking by the door"
        02 -> "Dogs are sitting by the door"

    REPETITION:
        01 -> 1st repetition
        02 -> 2nd repetition

    ACTOR:
        01-24 -> speaker ID
        odd numbers -> male actors
        even numbers -> female actors

This project only loads audio-only speech files, which start with `03-01-`.
The RAVDESS emotion id is 1-based, while the project model emotion id is
0-based. For example, RAVDESS emotion id 5 maps to project label `angry` and
project emotion id 4.
"""

import logging
from pathlib import Path

from psychic.data.ids import build_sample_id
from psychic.data.schema import AudioSample, validate_audio_sample
from psychic.labels import EMOTION_LABELS, EMOTION_TO_ID

logger = logging.getLogger(__name__)

DATASET_ID = "ravdess"
DEFAULT_RAVDESS_ROOT = Path("data/original/ravdess")
RAVDESS_AUDIO_SPEECH_PREFIX = "03-01-"
RAVDESS_AUDIO_EXTENSION = ".wav"
RAVDESS_METADATA_FIELDS = (
    "modality",
    "vocal_channel",
    "ravdess_emotion_id",
    "intensity",
    "statement",
    "repetition",
    "actor",
)
RAVDESS_EMOTION_ID_TO_LABEL = {
    ravdess_emotion_id: label
    for ravdess_emotion_id, label in enumerate(EMOTION_LABELS, 1)
}


def parse_ravdess_filename(sample_path: str | Path) -> dict[str, int]:
    """Parse RAVDESS metadata encoded in an audio file name."""
    sample_path = Path(sample_path)
    assert (
        sample_path.suffix.lower() == RAVDESS_AUDIO_EXTENSION
    ), "RAVDESS files must be wav files"

    parts = sample_path.stem.split("-")
    assert len(parts) == len(
        RAVDESS_METADATA_FIELDS
    ), "RAVDESS file names must contain seven metadata fields"

    values = [int(part) for part in parts]

    metadata = dict(zip(RAVDESS_METADATA_FIELDS, values, strict=True))
    assert metadata["modality"] == 3, "RAVDESS sample must be audio-only"
    assert metadata["vocal_channel"] == 1, "RAVDESS sample must be speech"
    assert (
        metadata["ravdess_emotion_id"] in RAVDESS_EMOTION_ID_TO_LABEL
    ), "RAVDESS emotion id must be known"
    assert metadata["actor"] > 0, "RAVDESS actor id must be positive"

    return metadata


def load_ravdess_samples(
    dataset_dir: str | Path = DEFAULT_RAVDESS_ROOT,
) -> list[AudioSample]:
    """Discover RAVDESS speech audio files and return common sample records."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"RAVDESS dataset not found: {dataset_dir}")

    logger.info("Loading RAVDESS samples from %s", dataset_dir)
    samples = []
    audio_paths = sorted(
        dataset_dir.rglob(f"*{RAVDESS_AUDIO_EXTENSION}")
    )
    for sample_path in audio_paths:
        if not sample_path.name.startswith(RAVDESS_AUDIO_SPEECH_PREFIX):
            continue

        sample = build_ravdess_sample(sample_path, dataset_dir)
        validate_audio_sample(sample)
        samples.append(sample)

    if not samples:
        raise FileNotFoundError(
            f"No RAVDESS speech wav files found in {dataset_dir}"
        )

    logger.info("Loaded %s RAVDESS samples", len(samples))
    return samples


def build_ravdess_sample(
    sample_path: str | Path,
    dataset_dir: str | Path,
) -> AudioSample:
    """Build a common sample record from one RAVDESS file."""
    sample_path = Path(sample_path)
    dataset_dir = Path(dataset_dir)
    metadata = parse_ravdess_filename(sample_path)
    actor = metadata["actor"]
    expected_actor_dir = f"Actor_{actor:02d}"
    assert (
        sample_path.parent.name == expected_actor_dir
    ), "RAVDESS actor folder must match filename actor"

    ravdess_emotion_id = metadata["ravdess_emotion_id"]
    emotion = RAVDESS_EMOTION_ID_TO_LABEL[ravdess_emotion_id]

    return AudioSample(
        sample_id=build_sample_id(DATASET_ID, sample_path, dataset_dir),
        dataset=DATASET_ID,
        sample_path=sample_path,
        emotion=emotion,
        emotion_id=EMOTION_TO_ID[emotion],
        speaker_id=f"{DATASET_ID}_actor_{actor:02d}",
        metadata={
            "modality": metadata["modality"],
            "vocal_channel": metadata["vocal_channel"],
            "ravdess_emotion_id": ravdess_emotion_id,
            "intensity": metadata["intensity"],
            "statement": metadata["statement"],
            "repetition": metadata["repetition"],
            "actor": actor,
        },
    )
