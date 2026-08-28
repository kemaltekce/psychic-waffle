"""Shared emotion labels and ids."""

from __future__ import annotations

EMOTION_LABELS: tuple[str, ...] = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)

ID_TO_EMOTION: dict[int, str] = {
    emotion_id: label for emotion_id, label in enumerate(EMOTION_LABELS)
}
EMOTION_TO_ID: dict[str, int] = {
    label: emotion_id for emotion_id, label in ID_TO_EMOTION.items()
}

