from psychic.labels import EMOTION_LABELS, EMOTION_TO_ID, ID_TO_EMOTION


def test_emotion_mappings_are_complete_and_stable() -> None:
    assert EMOTION_LABELS == (
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    )
    assert ID_TO_EMOTION == {
        0: "neutral",
        1: "calm",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fearful",
        6: "disgust",
        7: "surprised",
    }
    assert EMOTION_TO_ID == {
        label: emotion_id for emotion_id, label in ID_TO_EMOTION.items()
    }
