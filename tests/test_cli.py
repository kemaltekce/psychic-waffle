import pytest

from psychic.cli import build_parser


def test_parser_accepts_flat_workflow_commands() -> None:
    parser = build_parser()

    for command in ("preprocess", "train", "predict-file"):
        args = parser.parse_args([command])

        assert args.command == command


def test_parser_rejects_separate_eval_command() -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["eval"])

    assert error.value.code == 2


def test_preprocess_parser_stays_simple() -> None:
    parser = build_parser()

    args = parser.parse_args(["preprocess"])

    assert args.command == "preprocess"


def test_train_parser_stays_simple() -> None:
    args = build_parser().parse_args(["train"])

    assert vars(args) == {"command": "train"}
