from psychic.cli import build_parser


def test_parser_accepts_flat_workflow_commands() -> None:
    parser = build_parser()

    for command in ("preprocess", "train", "eval", "predict-file"):
        args = parser.parse_args([command])

        assert args.command == command


def test_preprocess_parser_stays_simple() -> None:
    parser = build_parser()

    args = parser.parse_args(["preprocess"])

    assert args.command == "preprocess"
