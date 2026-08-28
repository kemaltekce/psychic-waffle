from psychic.cli import build_parser


def test_parser_accepts_flat_workflow_commands() -> None:
    parser = build_parser()

    for command in ("preprocess", "train", "eval", "predict-file"):
        args = parser.parse_args([command])

        assert args.command == command
