import argparse
import coloredlogs
import logging

from psychic import pipeline
from psychic.model import load_latest_model, predict_folder, save_model

logger = logging.getLogger("psychic")
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s  %(message)s"


def define_logging():
    coloredlogs.install(
        level=logging.DEBUG,
        fmt=LOG_FORMAT,
        field_styles={
            "asctime": {"color": "green"},
            "name": {"color": "blue"},
            "levelname": {"color": "black", "bright": True},
        },
    )
    # Disable some third-party noise
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logger.info("Starting speech emotion recognition pipeline")


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        prog="psy",
        description="Speech emotion recognition",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model with the training pipeline.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the latest saved model from the models folder.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the trained model into the models folder.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predict all supported audio files from the to_predict folder.",
    )
    return parser, parser.parse_args()


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not any(vars(args).values()):
        parser.error(
            "No arguments provided. Please use --train and/or --load "
            "together with optional --save or --predict."
        )

    if args.train and args.load:
        parser.error(
            "Choose either --train or --load. psy cannot train and load a "
            "model in the same run."
        )

    if (args.save or args.predict) and not (args.train or args.load):
        parser.error(
            "psy does not know which model to use. Please add --train to "
            "create a model or --load to load one."
        )


def main() -> None:
    define_logging()
    parser, args = parse_args()
    validate_args(parser, args)

    model = None

    if args.train:
        model = pipeline.run()
        if args.save:
            save_model(model)
    elif args.load:
        model = load_latest_model()

    if args.predict and model is not None:
        predict_folder(model)


if __name__ == "__main__":
    main()
