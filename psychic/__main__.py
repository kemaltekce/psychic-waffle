import coloredlogs
import logging

from psychic import pipeline

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


def main() -> None:
    define_logging()
    pipeline.run()


if __name__ == "__main__":
    main()
