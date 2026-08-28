"""Logging setup shared by command-line workflows."""

from __future__ import annotations

import logging

import coloredlogs

LOGGER_NAME = "psychic"
LOGGER_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s  %(message)s"
THIRD_PARTY_LOGGERS = (
    "asyncio",
    "matplotlib",
    "numba",
    "numexpr",
    "urllib3",
)


def configure_logging() -> None:
    """Configure project logging for command-line workflows."""
    coloredlogs.install(
        level=LOGGER_LEVEL,
        logger=logging.getLogger(LOGGER_NAME),
        fmt=LOG_FORMAT,
        field_styles={
            "asctime": {"color": "green"},
            "name": {"color": "blue"},
            "levelname": {"color": "black", "bright": True},
        },
    )

    for logger_name in THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
