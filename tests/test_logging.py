import logging

from psychic.logging import LOGGER_LEVEL, LOGGER_NAME, configure_logging


def test_configure_logging() -> None:
    configure_logging()

    assert LOGGER_LEVEL == logging.DEBUG, "Correct level should be DEBUG"
    assert (
        logging.getLogger("matplotlib").level == logging.WARNING
    ), "Noisy third party loggers should be suppressed"
    assert (
        logging.getLogger("urllib3").level == logging.WARNING
    ), "Noisy third party loggers should be suppressed"


def test_module_loggers_are_children_of_project_logger() -> None:
    module_logger = logging.getLogger("psychic.cli")

    assert module_logger.name.startswith(f"{LOGGER_NAME}."), (
        "The logger name should be psychic, so that the config can properly "
        "propagate to children loggers."
    )
    assert module_logger.propagate
