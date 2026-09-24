"""Command-line interface for project workflows."""

import argparse
import logging

from psychic.data.preprocessing import preprocess_cache_waveforms
from psychic.data.ravdess import load_ravdess_samples
from psychic.data.splits import build_speaker_disjoint_splits
from psychic.logging import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="psy",
        description="Speech emotion recognition workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "preprocess",
        help="Build the deterministic RAVDESS waveform cache.",
    )
    subparsers.add_parser("train")
    subparsers.add_parser("eval")
    subparsers.add_parser("predict-file")

    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging()
    logger.info("\U0001f680 Starting pipeline: %s", args.command)

    if args.command == "preprocess":
        samples = load_ravdess_samples()
        splits = build_speaker_disjoint_splits(samples)
        preprocess_cache_waveforms(samples, splits)
