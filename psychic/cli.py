"""Command-line interface for project workflows."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="psy",
        description="Speech emotion recognition workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preprocess")
    subparsers.add_parser("train")
    subparsers.add_parser("eval")
    subparsers.add_parser("predict-file")

    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    parser.parse_args()
