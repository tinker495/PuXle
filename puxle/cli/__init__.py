from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from . import human_play, world_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puxle", description="PuXle puzzle utilities."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    human_play.configure_parser(
        commands.add_parser("human-play", help="Play a puzzle interactively.")
    )
    world_model.configure_parser(
        commands.add_parser(
            "world-model-train", help="Generate datasets and train world models."
        )
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


__all__ = ["build_parser", "main"]
