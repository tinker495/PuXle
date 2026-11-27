from __future__ import annotations

import json
from typing import Annotated, Any

import tyro

from puxle.core.puzzle_base import Puzzle
from puxle.puzzles import (
    discover_puzzle_classes,
    normalize_puzzle_name,
    resolve_puzzle_class,
)

puzzle_classes = discover_puzzle_classes

PuzzleName = Annotated[
    str,
    tyro.conf.arg(
        constructor_factory=lambda: tyro.extras.literal_type_from_choices(
            discover_puzzle_classes()
        )
    ),
]


def create_puzzle(name: str, puzzle_args: str) -> Puzzle:
    puzzle_class = resolve_puzzle_class(name)
    kwargs: dict[str, Any] = json.loads(puzzle_args)
    if not isinstance(kwargs, dict):
        raise ValueError("--puzzle-args must be a JSON object")
    return puzzle_class(**kwargs)


def data_name_for_puzzle(puzzle: Puzzle) -> str:
    normalized = normalize_puzzle_name(type(puzzle).__name__)
    if normalized.startswith("rubikscube"):
        return "rubikscube"
    if normalized.startswith("sokoban"):
        return "sokoban"
    return normalized
