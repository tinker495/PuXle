from __future__ import annotations

import inspect
import json
from typing import Annotated, Any

import tyro

from puxle.core.puzzle_base import Puzzle


def _normalized(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def puzzle_classes() -> dict[str, type[Puzzle]]:
    import puxle.puzzles as puzzle_module

    classes = {
        value.__name__: value
        for name in dir(puzzle_module)
        if inspect.isclass(value := getattr(puzzle_module, name))
        if issubclass(value, Puzzle)
        and value is not Puzzle
        and not inspect.isabstract(value)
    }
    aliases = {_normalized(name): value for name, value in classes.items()}
    if "SlidePuzzle" in classes:
        aliases["npuzzle"] = classes["SlidePuzzle"]
    return classes | aliases


PuzzleName = Annotated[
    str,
    tyro.conf.arg(
        constructor_factory=lambda: tyro.extras.literal_type_from_choices(
            puzzle_classes()
        )
    ),
]


def create_puzzle(name: str, puzzle_args: str) -> Puzzle:
    classes = puzzle_classes()
    puzzle_class = classes.get(name) or classes.get(_normalized(name))
    if puzzle_class is None:
        available = ", ".join(sorted(key for key in classes if not key.islower()))
        raise ValueError(f"Unknown puzzle '{name}'. Available: {available}")
    kwargs: dict[str, Any] = json.loads(puzzle_args)
    if not isinstance(kwargs, dict):
        raise ValueError("--puzzle-args must be a JSON object")
    return puzzle_class(**kwargs)


def data_name_for_puzzle(puzzle: Puzzle) -> str:
    normalized = _normalized(type(puzzle).__name__)
    if normalized.startswith("rubikscube"):
        return "rubikscube"
    if normalized.startswith("sokoban"):
        return "sokoban"
    return normalized
