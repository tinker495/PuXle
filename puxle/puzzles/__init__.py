"""Puzzle implementations for PuXle."""

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING

from puxle._lazy_imports import lazy_dir, load_lazy_export

if TYPE_CHECKING:
    from puxle.core.puzzle_base import Puzzle

__all__ = [
    "DotKnot",
    "TowerOfHanoi",
    "LightsOut",
    "LightsOutRandom",
    "Maze",
    "Room",
    "PancakeSorting",
    "RubiksCube",
    "RubiksCubeRandom",
    "SlidePuzzle",
    "SlidePuzzleHard",
    "SlidePuzzleRandom",
    "Sokoban",
    "SokobanHard",
    "TSP",
    "TopSpin",
    "CayleyPuzzle",
    "CayleyPresets",
    "PDDL",
    "PDDLPresets",
    "WorldModelPuzzle",
    "WorldModelPresets",
]

_EXPORTS = {
    "DotKnot": (".dotknot", "DotKnot"),
    "TowerOfHanoi": (".hanoi", "TowerOfHanoi"),
    "LightsOut": (".lightsout", "LightsOut"),
    "LightsOutRandom": (".lightsout", "LightsOutRandom"),
    "Maze": (".maze", "Maze"),
    "Room": (".room", "Room"),
    "PancakeSorting": (".pancake", "PancakeSorting"),
    "RubiksCube": (".rubikscube", "RubiksCube"),
    "RubiksCubeRandom": (".rubikscube", "RubiksCubeRandom"),
    "SlidePuzzle": (".slidepuzzle", "SlidePuzzle"),
    "SlidePuzzleHard": (".slidepuzzle", "SlidePuzzleHard"),
    "SlidePuzzleRandom": (".slidepuzzle", "SlidePuzzleRandom"),
    "Sokoban": (".sokoban", "Sokoban"),
    "SokobanHard": (".sokoban", "SokobanHard"),
    "TSP": (".tsp", "TSP"),
    "TopSpin": (".topspin", "TopSpin"),
    "CayleyPuzzle": (".adapters.cayley", "CayleyPuzzle"),
    "CayleyPresets": (".adapters.cayley", "CayleyPresets"),
    "PDDL": (".adapters.pddl", "PDDL"),
    "PDDLPresets": (".adapters.pddl", "PDDLPresets"),
    "WorldModelPuzzle": (".adapters.world_model", "WorldModelPuzzle"),
    "WorldModelPresets": (".adapters.world_model", "WorldModelPresets"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)


def normalize_puzzle_name(name: str) -> str:
    """Normalize a puzzle lookup key to lowercase alphanumeric characters."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def discover_puzzle_classes() -> dict[str, type[Puzzle]]:
    """Return concrete Puzzle subclasses keyed by canonical name and normalized aliases."""
    from puxle.core.puzzle_base import Puzzle

    module = sys.modules[__name__]
    classes = {
        value.__name__: value
        for name in __all__
        if inspect.isclass(value := getattr(module, name))
        and issubclass(value, Puzzle)
        and value is not Puzzle
        and not inspect.isabstract(value)
    }
    aliases = {normalize_puzzle_name(name): value for name, value in classes.items()}
    if "SlidePuzzle" in classes:
        aliases["npuzzle"] = classes["SlidePuzzle"]
    return classes | aliases


def resolve_puzzle_class(name: str) -> type[Puzzle]:
    """Resolve a canonical or normalized puzzle name to its concrete Puzzle class."""
    classes = discover_puzzle_classes()
    puzzle_class = classes.get(name) or classes.get(normalize_puzzle_name(name))
    if puzzle_class is None:
        available = ", ".join(sorted(key for key in classes if not key.islower()))
        raise ValueError(f"Unknown puzzle '{name}'. Available puzzles: {available}")
    return puzzle_class
