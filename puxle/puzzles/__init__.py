"""Puzzle implementations for PuXle."""

from __future__ import annotations

import importlib
from typing import Any

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
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
