"""Puzzle implementations for PuXle."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

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
    "CayleyPancake7",
    "CayleyPancake8",
    "CayleyLRX8",
    "CayleyTopSpin8K4",
    "CayleyCoxeter8",
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
    "CayleyPuzzle": (".cayley_puzzle", "CayleyPuzzle"),
    "CayleyPancake7": (".cayley_subclasses", "CayleyPancake7"),
    "CayleyPancake8": (".cayley_subclasses", "CayleyPancake8"),
    "CayleyLRX8": (".cayley_subclasses", "CayleyLRX8"),
    "CayleyTopSpin8K4": (".cayley_subclasses", "CayleyTopSpin8K4"),
    "CayleyCoxeter8": (".cayley_subclasses", "CayleyCoxeter8"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
