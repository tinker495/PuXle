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
