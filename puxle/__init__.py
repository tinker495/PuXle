"""PuXle public API facade."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"

__all__ = [
    "Puzzle",
    "PuzzleState",
    "FieldDescriptor",
    "state_dataclass",
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
    "PDDL",
    "Benchmark",
    "BenchmarkSample",
    "LightsOutDeepCubeABenchmark",
    "RubiksCubeDeepCubeABenchmark",
    "SlidePuzzleDeepCubeABenchmark",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "Puzzle": (".core.puzzle_base", "Puzzle"),
    "PuzzleState": (".core.puzzle_state", "PuzzleState"),
    "FieldDescriptor": (".core.puzzle_state", "FieldDescriptor"),
    "state_dataclass": (".core.puzzle_state", "state_dataclass"),
    "DotKnot": (".puzzles.dotknot", "DotKnot"),
    "TowerOfHanoi": (".puzzles.hanoi", "TowerOfHanoi"),
    "LightsOut": (".puzzles.lightsout", "LightsOut"),
    "LightsOutRandom": (".puzzles.lightsout", "LightsOutRandom"),
    "Maze": (".puzzles.maze", "Maze"),
    "Room": (".puzzles.room", "Room"),
    "PancakeSorting": (".puzzles.pancake", "PancakeSorting"),
    "RubiksCube": (".puzzles.rubikscube", "RubiksCube"),
    "RubiksCubeRandom": (".puzzles.rubikscube", "RubiksCubeRandom"),
    "SlidePuzzle": (".puzzles.slidepuzzle", "SlidePuzzle"),
    "SlidePuzzleHard": (".puzzles.slidepuzzle", "SlidePuzzleHard"),
    "SlidePuzzleRandom": (".puzzles.slidepuzzle", "SlidePuzzleRandom"),
    "Sokoban": (".puzzles.sokoban", "Sokoban"),
    "SokobanHard": (".puzzles.sokoban", "SokobanHard"),
    "TSP": (".puzzles.tsp", "TSP"),
    "TopSpin": (".puzzles.topspin", "TopSpin"),
    "PDDL": (".pddls.pddl", "PDDL"),
    "Benchmark": (".benchmark.benchmark", "Benchmark"),
    "BenchmarkSample": (".benchmark.benchmark", "BenchmarkSample"),
    "LightsOutDeepCubeABenchmark": (
        ".benchmark.lightsout_deepcubea",
        "LightsOutDeepCubeABenchmark",
    ),
    "RubiksCubeDeepCubeABenchmark": (
        ".benchmark.rubikscube_deepcubea",
        "RubiksCubeDeepCubeABenchmark",
    ),
    "SlidePuzzleDeepCubeABenchmark": (
        ".benchmark.slidepuzzle_deepcubea",
        "SlidePuzzleDeepCubeABenchmark",
    ),
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
