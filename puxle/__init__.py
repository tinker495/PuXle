"""PuXle public API facade."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__version__ = "0.2.0"

__all__ = [
    "Puzzle",
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
    "Benchmark",
    "BenchmarkSample",
    "LightsOutDeepCubeABenchmark",
    "RubiksCubeDeepCubeABenchmark",
    "SlidePuzzleDeepCubeABenchmark",
    "RubiksCubeWorldModel",
    "RubiksCubeWorldModelOptimized",
    "SokobanWorldModel",
    "SokobanWorldModelOptimized",
    "WorldModelPuzzleBase",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "Puzzle": (".core.puzzle_base", "Puzzle"),
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
    "CayleyPuzzle": (".puzzles.adapters.cayley", "CayleyPuzzle"),
    "CayleyPresets": (".puzzles.adapters.cayley", "CayleyPresets"),
    "PDDL": (".puzzles.adapters.pddl", "PDDL"),
    "PDDLPresets": (".puzzles.adapters.pddl", "PDDLPresets"),
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
    "RubiksCubeWorldModel": (".world_model", "RubiksCubeWorldModel"),
    "RubiksCubeWorldModelOptimized": (".world_model", "RubiksCubeWorldModelOptimized"),
    "SokobanWorldModel": (".world_model", "SokobanWorldModel"),
    "SokobanWorldModelOptimized": (".world_model", "SokobanWorldModelOptimized"),
    "WorldModelPuzzleBase": (".world_model", "WorldModelPuzzleBase"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
