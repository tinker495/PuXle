"""
PuXle: Parallelized Puzzles with JAX

A high-performance library for parallelized puzzle environments built on JAX.
Provides a collection of classic puzzles optimized for AI research, reinforcement learning,
and search algorithms.
"""

# Core framework
from puxle.core import FieldDescriptor, Puzzle, PuzzleState, state_dataclass
from puxle.benchmark import Benchmark, BenchmarkSample 
from puxle.pddls import PDDL

# All puzzle implementations
from puxle.puzzles import (
    TSP,
    DotKnot,
    LightsOut,
    LightsOutRandom,
    Maze,
    PancakeSorting,
    Room,
    RubiksCube,
    RubiksCubeRandom,
    SlidePuzzle,
    SlidePuzzleHard,
    SlidePuzzleRandom,
    Sokoban,
    SokobanHard,
    TopSpin,
    TowerOfHanoi,
)

__version__ = "0.1.0"

__all__ = [
    # Core framework
    "Puzzle",
    "PuzzleState",
    "FieldDescriptor",
    "state_dataclass",
    # Puzzle implementations
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
    # Benchmark
    "Benchmark",
    "BenchmarkSample",
    "RubiksCubeDeepCubeABenchmark",
]
