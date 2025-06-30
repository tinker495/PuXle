"""
Puzzle implementations for PuXle.

This module contains implementations of various classic puzzles optimized for JAX-based computation.
All puzzle classes inherit from the base Puzzle class and provide parallelized environments
for AI research and reinforcement learning.
"""

from puxle.puzzles.dotknot import DotKnot
from puxle.puzzles.hanoi import TowerOfHanoi
from puxle.puzzles.lightsout import LightsOut, LightsOutRandom
from puxle.puzzles.maze import Maze
from puxle.puzzles.room import Room
from puxle.puzzles.pancake import PancakeSorting
from puxle.puzzles.rubikscube import RubiksCube, RubiksCubeRandom
from puxle.puzzles.slidepuzzle import SlidePuzzle, SlidePuzzleHard, SlidePuzzleRandom
from puxle.puzzles.sokoban import Sokoban, SokobanHard
from puxle.puzzles.topspin import TopSpin
from puxle.puzzles.tsp import TSP
from puxle.puzzles.knapsack import Knapsack
from puxle.puzzles.cvrp import CVRP
from puxle.puzzles.jobshop import JobShop

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
    "Knapsack",
    "JobShop",
    "CVRP",
]