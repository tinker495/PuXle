from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Hashable, Iterable, Sequence, TypeVar

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import PuzzleState

__all__ = ["BenchmarkSample", "Benchmark"]

StateT = TypeVar("StateT", bound=PuzzleState)
SolveConfigT = TypeVar("SolveConfigT", bound=PuzzleState)


@dataclass(frozen=True)
class BenchmarkSample(Generic[StateT, SolveConfigT]):
    """Container for a benchmark instance."""

    state: StateT
    solve_config: SolveConfigT
    optimal_path: Sequence[Any]


class Benchmark(ABC, Generic[StateT, SolveConfigT]):
    """Abstract base class describing a benchmark dataset."""

    def __init__(self) -> None:
        self._puzzle: Puzzle | None = None
        self._dataset: Any = None

    @property
    def puzzle(self) -> Puzzle:
        """Return the puzzle used for this benchmark, constructing it lazily."""
        if self._puzzle is None:
            self._puzzle = self.build_puzzle()
        return self._puzzle

    @abstractmethod
    def build_puzzle(self) -> Puzzle:
        """Instantiate the puzzle that defines this benchmark."""

    @property
    def dataset(self) -> Any:
        """Load the dataset on demand and cache the result."""
        if self._dataset is None:
            self._dataset = self.load_dataset()
        return self._dataset

    @abstractmethod
    def load_dataset(self) -> Any:
        """Return the raw dataset object backing the benchmark."""

    @abstractmethod
    def sample_ids(self) -> Iterable[Hashable]:
        """Return iterable sample identifiers available in the dataset."""

    @abstractmethod
    def get_sample(self, sample_id: Hashable) -> BenchmarkSample[StateT, SolveConfigT]:
        """Fetch the state, solve configuration and optimal path for a sample."""
