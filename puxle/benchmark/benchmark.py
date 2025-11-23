from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Hashable, Iterable, Optional, Sequence, TypeVar

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
    optimal_action_sequence: Optional[None | Sequence[str]]
    optimal_path: Optional[None | Sequence[StateT]]
    optimal_path_costs: Optional[None | float]


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
        """Fetch the state, solve configuration and optimal action sequence for a sample."""

    def verify_solution(
        self,
        sample: BenchmarkSample[StateT, SolveConfigT],
        states: Sequence[StateT] | None = None,
        action_sequence: Sequence[str] | None = None,
    ) -> bool:
        """
        Verify that a solution is valid and optimal for the given sample.

        If `action_sequence` or `states` are provided, they are treated as the candidate solution.
        Otherwise, verifies `sample.optimal_action_sequence`.

        Checks:
        1. The sequence of actions leads to a solved state.
        2. If `sample.optimal_path_costs` is known and we are verifying a candidate solution,
           checks if the candidate length is <= optimal cost.
        """
        target_sequence = action_sequence if action_sequence is not None else sample.optimal_action_sequence
        target_path = states if states is not None else sample.optimal_path

        if target_sequence is None:
            if target_path and not action_sequence:
                # Only raise if we are validating the sample itself and it's inconsistent
                if sample.optimal_path and not sample.optimal_action_sequence:
                    raise ValueError("Sample has optimal_path but no optimal_action_sequence.")
            return True

        final_state: StateT
        # Use path if available to avoid simulation
        if target_path:
            final_state = target_path[-1]
        else:
            # Reconstruct path from sequence
            puzzle = self.puzzle
            action_lookup = {
                puzzle.action_to_string(action): action for action in range(puzzle.action_size)
            }
            current_state = sample.state
            for i, notation in enumerate(target_sequence):
                if notation not in action_lookup:
                    raise KeyError(f"Unknown action notation '{notation}' at index {i}")
                action_idx = action_lookup[notation]
                neighbours, _ = puzzle.get_neighbours(sample.solve_config, current_state, filled=True)
                current_state = neighbours[action_idx]
            final_state = current_state

        # 1. Check validity
        if not self.puzzle.is_solved(sample.solve_config, final_state):
            return False

        # 2. Check optimality (if checking a candidate against known optimal cost)
        if action_sequence is not None and sample.optimal_path_costs is not None:
            # We use loose comparison to account for potential float costs,
            # though usually length is integer.
            if len(action_sequence) > sample.optimal_path_costs:
                return False

        return True
