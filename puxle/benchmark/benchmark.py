from __future__ import annotations

import math
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
    """Immutable container for a single benchmark instance.

    Attributes:
        state: Initial puzzle state for this sample.
        solve_config: Target/goal configuration for verification.
        optimal_action_sequence: Known-optimal action string list, or
            ``None`` if unavailable.
        optimal_path: Sequence of states along the optimal path, or
            ``None``.
        optimal_path_costs: Total cost of the optimal path, or ``None``.
    """

    state: StateT
    solve_config: SolveConfigT
    optimal_action_sequence: Optional[None | Sequence[str]]
    optimal_path: Optional[None | Sequence[StateT]]
    optimal_path_costs: Optional[None | float]


class Benchmark(ABC, Generic[StateT, SolveConfigT]):
    """Abstract base class for a benchmark dataset.

    Subclasses must implement:

    * :meth:`build_puzzle` — create the :class:`Puzzle` instance.
    * :meth:`load_dataset` — load or generate the raw dataset.
    * :meth:`sample_ids` — enumerate available sample IDs.
    * :meth:`get_sample` — retrieve a :class:`BenchmarkSample` by ID.

    The base class provides lazy caching for ``puzzle`` and ``dataset``
    and a generic :meth:`verify_solution` that checks both validity
    (is the final state solved?) and optimality (is the cost ≤ optimal?).
    """

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
    ) -> bool | None:
        """
        Verify that a solution is valid and optimal for the given sample.

        If `action_sequence` or `states` are provided, they are treated as the candidate solution.
        Otherwise, verifies `sample.optimal_action_sequence`.

        Returns:
            - True: if valid (solved) and length matches optimal (<= optimal cost).
            - False: if invalid (not solved) or suboptimal (> optimal cost).
            - None: if valid (solved) but sample has no optimal info to compare against.
        """
        target_sequence = action_sequence if action_sequence is not None else sample.optimal_action_sequence
        target_path = states if states is not None else sample.optimal_path

        if target_sequence is None:
            if target_path and not action_sequence:
                # Only raise if we are validating the sample itself and it's inconsistent
                if sample.optimal_path and not sample.optimal_action_sequence:
                    raise ValueError("Sample has optimal_path but no optimal_action_sequence.")
            # If no sequence provided and sample has none, we can't verify steps.
            # But if path provided, we can check validity.
            if target_path is None:
                return None

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

        # 2. Check optimality
        if sample.optimal_action_sequence is None:
            return None

        optimal_cost = sample.optimal_path_costs
        if optimal_cost is None:
            optimal_cost = len(sample.optimal_action_sequence)
        
        candidate_cost = 0
        if action_sequence is not None:
            candidate_cost = len(action_sequence)
        elif states is not None:
            candidate_cost = max(0, len(states) - 1)
        else:
            # Verifying sample against itself
            candidate_cost = len(sample.optimal_action_sequence)

        # Allow for floating point inaccuracies if costs are floats
        if candidate_cost > optimal_cost:
            if math.isclose(candidate_cost, optimal_cost):
                return True
            return False

        return True
