from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Hashable, Sequence

import jax.numpy as jnp

from puxle.benchmark._deepcubea import (
    extract_tiles,
    infer_square_size,
    load_deepcubea_dataset,
)
from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
from puxle.core.puzzle_state import PuzzleState
from puxle.puzzles.slidepuzzle import SlidePuzzle

HARD_17_STATES = [
    [0, 12, 9, 13, 15, 11, 10, 14, 3, 7, 2, 5, 4, 8, 6, 1],
    [0, 12, 10, 13, 15, 11, 14, 9, 3, 7, 2, 5, 4, 8, 6, 1],
    [0, 11, 9, 13, 12, 15, 10, 14, 3, 7, 6, 2, 4, 8, 5, 1],
    [0, 15, 9, 13, 11, 12, 10, 14, 3, 7, 6, 2, 4, 8, 5, 1],
    [0, 12, 9, 13, 15, 11, 10, 14, 3, 7, 6, 2, 4, 8, 5, 1],
    [0, 12, 14, 13, 15, 11, 9, 10, 3, 7, 6, 2, 4, 8, 5, 1],
    [0, 12, 10, 13, 15, 11, 14, 9, 3, 7, 6, 2, 4, 8, 5, 1],
    [0, 12, 11, 13, 15, 14, 10, 9, 3, 7, 6, 2, 4, 8, 5, 1],
    [0, 12, 10, 13, 15, 11, 9, 14, 7, 3, 6, 2, 4, 8, 5, 1],
    [0, 12, 9, 13, 15, 11, 14, 10, 3, 8, 6, 2, 4, 7, 5, 1],
    [0, 12, 9, 13, 15, 11, 10, 14, 8, 3, 6, 2, 4, 7, 5, 1],
    [0, 12, 14, 13, 15, 11, 9, 10, 8, 3, 6, 2, 4, 7, 5, 1],
    [0, 12, 9, 13, 15, 11, 10, 14, 7, 8, 6, 2, 4, 3, 5, 1],
    [0, 12, 10, 13, 15, 11, 14, 9, 7, 8, 6, 2, 4, 3, 5, 1],
    [0, 12, 9, 13, 15, 8, 10, 14, 11, 7, 6, 2, 4, 3, 5, 1],
    [0, 12, 9, 13, 15, 11, 10, 14, 3, 7, 5, 6, 4, 8, 2, 1],
    [0, 12, 9, 13, 15, 11, 10, 14, 7, 8, 5, 6, 4, 3, 2, 1],
]


class SlidePuzzlePreset(Enum):
    SIZE15 = ("size15-deepcubeA.pkl", 4)
    SIZE24 = ("size24-deepcubeA.pkl", 5)
    SIZE35 = ("size35-deepcubeA.pkl", 6)
    SIZE48 = ("size48-deepcubeA.pkl", 7)
    SIZE15_HARD = ("size15-deepcubeA.pkl", 4)

    def __init__(self, dataset_name: str, board_size: int):
        self.dataset_name = dataset_name
        self.board_size = board_size

    @property
    def states(self) -> Sequence[Any] | None:
        if self is SlidePuzzlePreset.SIZE15_HARD:
            return HARD_17_STATES
        return None


DEFAULT_DATASET_NAME = SlidePuzzlePreset.SIZE15.dataset_name
DATA_RELATIVE_PATH = Path("data") / "slidepuzzle"
MOVE_TO_NOTATION = {
    "L": "←",
    "R": "→",
    "U": "↑",
    "D": "↓",
}


class SlidePuzzleDeepCubeABenchmark(Benchmark):
    """Benchmark exposing the DeepCubeA slide puzzle datasets (4x4 through 7x7)."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        dataset_name: str | None = None,
        board_size: int | None = None,
        preset: SlidePuzzlePreset | None = SlidePuzzlePreset.SIZE15,
    ) -> None:
        super().__init__()
        self._dataset_path = self._normalize_dataset_path(dataset_path)
        preset_dataset_name = preset.dataset_name if preset else DEFAULT_DATASET_NAME
        preset_board_size = preset.board_size if preset else None
        self._dataset_name = dataset_name or preset_dataset_name
        self._board_size = board_size or preset_board_size
        self._explicit_states = preset.states if preset else None

    def build_puzzle(self) -> SlidePuzzle:
        return SlidePuzzle(size=self._ensure_board_size())

    def load_dataset(self) -> dict[str, Any]:
        if self._explicit_states is not None:
            return {"states": self._explicit_states, "solutions": None}

        fallback_dir = Path(__file__).resolve().parents[1] / DATA_RELATIVE_PATH
        return load_deepcubea_dataset(
            self._dataset_path,
            self._dataset_name,
            "puxle.data.slidepuzzle",
            fallback_dir,
        )

    def get_sample(self, sample_id: Hashable) -> BenchmarkSample:
        index = int(sample_id)
        dataset = self.dataset
        state = self._convert_state(dataset["states"][index])
        solve_config = self._ensure_solve_config()

        solutions = dataset.get("solutions")
        optimal_action_sequence = None
        optimal_path = None
        optimal_cost = None
        if solutions:
            optimal_action_sequence, optimal_cost = self._convert_solution(
                solutions[index]
            )
            optimal_path = self._build_optimal_path(
                state, solve_config, optimal_action_sequence
            )

        return BenchmarkSample(
            state=state,
            solve_config=solve_config,
            optimal_action_sequence=optimal_action_sequence,
            optimal_path=optimal_path,
            optimal_path_costs=optimal_cost,
        )

    def _ensure_board_size(self) -> int:
        return self._ensure_cached(
            "_board_size",
            lambda: infer_square_size(self.dataset.get("states"), "SlidePuzzle"),
        )

    def _convert_state(self, raw_state: Any) -> PuzzleState:
        tiles = jnp.asarray(extract_tiles(raw_state), dtype=jnp.uint8)
        puzzle: SlidePuzzle = self.puzzle
        return puzzle.State.from_unpacked(board=tiles)

    def _convert_solution(self, moves: Sequence[str]) -> tuple[tuple[str, ...], float]:
        if not moves:
            return tuple(), 0.0

        action_sequence: list[str] = []
        for move in moves:
            try:
                action_sequence.append(MOVE_TO_NOTATION[move.upper()])
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported move '{move}' in DeepCubeA slide puzzle dataset."
                ) from exc
        optimal_action_sequence = tuple(action_sequence)
        return optimal_action_sequence, float(len(optimal_action_sequence))


class SlidePuzzleDeepCubeA15Benchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path=dataset_path, preset=SlidePuzzlePreset.SIZE15)


class SlidePuzzleDeepCubeA24Benchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path=dataset_path, preset=SlidePuzzlePreset.SIZE24)


class SlidePuzzleDeepCubeA35Benchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path=dataset_path, preset=SlidePuzzlePreset.SIZE35)


class SlidePuzzleDeepCubeA48Benchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path=dataset_path, preset=SlidePuzzlePreset.SIZE48)


class SlidePuzzleDeepCubeA15HardBenchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path, preset=SlidePuzzlePreset.SIZE15_HARD)

    def verify_solution(
        self,
        sample: BenchmarkSample,
        states: Sequence[PuzzleState] | None = None,
        action_sequence: Sequence[str] | None = None,
    ) -> bool | None:
        is_solved = super().verify_solution(sample, states, action_sequence)
        if is_solved is False:
            return False

        candidate_cost = 0
        if action_sequence is not None:
            candidate_cost = len(action_sequence)
        elif states is not None:
            candidate_cost = max(0, len(states) - 1)

        if is_solved is None:
            return candidate_cost <= 80

        return is_solved


__all__ = [
    "SlidePuzzleDeepCubeABenchmark",
    "SlidePuzzlePreset",
    "SlidePuzzleDeepCubeA15Benchmark",
    "SlidePuzzleDeepCubeA24Benchmark",
    "SlidePuzzleDeepCubeA35Benchmark",
    "SlidePuzzleDeepCubeA48Benchmark",
    "SlidePuzzleDeepCubeA15HardBenchmark",
]
