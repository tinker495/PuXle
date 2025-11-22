from __future__ import annotations

import math
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

import jax.numpy as jnp
from puxle.benchmark._deepcubea import load_deepcubea
from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
from puxle.core.puzzle_state import PuzzleState
from puxle.puzzles.slidepuzzle import SlidePuzzle

class SlidePuzzlePreset(Enum):
    SIZE15 = ("size15-deepcubeA.pkl", 4)
    SIZE24 = ("size24-deepcubeA.pkl", 5)
    SIZE35 = ("size35-deepcubeA.pkl", 6)
    SIZE48 = ("size48-deepcubeA.pkl", 7)

    def __init__(self, dataset_name: str, board_size: int):
        self.dataset_name = dataset_name
        self.board_size = board_size


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
        self._dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        preset_dataset_name = preset.dataset_name if preset else DEFAULT_DATASET_NAME
        preset_board_size = preset.board_size if preset else None
        self._dataset_name = dataset_name or preset_dataset_name
        self._board_size = board_size or preset_board_size
        self._solve_config_cache = None
        self._notation_to_action: dict[str, int] | None = None

    def build_puzzle(self) -> SlidePuzzle:
        return SlidePuzzle(size=self._ensure_board_size())

    def load_dataset(self) -> dict[str, Any]:
        if self._dataset_path is not None:
            if not self._dataset_path.is_file():
                raise FileNotFoundError(
                    f"SlidePuzzle DeepCubeA dataset not found at {self._dataset_path}"
                )
            with self._dataset_path.open("rb") as handle:
                return load_deepcubea(handle)

        try:
            resource = files("puxle.data.slidepuzzle") / self._dataset_name
            with resource.open("rb") as handle:
                return load_deepcubea(handle)
        except (ModuleNotFoundError, FileNotFoundError):
            pass

        fallback = Path(__file__).resolve().parents[1] / DATA_RELATIVE_PATH / self._dataset_name
        if not fallback.is_file():
            raise FileNotFoundError(
                f"Unable to locate {self._dataset_name} under package resources or at {fallback}"
            )
        with fallback.open("rb") as handle:
            return load_deepcubea(handle)

    def sample_ids(self) -> Iterable[Hashable]:
        return range(len(self.dataset["states"]))

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
            optimal_action_sequence, optimal_cost = self._convert_solution(solutions[index])
            optimal_path = self._build_optimal_path(state, solve_config, optimal_action_sequence)

        return BenchmarkSample(
            state=state,
            solve_config=solve_config,
            optimal_action_sequence=optimal_action_sequence,
            optimal_path=optimal_path,
            optimal_path_costs=optimal_cost,
        )

    def _ensure_board_size(self) -> int:
        if self._board_size is None:
            dataset = self.dataset
            states = dataset.get("states")
            if not states:
                raise ValueError("SlidePuzzle dataset does not contain any states.")
            tiles = self._extract_tiles(states[0])
            length = len(tiles)
            size = int(math.isqrt(length))
            if size * size != length:
                raise ValueError(
                    f"Unable to infer puzzle size from state length {length}. Expected a perfect square."
                )
            self._board_size = size
        return self._board_size

    def _ensure_solve_config(self):
        if self._solve_config_cache is None:
            self._solve_config_cache = self.puzzle.get_solve_config()
        return self._solve_config_cache

    def _extract_tiles(self, raw_state: Any):
        return getattr(raw_state, "tiles", raw_state)

    def _convert_state(self, raw_state: Any) -> PuzzleState:
        tiles = jnp.asarray(self._extract_tiles(raw_state), dtype=jnp.uint8)
        puzzle: SlidePuzzle = self.puzzle
        return puzzle.State(board=tiles).packed

    def _convert_solution(self, moves: Sequence[str]) -> tuple[tuple[str, ...], float]:
        if not moves:
            return tuple(), 0.0

        action_sequence: list[str] = []
        for move in moves:
            try:
                action_sequence.append(MOVE_TO_NOTATION[move.upper()])
            except KeyError as exc:
                raise ValueError(f"Unsupported move '{move}' in DeepCubeA slide puzzle dataset.") from exc
        optimal_action_sequence = tuple(action_sequence)
        return optimal_action_sequence, float(len(optimal_action_sequence))

    def _build_action_lookup(self) -> dict[str, int]:
        if self._notation_to_action is None:
            puzzle = self.puzzle
            self._notation_to_action = {
                puzzle.action_to_string(action): action for action in range(puzzle.action_size)
            }
        return self._notation_to_action

    def _build_optimal_path(
        self,
        initial_state: PuzzleState,
        solve_config: SlidePuzzle.SolveConfig,
        action_sequence: Sequence[str] | None,
    ) -> tuple[PuzzleState, ...] | None:
        if not action_sequence:
            return None

        action_lookup = self._build_action_lookup()
        puzzle = self.puzzle
        current_state = initial_state
        path: list[PuzzleState] = []

        for step, notation in enumerate(action_sequence, start=1):
            try:
                action_idx = action_lookup[notation]
            except KeyError as exc:
                raise KeyError(f"Unknown action notation '{notation}' at step {step}") from exc

            neighbours, _ = puzzle.get_neighbours(solve_config, current_state, filled=True)
            next_state = neighbours[action_idx]
            path.append(next_state)
            current_state = next_state

        return tuple(path)


class SlidePuzzleDeepCubeA15Benchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path, dataset_name=SlidePuzzlePreset.SIZE15.dataset_name, board_size=SlidePuzzlePreset.SIZE15.board_size)

class SlidePuzzleDeepCubeA24Benchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path, dataset_name=SlidePuzzlePreset.SIZE24.dataset_name, board_size=SlidePuzzlePreset.SIZE24.board_size)

class SlidePuzzleDeepCubeA35Benchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path, dataset_name=SlidePuzzlePreset.SIZE35.dataset_name, board_size=SlidePuzzlePreset.SIZE35.board_size)

class SlidePuzzleDeepCubeA48Benchmark(SlidePuzzleDeepCubeABenchmark):
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__(dataset_path, dataset_name=SlidePuzzlePreset.SIZE48.dataset_name, board_size=SlidePuzzlePreset.SIZE48.board_size)


__all__ = ["SlidePuzzleDeepCubeABenchmark", "SlidePuzzlePreset", "SlidePuzzleDeepCubeA15Benchmark", "SlidePuzzleDeepCubeA24Benchmark", "SlidePuzzleDeepCubeA35Benchmark", "SlidePuzzleDeepCubeA48Benchmark"]

