from __future__ import annotations

import math
from importlib.resources import files
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

import jax.numpy as jnp
import numpy as np
from puxle.benchmark._deepcubea import load_deepcubea
from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
from puxle.core.puzzle_state import PuzzleState
from puxle.puzzles.lightsout import LightsOut

DEFAULT_DATASET_NAME = "size7-deepcubeA.pkl"
DATA_RELATIVE_PATH = Path("data") / "lightsout"


class LightsOutDeepCubeABenchmark(Benchmark):
    """Benchmark that exposes the DeepCubeA LightsOut dataset."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        dataset_name: str = DEFAULT_DATASET_NAME,
        size: int | None = None,
    ) -> None:
        super().__init__()
        self._dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        self._dataset_name = dataset_name
        self._size = size
        self._solve_config_cache = None

    def build_puzzle(self) -> LightsOut:
        return LightsOut(size=self._ensure_size())

    def load_dataset(self) -> dict[str, Any]:
        if self._dataset_path is not None:
            if not self._dataset_path.is_file():
                raise FileNotFoundError(
                    f"LightsOut DeepCubeA dataset not found at {self._dataset_path}"
                )
            with self._dataset_path.open("rb") as handle:
                return load_deepcubea(handle)

        try:
            resource = files("puxle.data.lightsout") / self._dataset_name
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

        return BenchmarkSample(
            state=state,
            solve_config=solve_config,
            optimal_action_sequence=None,
            optimal_path=None,
            optimal_path_costs=None,
        )

    def _ensure_size(self) -> int:
        if self._size is None:
            dataset = self.dataset
            states = dataset.get("states")
            if not states:
                raise ValueError("LightsOut dataset does not contain any states.")
            tiles = self._extract_tiles(states[0])
            length = len(tiles)
            size = int(math.isqrt(length))
            if size * size != length:
                raise ValueError(
                    f"Unable to infer puzzle size from state length {length}. Expected a perfect square."
                )
            self._size = size
        return self._size

    def _ensure_solve_config(self):
        if self._solve_config_cache is None:
            self._solve_config_cache = self.puzzle.get_solve_config()
        return self._solve_config_cache

    def verify_solution(
        self,
        sample: BenchmarkSample,
        states: Sequence[PuzzleState] | None = None,
        action_sequence: Sequence[str] | None = None,
    ) -> bool:
        """
        Verify that a solution is valid for the given sample.
        For 7x7 Lights Out, any solution without duplicate moves is considered optimal.
        """
        # If action_sequence is provided, check for duplicates (theorem condition)
        if action_sequence is not None:
            if len(set(action_sequence)) != len(action_sequence):
                # Duplicate moves found, so it might not be optimal (or is trivially redundant)
                return False

        return super().verify_solution(sample, states, action_sequence)

    @staticmethod
    def _extract_tiles(raw_state: Any):
        return getattr(raw_state, "tiles", raw_state)

    def _convert_state(self, raw_state: Any) -> PuzzleState:
        tiles = self._extract_tiles(raw_state)
        puzzle: LightsOut = self.puzzle
        board = np.asarray(tiles, dtype=np.bool_)
        if not puzzle.board_is_solvable(board, puzzle.size):
            raise ValueError(
                "Encountered unsolvable LightsOut state in DeepCubeA dataset. "
                f"State: {board.astype(int).tolist()}"
            )
        faces = jnp.asarray(board, dtype=jnp.bool_)
        return puzzle.State(board=faces).packed


__all__ = ["LightsOutDeepCubeABenchmark"]

