from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Sequence

import jax.numpy as jnp
import numpy as np
from xtructure import Xtructurable

from puxle.benchmark._deepcubea import (
    extract_tiles,
    infer_square_size,
    load_deepcubea_dataset,
)
from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
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
        self._dataset_path = self._normalize_dataset_path(dataset_path)
        self._dataset_name = dataset_name
        self._size = size

    def build_puzzle(self) -> LightsOut:
        return LightsOut(size=self._ensure_size())

    def load_dataset(self) -> dict[str, Any]:
        fallback_dir = Path(__file__).resolve().parents[1] / DATA_RELATIVE_PATH
        return load_deepcubea_dataset(
            self._dataset_path, self._dataset_name, "puxle.data.lightsout", fallback_dir
        )

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
        return self._ensure_cached(
            "_size",
            lambda: infer_square_size(self.dataset.get("states"), "LightsOut"),
        )

    def verify_solution(
        self,
        sample: BenchmarkSample,
        states: Sequence[Xtructurable] | None = None,
        action_sequence: Sequence[str] | None = None,
    ) -> bool | None:
        """
        Verify that a solution is valid for the given sample.
        For 7x7 Lights Out, any solution without duplicate moves is considered optimal.
        """
        # If action_sequence is provided, check for duplicates (theorem condition)
        if action_sequence is not None:
            if len(set(action_sequence)) != len(action_sequence):
                # Duplicate moves found, so it might not be optimal (or is trivially redundant)
                return False

        result = super().verify_solution(sample, states, action_sequence)

        # If base class returns None, it means solved but no ground truth to compare.
        # Since we passed the duplicate check (which is our optimality condition),
        # we can confirm it is optimal.
        if result is None:
            return True

        return result

    def _convert_state(self, raw_state: Any) -> Xtructurable:
        tiles = extract_tiles(raw_state)
        puzzle: LightsOut = self.puzzle
        board = np.asarray(tiles, dtype=np.bool_)
        if not puzzle.board_is_solvable(board, puzzle.size):
            raise ValueError(
                f"Encountered unsolvable LightsOut state in DeepCubeA dataset. State: {board.astype(int).tolist()}"
            )
        faces = jnp.asarray(board, dtype=jnp.bool_)
        return puzzle.State.from_unpacked(board=faces)


__all__ = ["LightsOutDeepCubeABenchmark"]
