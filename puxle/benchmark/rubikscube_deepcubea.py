from __future__ import annotations

from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

import jax
import jax.numpy as jnp
from puxle.core.puzzle_state import PuzzleState
from puxle.benchmark._deepcubea import load_deepcubea
from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
from puxle.puzzles.rubikscube import RubiksCube

DEFAULT_DATASET_NAME = "size3-deepcubeA.pkl"
DATA_RELATIVE_PATH = Path("data") / "rubikscube" / DEFAULT_DATASET_NAME

POS_MAP = (
    6,  3,  0,  7,  4,  1,  8,  5,  2, # U
    51, 48, 45, 52, 49, 46, 53, 50, 47, # D
    42, 39, 36, 43, 40, 37, 44, 41, 38, # L
    24, 21, 18, 25, 22, 19, 26, 23, 20, # R
    33, 30, 27, 34, 31, 28, 35, 32, 29, # B
    15, 12,  9, 16, 13, 10, 17, 14, 11, # F
)
ID_MAP = (
    6,  3,  0,  7,  4,  1,  8,  5,  2, # U
    51, 48, 45, 52, 49, 46, 53, 50, 47, # D
    42, 39, 36, 43, 40, 37, 44, 41, 38, # L
    24, 21, 18, 25, 22, 19, 26, 23, 20, # R
    33, 30, 27, 34, 31, 28, 35, 32, 29, # B
    15, 12,  9, 16, 13, 10, 17, 14, 11, # F
)

HARD_26_STATES = [
    [0, 1, 0, 2, 0, 4, 0, 3, 0, 3, 0, 3, 2, 1, 4, 3, 5, 3, 4, 0, 4, 3, 2, 1, 4, 5, 4, 1, 0, 1, 4, 3, 2, 1, 5, 1, 2, 0, 2, 1, 4, 3, 2, 5, 2, 5, 3, 5, 2, 5, 4, 5, 1, 5],
]

def rot90_traceable(m, k=1, axes=(0, 1)):
    k %= 4
    return jax.lax.switch(k, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])


class RubiksCubeDeepCubeABenchmark(Benchmark):
    """Benchmark exposing the DeepCubeA 3x3 Rubik's Cube dataset."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        use_color_embedding: bool = True,
        states: Sequence[Any] | None = None
    ) -> None:
        super().__init__()
        self._dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        self._solve_config_cache = None
        self._use_color_embedding = use_color_embedding
        self._explicit_states = states

    def build_puzzle(self):
        from puxle.puzzles.rubikscube import RubiksCube

        return RubiksCube(size=3, initial_shuffle=100, color_embedding=self._use_color_embedding)

import puxle.benchmark._deepcubea as _dc

    def load_dataset(self) -> dict[str, Any]:
        if self._explicit_states is not None:
            return {"states": self._explicit_states, "solutions": None}

        fallback_dir = Path(__file__).resolve().parents[1] / DATA_RELATIVE_PATH.parent
        return _dc.load_deepcubea_dataset(
            self._dataset_path, DEFAULT_DATASET_NAME, "puxle.data.rubikscube", fallback_dir
        )

    def sample_ids(self) -> Iterable[Hashable]:
        return range(len(self.dataset["states"]))

    def get_sample(self, sample_id: Hashable) -> BenchmarkSample:
        index = int(sample_id)
        dataset = self.dataset
        
        state_data = dataset["states"][index]
        state = self._convert_state(state_data)
        
        solve_config = self._ensure_solve_config()
        
        optimal_action_sequence = None
        optimal_path_costs = None
        optimal_path = None

        if dataset.get("solutions") is not None:
            optimal_action_sequence, optimal_path_costs = self._convert_solution(dataset["solutions"][index])
            optimal_path = self._build_optimal_path(state, solve_config, optimal_action_sequence)

        return BenchmarkSample(
            state=state,
            solve_config=solve_config,
            optimal_action_sequence=optimal_action_sequence,
            optimal_path=optimal_path,
            optimal_path_costs=optimal_path_costs,
        )

    def _convert_deepcubea_to_puxle(self, faces: jnp.ndarray, size: int) -> jnp.ndarray:
        faces = jnp.asarray(faces, dtype=jnp.int32)

        total_tiles = 6 * size * size
        if faces.size != total_tiles:
            raise ValueError(
                f"Expected {total_tiles} tiles for a size-{size} cube, received {faces.size}."
            )

        pos_map = jnp.asarray(POS_MAP, dtype=jnp.int32)
        faces = jnp.zeros_like(faces).at[pos_map].set(faces)

        id_map = jnp.asarray(ID_MAP, dtype=jnp.int32)
        faces = jnp.take(id_map, faces)

        return faces

    def _ensure_solve_config(self):
        if self._solve_config_cache is None:
            self._solve_config_cache = self.puzzle.get_solve_config()
        return self._solve_config_cache

    def _convert_state(self, raw_state: Any):
        colors = getattr(raw_state, "colors", raw_state)
        faces = jnp.asarray(colors, dtype=jnp.uint8)
        puzzle : RubiksCube = self.puzzle
        faces = self._convert_deepcubea_to_puxle(faces, puzzle.size)
        faces = puzzle.convert_tile_to_color_embedding(faces)
        return puzzle.State.from_unpacked(faces=faces.reshape(6, -1))

    def _convert_solution(
        self,
        moves: Sequence[Sequence[Any]],
    ) -> tuple[tuple[str, ...], float]:
        action_sequence: list[str] = []
        for face, direction in moves:
            if direction not in (-1, 1):
                raise ValueError(f"Unsupported rotation direction {direction} for move {face}.")
            notation = face if direction == 1 else f"{face}'"
            action_sequence.append(notation)
        optimal_action_sequence = tuple(action_sequence)
        return optimal_action_sequence, float(len(optimal_action_sequence))


class RubiksCubeDeepCubeAHardBenchmark(RubiksCubeDeepCubeABenchmark):
    """Benchmark exposing the DeepCubeA 3x3 Rubik's Cube hard cases (26 moves)."""
    
    def __init__(self, dataset_path: str | Path | None = None, use_color_embedding: bool = True) -> None:
        super().__init__(dataset_path, use_color_embedding, states=HARD_26_STATES)
        
    def _convert_state(self, raw_state: Any):
        # States are already in Puxle format
        faces = jnp.asarray(raw_state, dtype=jnp.uint8)
        puzzle = self.puzzle
        return puzzle.State.from_unpacked(faces=faces.reshape(6, -1))
