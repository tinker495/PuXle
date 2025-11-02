from __future__ import annotations

import pickle
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

import jax
import jax.numpy as jnp
from puxle.core.puzzle_state import PuzzleState
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

class _DeepCubeUnpickler(pickle.Unpickler):
    """Unpickler that recreates missing DeepCube classes on the fly."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "environments.cube3":
            return globals().setdefault(name, type(name, (), {}))
        return super().find_class(module, name)

def rot90_traceable(m, k=1, axes=(0, 1)):
    k %= 4
    return jax.lax.switch(k, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])


class RubiksCubeDeepCubeABenchmark(Benchmark):
    """Benchmark exposing the DeepCubeA 3x3 Rubik's Cube dataset."""

    def __init__(self, dataset_path: str | Path | None = None, use_color_embedding: bool = True) -> None:
        super().__init__()
        self._dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        self._solve_config_cache = None
        self._notation_to_action: dict[str, int] | None = None
        self._use_color_embedding = use_color_embedding

    def build_puzzle(self):
        from puxle.puzzles.rubikscube import RubiksCube

        return RubiksCube(size=3, initial_shuffle=100, color_embedding=self._use_color_embedding)

    def load_dataset(self) -> dict[str, Any]:
        if self._dataset_path is not None:
            if not self._dataset_path.is_file():
                raise FileNotFoundError(
                    f"DeepCubeA dataset not found at {self._dataset_path}"
                )
            with self._dataset_path.open("rb") as handle:
                return _DeepCubeUnpickler(handle).load()

        try:
            resource = files("puxle.data.rubikscube") / DEFAULT_DATASET_NAME
            with resource.open("rb") as handle:
                return _DeepCubeUnpickler(handle).load()
        except (ModuleNotFoundError, FileNotFoundError):
            pass

        fallback = Path(__file__).resolve().parents[1] / DATA_RELATIVE_PATH
        if not fallback.is_file():
            raise FileNotFoundError(
                f"Unable to locate {DEFAULT_DATASET_NAME} under package resources or at {fallback}"
            )
        with fallback.open("rb") as handle:
            return _DeepCubeUnpickler(handle).load()

    def sample_ids(self) -> Iterable[Hashable]:
        return range(len(self.dataset["states"]))

    def get_sample(self, sample_id: Hashable) -> BenchmarkSample:
        index = int(sample_id)
        dataset = self.dataset
        state = self._convert_state(dataset["states"][index])
        solve_config = self._ensure_solve_config()
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
        return puzzle.State(faces=faces).packed

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

    def _build_action_lookup(self) -> dict[str, int]:
        if self._notation_to_action is None:
            self._notation_to_action = {
                self.puzzle.action_to_string(action): action
                for action in range(self.puzzle.action_size)
            }
        return self._notation_to_action

    def _build_optimal_path(
        self,
        initial_state: PuzzleState,
        solve_config: RubiksCube.SolveConfig,
        action_sequence: Sequence[str],
    ) -> tuple[PuzzleState, ...]:
        if not action_sequence:
            return tuple()

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
