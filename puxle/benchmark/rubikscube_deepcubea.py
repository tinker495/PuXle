from __future__ import annotations

import pickle
from importlib.resources import files
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

import jax
import jax.numpy as jnp
from functools import partial
from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
from puxle.puzzles.rubikscube import RubiksCube

DEFAULT_DATASET_NAME = "size3-deepcubeA.pkl"
DATA_RELATIVE_PATH = Path("data") / "rubikscube" / DEFAULT_DATASET_NAME

MAPPING_AXIS_TO_FACE = (0, 5, 2, 3, 4, 1)
MAPPING_SPIN_TO_FACE = (0, 1, 1, 1, 1, 0)
MAPPING_ACTION = {
    "U": "U",
    "D": "D",
    "L": "L",
    "R": "R",
    "F": "F",
    "B": "B",
}

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

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__()
        self._dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        self._solve_config_cache = None
        self._notation_to_action: dict[str, int] | None = None

    def build_puzzle(self):
        from puxle.puzzles.rubikscube import RubiksCube

        return RubiksCube(size=3, initial_shuffle=100)

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

        return BenchmarkSample(
            state=state,
            solve_config=solve_config,
            optimal_action_sequence=optimal_action_sequence,
            optimal_path=None,
            optimal_path_costs=optimal_path_costs,
        )

    def _convert_deepcubea_to_puxle(self, faces: jnp.ndarray, size: int) -> jnp.ndarray:
        faces = jnp.reshape(faces, (6, size * size))
        new_faces = jnp.zeros_like(faces)
        for frm, to in enumerate(MAPPING_AXIS_TO_FACE):
            frm_n_start = frm * size * size
            frm_n_end = (frm + 1) * size * size
            inside_mask = jnp.logical_and(
                jnp.greater_equal(faces, frm_n_start),
                jnp.less(faces, frm_n_end),
            )
            additional_idxs = (to - frm) * size * size
            new_faces = new_faces.at[inside_mask].set(additional_idxs + faces[inside_mask])
        for i in range(6):
            rotated_face = rot90_traceable(
                jnp.reshape(new_faces[i, :], (size, size)),
                MAPPING_SPIN_TO_FACE[i],
            )
            new_faces = new_faces.at[i, :].set(jnp.reshape(rotated_face, (size * size,)))
        new_faces = jnp.reshape(new_faces, (6 * size * size, ))
        return new_faces

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
            face_notation = MAPPING_ACTION[str(face).upper()]
            notation = face_notation if direction == 1 else f"{face_notation}'"
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
