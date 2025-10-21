from __future__ import annotations

import pickle
from importlib.resources import files
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence

import jax.numpy as jnp

from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
from puxle.utils.util import to_uint8

DEFAULT_DATASET_NAME = "size3-deepcubeA.pkl"
DATA_RELATIVE_PATH = Path("data") / "rubikscube" / DEFAULT_DATASET_NAME


class _DeepCubeUnpickler(pickle.Unpickler):
    """Unpickler that recreates missing DeepCube classes on the fly."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "environments.cube3":
            return globals().setdefault(name, type(name, (), {}))
        return super().find_class(module, name)


class RubiksCubeDeepCubeABenchmark(Benchmark):
    """Benchmark exposing the DeepCubeA 3x3 Rubik's Cube dataset."""

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        super().__init__()
        self._dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        self._solve_config_cache = None
        self._notation_to_action: dict[str, int] | None = None

    def build_puzzle(self):
        from puxle.puzzles.rubikscube import RubiksCube

        return RubiksCube(size=3)

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
        raw_state = dataset["states"][index]

        state = self._convert_state(raw_state)
        solve_config = self._ensure_solve_config()
        optimal_path = self._convert_solution(dataset["solutions"][index])

        return BenchmarkSample(state=state, solve_config=solve_config, optimal_path=optimal_path)

    def _ensure_solve_config(self):
        if self._solve_config_cache is None:
            self._solve_config_cache = self.puzzle.get_solve_config()
        return self._solve_config_cache

    def _convert_state(self, raw_state: Any):
        colors = getattr(raw_state, "colors", raw_state)
        colors_array = jnp.asarray(colors, dtype=jnp.uint8)
        puzzle = self.puzzle
        tiles = colors_array.reshape(6, puzzle.size * puzzle.size)
        color_faces = puzzle.convert_tile_to_color_embedding(tiles)
        packed_faces = to_uint8(color_faces, puzzle._active_bits)
        return puzzle.State(faces=packed_faces)

    def _convert_solution(self, moves: Sequence[Sequence[Any]]):
        notation_lookup = self._build_action_lookup()
        actions: list[int] = []
        for face, direction in moves:
            if direction not in (-1, 1):
                raise ValueError(f"Unsupported rotation direction {direction} for move {face}.")
            face_notation = str(face).upper()
            notation = face_notation if direction == 1 else f"{face_notation}'"
            try:
                actions.append(notation_lookup[notation])
            except KeyError as exc:
                raise KeyError(f"Unknown move notation '{notation}' in solution path") from exc
        return tuple(actions)

    def _build_action_lookup(self) -> dict[str, int]:
        if self._notation_to_action is None:
            self._notation_to_action = {
                self.puzzle.action_to_string(action): action
                for action in range(self.puzzle.action_size)
            }
        return self._notation_to_action
