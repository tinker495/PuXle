from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Mapping

import jax.numpy as jnp
import numpy as np

from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
from puxle.puzzles.rubikscube import RubiksCube

DATA_RELATIVE_PATH = Path("data") / "santa-2023" / "puzzles.csv"


class RubiksCubeSantaPreset(Enum):
    CUBE_2 = "cube_2/2/2"
    CUBE_3 = "cube_3/3/3"
    CUBE_4 = "cube_4/4/4"
    CUBE_5 = "cube_5/5/5"
    CUBE_6 = "cube_6/6/6"
    CUBE_7 = "cube_7/7/7"
    CUBE_8 = "cube_8/8/8"
    CUBE_9 = "cube_9/9/9"
    CUBE_10 = "cube_10/10/10"
    CUBE_19 = "cube_19/19/19"
    CUBE_33 = "cube_33/33/33"

    def __init__(self, puzzle_type: str):
        self.puzzle_type = puzzle_type


class RubiksCubeSantaBenchmark(Benchmark):
    """
    Benchmark exposing the Santa 2023 Rubik's Cube puzzles.

    The dataset contains puzzles of various sizes (2x2x2 to 33x33x33).
    Each sample defines an initial state and a target solution state.

    This base class only includes samples where the target state matches
    the standard solved state (A;A...B;B...).

    References:
        R. Holbrook, W. Reade, and A. Howard, Santa 2023 the polytope
        permutation puzzle, https://kaggle.com/competitions/santa-2023 (2023), kaggle.
    """

    def __init__(
        self,
        preset: RubiksCubeSantaPreset | None = RubiksCubeSantaPreset.CUBE_3,
        dataset_path: str | Path | None = None,
        puzzle_type: str | None = None,
    ) -> None:
        super().__init__()

        # Use preset if provided, otherwise use puzzle_type string directly
        if puzzle_type is not None:
            self.puzzle_type = puzzle_type
        else:
            self.puzzle_type = (
                preset.puzzle_type
                if preset
                else RubiksCubeSantaPreset.CUBE_3.puzzle_type
            )

        self._dataset_path = (
            Path(dataset_path).expanduser().resolve() if dataset_path else None
        )
        self._solve_config_cache = None

        # Extract size from puzzle_type
        if not self.puzzle_type.startswith("cube_"):
            raise ValueError(
                f"Invalid puzzle type '{self.puzzle_type}'. Must start with 'cube_'."
            )
        try:
            dims = self.puzzle_type.split("_")[1].split("/")
            self.size = int(dims[0])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not parse size from '{self.puzzle_type}'.") from e

    def build_puzzle(self) -> RubiksCube:
        # We enforce color embedding because Santa dataset uses colors (A, B...)
        # which we map to 0..5, rather than unique tile IDs.
        return RubiksCube(
            size=self.size, initial_shuffle=0, color_embedding=True, metric="UQTM"
        )

    def load_dataset(self) -> Dict[str, Any]:
        if self._dataset_path is not None:
            path = self._dataset_path
        else:
            # Try to find in default location
            path = Path(__file__).resolve().parents[2] / "puxle" / DATA_RELATIVE_PATH
            if not path.exists():
                # Fallback to package resource or other locations if needed
                pass

        if not path.is_file():
            raise FileNotFoundError(f"Santa 2023 dataset not found at {path}")

        with path.open(newline="") as f:
            rows = [
                row
                for row in csv.DictReader(f)
                if row["puzzle_type"] == self.puzzle_type
            ]

        if not rows:
            raise ValueError(
                f"No puzzles found for type '{self.puzzle_type}' in {path}"
            )

        # Pre-process samples
        samples = []
        for row in rows:
            parsed = self._parse_row(row)
            if parsed is not None:
                samples.append(parsed)

        return {"samples": samples}

    def _is_standard_solution(self, solution_mapped: np.ndarray) -> bool:
        """Check if the solution state follows the standard A;A...;B;B... pattern.

        Colours are mapped to 0..5 by sorted unique value in ``_parse_and_map_row``,
        so a standard (solid-face) solution is exactly a sorted array.
        """
        return np.array_equal(solution_mapped, np.sort(solution_mapped))

    def _parse_and_map_row(self, row: Mapping[str, str]) -> Dict[str, Any] | None:
        """
        Parse and map a row without applying the is_standard filter.
        Returns a dict with parsed data including is_standard flag, or None if parsing fails.
        """
        initial_str = row["initial_state"]
        solution_str = row["solution_state"]

        # Filter out puzzles that use N0, N1... notation (permutation puzzles)
        if initial_str.startswith("N") or solution_str.startswith("N"):
            return None

        wildcards = int(row.get("num_wildcards") or 0)

        # Filter out puzzles with wildcards
        if wildcards > 0:
            return None

        initial_raw = np.array(initial_str.split(";"))
        solution_raw = np.array(solution_str.split(";"))

        unique_colors = sorted(np.unique(solution_raw))
        mapping = {color: i for i, color in enumerate(unique_colors)}

        initial_mapped = np.vectorize(mapping.get)(initial_raw).astype(np.uint8)
        solution_mapped = np.vectorize(mapping.get)(solution_raw).astype(np.uint8)

        is_standard = self._is_standard_solution(solution_mapped)

        return {
            "id": int(row["id"]),
            "initial": initial_mapped,
            "target": solution_mapped,
            "wildcards": wildcards,
            "is_standard": is_standard,
        }

    def _parse_row(self, row: Mapping[str, str]) -> Dict[str, Any] | None:
        """
        Parse a row and return sample data. Filters to only STANDARD solution puzzles.
        """
        parsed = self._parse_and_map_row(row)
        if parsed is None:
            return None

        # This class (RubiksCubeSantaBenchmark) only returns STANDARD solution puzzles.
        if not parsed["is_standard"]:
            return None

        return {
            "id": parsed["id"],
            "initial": parsed["initial"],
            "target": parsed["target"],
            "wildcards": parsed["wildcards"],
        }

    def sample_ids(self) -> Iterable[Hashable]:
        return range(len(self.dataset["samples"]))

    def get_sample(self, sample_id: Hashable) -> BenchmarkSample:
        index = int(sample_id)
        sample_data = self.dataset["samples"][index]

        puzzle = self.puzzle

        initial_faces = jnp.asarray(sample_data["initial"], dtype=jnp.uint8)
        target_faces = jnp.asarray(sample_data["target"], dtype=jnp.uint8)

        initial_state = puzzle.State.from_unpacked(faces=initial_faces.reshape(6, -1))
        target_state = puzzle.State.from_unpacked(faces=target_faces.reshape(6, -1))

        solve_config = puzzle.SolveConfig(TargetState=target_state)

        return BenchmarkSample(
            state=initial_state,
            solve_config=solve_config,
            optimal_action_sequence=None,
            optimal_path=None,
            optimal_path_costs=None,
        )


class RubiksCubeSantaRandomBenchmark(RubiksCubeSantaBenchmark):
    """
    Benchmark for Santa 2023 Rubik's Cube puzzles with non-standard (random pattern) target states.

    References:
        R. Holbrook, W. Reade, and A. Howard, Santa 2023 the polytope
        permutation puzzle, https://kaggle.com/competitions/santa-2023 (2023), kaggle.
    """

    def _parse_row(self, row: Mapping[str, str]) -> Dict[str, Any] | None:
        """
        Parse a row and return sample data. Filters to only NON-STANDARD (random) solution puzzles.
        """
        parsed = self._parse_and_map_row(row)
        if parsed is None:
            return None

        # This class only returns NON-STANDARD (random) solution puzzles.
        if parsed["is_standard"]:
            return None

        return {
            "id": parsed["id"],
            "initial": parsed["initial"],
            "target": parsed["target"],
            "wildcards": parsed["wildcards"],
        }


def _santa_preset_class(
    name: str,
    base_class: type[RubiksCubeSantaBenchmark],
    preset: RubiksCubeSantaPreset,
) -> type[RubiksCubeSantaBenchmark]:
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        base_class.__init__(self, preset=preset, dataset_path=dataset_path)

    return type(
        name,
        (base_class,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "__qualname__": name,
            "__doc__": f"{base_class.__name__} for {preset.puzzle_type}.",
        },
    )


for _name, _base_class, _preset in (
    (
        "RubiksCubeSanta222Benchmark",
        RubiksCubeSantaBenchmark,
        RubiksCubeSantaPreset.CUBE_2,
    ),
    (
        "RubiksCubeSanta333Benchmark",
        RubiksCubeSantaBenchmark,
        RubiksCubeSantaPreset.CUBE_3,
    ),
    (
        "RubiksCubeSanta444Benchmark",
        RubiksCubeSantaBenchmark,
        RubiksCubeSantaPreset.CUBE_4,
    ),
    (
        "RubiksCubeSanta555Benchmark",
        RubiksCubeSantaBenchmark,
        RubiksCubeSantaPreset.CUBE_5,
    ),
    (
        "RubiksCubeSanta666Benchmark",
        RubiksCubeSantaBenchmark,
        RubiksCubeSantaPreset.CUBE_6,
    ),
    (
        "RubiksCubeSantaRandom222Benchmark",
        RubiksCubeSantaRandomBenchmark,
        RubiksCubeSantaPreset.CUBE_2,
    ),
    (
        "RubiksCubeSantaRandom333Benchmark",
        RubiksCubeSantaRandomBenchmark,
        RubiksCubeSantaPreset.CUBE_3,
    ),
    (
        "RubiksCubeSantaRandom444Benchmark",
        RubiksCubeSantaRandomBenchmark,
        RubiksCubeSantaPreset.CUBE_4,
    ),
    (
        "RubiksCubeSantaRandom555Benchmark",
        RubiksCubeSantaRandomBenchmark,
        RubiksCubeSantaPreset.CUBE_5,
    ),
    (
        "RubiksCubeSantaRandom666Benchmark",
        RubiksCubeSantaRandomBenchmark,
        RubiksCubeSantaPreset.CUBE_6,
    ),
):
    globals()[_name] = _santa_preset_class(_name, _base_class, _preset)

del _name, _base_class, _preset
