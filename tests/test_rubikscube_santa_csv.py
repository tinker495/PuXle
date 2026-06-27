import csv
import importlib.abc
import sys

STANDARD_SOLUTION = ";".join(color for color in "ABCDEF" for _ in range(4))
STANDARD_INITIAL = "D;E;D;A;E;B;A;B;C;A;C;A;D;C;D;F;F;F;E;E;B;F;B;C"
RANDOM_TARGET = "A;B;A;A;B;A;B;B;C;C;C;C;D;D;D;D;E;E;E;E;F;F;F;F"


class _BlockPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] == "pandas":
            raise ImportError(f"blocked pandas import: {fullname}")
        return None


def _write_santa_csv(path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "puzzle_type",
                "solution_state",
                "initial_state",
                "num_wildcards",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "10",
                    "puzzle_type": "cube_2/2/2",
                    "solution_state": STANDARD_SOLUTION,
                    "initial_state": STANDARD_INITIAL,
                    "num_wildcards": "0",
                },
                {
                    "id": "11",
                    "puzzle_type": "cube_2/2/2",
                    "solution_state": RANDOM_TARGET,
                    "initial_state": STANDARD_INITIAL,
                    "num_wildcards": "0",
                },
                {
                    "id": "12",
                    "puzzle_type": "cube_3/3/3",
                    "solution_state": STANDARD_SOLUTION,
                    "initial_state": STANDARD_INITIAL,
                    "num_wildcards": "0",
                },
            ]
        )


def test_santa_csv_reader_uses_stdlib_without_pandas(monkeypatch, tmp_path):
    for module_name in list(sys.modules):
        if module_name == "pandas" or module_name.startswith("pandas."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        if module_name == "puxle.benchmark.rubikscube_santa":
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    csv_path = tmp_path / "puzzles.csv"
    _write_santa_csv(csv_path)

    sys.meta_path.insert(0, _BlockPandas())
    try:
        from puxle.benchmark.rubikscube_santa import (
            RubiksCubeSantaBenchmark,
            RubiksCubeSantaPreset,
            RubiksCubeSantaRandomBenchmark,
        )

        standard = RubiksCubeSantaBenchmark(
            preset=RubiksCubeSantaPreset.CUBE_2,
            dataset_path=csv_path,
        ).load_dataset()
        random = RubiksCubeSantaRandomBenchmark(
            preset=RubiksCubeSantaPreset.CUBE_2,
            dataset_path=csv_path,
        ).load_dataset()
    finally:
        sys.meta_path = [
            finder for finder in sys.meta_path if not isinstance(finder, _BlockPandas)
        ]

    assert [sample["id"] for sample in standard["samples"]] == [10]
    assert [sample["id"] for sample in random["samples"]] == [11]
    assert standard["samples"][0]["wildcards"] == 0
