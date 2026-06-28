from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "Benchmark",
    "BenchmarkSample",
    "LightsOutDeepCubeABenchmark",
    "RubiksCubeDeepCubeABenchmark",
    "RubiksCubeDeepCubeAHardBenchmark",
    "RubiksCubeSantaBenchmark",
    "RubiksCubeSantaRandomBenchmark",
    "RubiksCubeSanta222Benchmark",
    "RubiksCubeSanta333Benchmark",
    "RubiksCubeSanta444Benchmark",
    "RubiksCubeSanta555Benchmark",
    "RubiksCubeSanta666Benchmark",
    "RubiksCubeSantaRandom222Benchmark",
    "RubiksCubeSantaRandom333Benchmark",
    "RubiksCubeSantaRandom444Benchmark",
    "RubiksCubeSantaRandom555Benchmark",
    "RubiksCubeSantaRandom666Benchmark",
    "SlidePuzzleDeepCubeABenchmark",
    "SlidePuzzleDeepCubeA15Benchmark",
    "SlidePuzzleDeepCubeA15HardBenchmark",
    "SlidePuzzleDeepCubeA24Benchmark",
    "SlidePuzzleDeepCubeA35Benchmark",
    "SlidePuzzleDeepCubeA48Benchmark",
]

_EXPORTS = {
    "Benchmark": (".benchmark", "Benchmark"),
    "BenchmarkSample": (".benchmark", "BenchmarkSample"),
    "LightsOutDeepCubeABenchmark": (
        ".lightsout_deepcubea",
        "LightsOutDeepCubeABenchmark",
    ),
    "RubiksCubeDeepCubeABenchmark": (
        ".rubikscube_deepcubea",
        "RubiksCubeDeepCubeABenchmark",
    ),
    "RubiksCubeDeepCubeAHardBenchmark": (
        ".rubikscube_deepcubea",
        "RubiksCubeDeepCubeAHardBenchmark",
    ),
    "RubiksCubeSantaBenchmark": (".rubikscube_santa", "RubiksCubeSantaBenchmark"),
    "RubiksCubeSantaRandomBenchmark": (
        ".rubikscube_santa",
        "RubiksCubeSantaRandomBenchmark",
    ),
    "RubiksCubeSanta222Benchmark": (".rubikscube_santa", "RubiksCubeSanta222Benchmark"),
    "RubiksCubeSanta333Benchmark": (".rubikscube_santa", "RubiksCubeSanta333Benchmark"),
    "RubiksCubeSanta444Benchmark": (".rubikscube_santa", "RubiksCubeSanta444Benchmark"),
    "RubiksCubeSanta555Benchmark": (".rubikscube_santa", "RubiksCubeSanta555Benchmark"),
    "RubiksCubeSanta666Benchmark": (".rubikscube_santa", "RubiksCubeSanta666Benchmark"),
    "RubiksCubeSantaRandom222Benchmark": (
        ".rubikscube_santa",
        "RubiksCubeSantaRandom222Benchmark",
    ),
    "RubiksCubeSantaRandom333Benchmark": (
        ".rubikscube_santa",
        "RubiksCubeSantaRandom333Benchmark",
    ),
    "RubiksCubeSantaRandom444Benchmark": (
        ".rubikscube_santa",
        "RubiksCubeSantaRandom444Benchmark",
    ),
    "RubiksCubeSantaRandom555Benchmark": (
        ".rubikscube_santa",
        "RubiksCubeSantaRandom555Benchmark",
    ),
    "RubiksCubeSantaRandom666Benchmark": (
        ".rubikscube_santa",
        "RubiksCubeSantaRandom666Benchmark",
    ),
    "SlidePuzzleDeepCubeABenchmark": (
        ".slidepuzzle_deepcubea",
        "SlidePuzzleDeepCubeABenchmark",
    ),
    "SlidePuzzleDeepCubeA15Benchmark": (
        ".slidepuzzle_deepcubea",
        "SlidePuzzleDeepCubeA15Benchmark",
    ),
    "SlidePuzzleDeepCubeA15HardBenchmark": (
        ".slidepuzzle_deepcubea",
        "SlidePuzzleDeepCubeA15HardBenchmark",
    ),
    "SlidePuzzleDeepCubeA24Benchmark": (
        ".slidepuzzle_deepcubea",
        "SlidePuzzleDeepCubeA24Benchmark",
    ),
    "SlidePuzzleDeepCubeA35Benchmark": (
        ".slidepuzzle_deepcubea",
        "SlidePuzzleDeepCubeA35Benchmark",
    ),
    "SlidePuzzleDeepCubeA48Benchmark": (
        ".slidepuzzle_deepcubea",
        "SlidePuzzleDeepCubeA48Benchmark",
    ),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
