from __future__ import annotations

from typing import Iterable, Sequence

from .benchmark import Benchmark, BenchmarkSample
from .lightsout_deepcubea import LightsOutDeepCubeABenchmark
from .rubikscube_deepcubea import RubiksCubeDeepCubeABenchmark, RubiksCubeDeepCubeAHardBenchmark
from .rubikscube_santa import (
    RubiksCubeSanta222Benchmark,
    RubiksCubeSanta333Benchmark,
    RubiksCubeSanta444Benchmark,
    RubiksCubeSanta555Benchmark,
    RubiksCubeSanta666Benchmark,
    RubiksCubeSantaBenchmark,
    RubiksCubeSantaRandom222Benchmark,
    RubiksCubeSantaRandom333Benchmark,
    RubiksCubeSantaRandom444Benchmark,
    RubiksCubeSantaRandom555Benchmark,
    RubiksCubeSantaRandom666Benchmark,
    RubiksCubeSantaRandomBenchmark,
)
from .slidepuzzle_deepcubea import (
    SlidePuzzleDeepCubeA15Benchmark,
    SlidePuzzleDeepCubeA15HardBenchmark,
    SlidePuzzleDeepCubeA24Benchmark,
    SlidePuzzleDeepCubeA35Benchmark,
    SlidePuzzleDeepCubeA48Benchmark,
    SlidePuzzleDeepCubeABenchmark,
)

BenchmarkSequence = Sequence["Benchmark"]
SampleIterable = Iterable["BenchmarkSample"]

# All benchmark implementations
__all__ = [
    "BenchmarkSequence",
    "SampleIterable",
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
