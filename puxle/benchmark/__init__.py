from .benchmark import Benchmark, BenchmarkSample
from .lightsout_deepcubea import LightsOutDeepCubeABenchmark
from .rubikscube_deepcubea import RubiksCubeDeepCubeABenchmark, RubiksCubeDeepCubeAHardBenchmark
from .rubikscube_santa import (
    RubiksCubeSantaBenchmark,
    RubiksCubeSantaRandomBenchmark,
    RubiksCubeSanta222Benchmark,
    RubiksCubeSanta333Benchmark,
    RubiksCubeSanta444Benchmark,
    RubiksCubeSanta555Benchmark,
    RubiksCubeSanta666Benchmark,
    RubiksCubeSantaRandom222Benchmark,
    RubiksCubeSantaRandom333Benchmark,
    RubiksCubeSantaRandom444Benchmark,
    RubiksCubeSantaRandom555Benchmark,
    RubiksCubeSantaRandom666Benchmark,
)
from .slidepuzzle_deepcubea import SlidePuzzleDeepCubeABenchmark, SlidePuzzleDeepCubeA15Benchmark, SlidePuzzleDeepCubeA15HardBenchmark, SlidePuzzleDeepCubeA24Benchmark, SlidePuzzleDeepCubeA35Benchmark, SlidePuzzleDeepCubeA48Benchmark

# All benchmark implementations
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
