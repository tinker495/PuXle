from __future__ import annotations

import inspect
from typing import Iterable, Sequence, Type

import pytest

from puxle.benchmark import Benchmark, BenchmarkSample
from puxle.core.puzzle_state import PuzzleState


def discover_benchmark_classes() -> list[Type[Benchmark]]:
    """Return all benchmark implementations exported by puxle.benchmark."""
    import puxle.benchmark as benchmark_module

    benchmark_classes: list[Type[Benchmark]] = []
    for attr_name in dir(benchmark_module):
        attr = getattr(benchmark_module, attr_name)
        if inspect.isclass(attr) and issubclass(attr, Benchmark) and attr is not Benchmark:
            benchmark_classes.append(attr)
    return benchmark_classes


def _reconstruct_optimal_path(
    benchmark: Benchmark,
    sample: BenchmarkSample,
) -> tuple[PuzzleState, ...]:
    if sample.optimal_path:
        return tuple(sample.optimal_path)

    action_sequence = sample.optimal_action_sequence
    if not action_sequence:
        return tuple()

    build_fn = getattr(benchmark, "_build_optimal_path", None)
    if callable(build_fn):
        return build_fn(sample.state, sample.solve_config, action_sequence)

    puzzle = benchmark.puzzle
    action_lookup = {
        puzzle.action_to_string(action): action for action in range(puzzle.action_size)
    }
    current_state = sample.state
    path: list[PuzzleState] = []

    for step, notation in enumerate(action_sequence, start=1):
        try:
            action_idx = action_lookup[notation]
        except KeyError as exc:
            raise KeyError(f"Unknown action notation '{notation}' at step {step}") from exc

        neighbours, _ = puzzle.get_neighbours(sample.solve_config, current_state, filled=True)
        next_state = neighbours[action_idx]
        path.append(next_state)
        current_state = next_state

    return tuple(path)


def _sample_ids(benchmark: Benchmark, limit: int = 3) -> Sequence[int]:
    ids = list(benchmark.sample_ids())
    if not ids:
        raise AssertionError(f"{benchmark.__class__.__name__} did not return any sample ids.")
    return ids[: min(limit, len(ids))]


def _assert_sample_valid(benchmark: Benchmark, sample_id: int) -> None:
    sample = benchmark.get_sample(sample_id)
    puzzle = benchmark.puzzle

    assert isinstance(sample.state, puzzle.State)
    assert sample.solve_config is not None

    if sample.optimal_action_sequence:
        path = _reconstruct_optimal_path(benchmark, sample)
        final_state = path[-1] if path else sample.state
        assert puzzle.is_solved(sample.solve_config, final_state), (
            f"{benchmark.__class__.__name__} sample {sample_id} did not reach solved state."
        )
    else:
        assert not sample.optimal_path, (
            f"{benchmark.__class__.__name__} provided optimal_path without action sequence."
        )


@pytest.mark.parametrize("benchmark_cls", discover_benchmark_classes())
def test_benchmarks_produce_sane_samples(benchmark_cls: Type[Benchmark]) -> None:
    benchmark = benchmark_cls()
    ids = _sample_ids(benchmark)
    for sample_id in ids:
        _assert_sample_valid(benchmark, sample_id)

