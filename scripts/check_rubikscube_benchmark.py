#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import sys
from typing import Iterable, Sequence
from puxle.core.puzzle_state import PuzzleState

import numpy as np
import xtructure.numpy as xnp

from puxle.benchmark.rubikscube_deepcubea import RubiksCubeDeepCubeABenchmark


def _reconstruct_optimal_path(
    benchmark: RubiksCubeDeepCubeABenchmark,
    sample,
) -> tuple[PuzzleState, ...]:
    if sample.optimal_path:
        return tuple(sample.optimal_path)

    action_sequence = sample.optimal_action_sequence
    if not action_sequence:
        return tuple()

    action_lookup = benchmark._build_action_lookup()
    puzzle = benchmark.puzzle
    solve_config = sample.solve_config
    current_state = sample.state
    path: list[PuzzleState] = []

    for step, notation in enumerate(action_sequence, start=1):
        try:
            action_idx = action_lookup[notation]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown action notation '{notation}' at step {step}") from exc

        neighbours, _ = puzzle.get_neighbours(solve_config, current_state, filled=True)
        next_state = neighbours[action_idx]
        path.append(next_state)
        current_state = next_state

    return tuple(path)

def _stack_states(states: Sequence[PuzzleState]) -> str:
    return xnp.concatenate(states, axis=0)


def _validate_solution(
    benchmark: RubiksCubeDeepCubeABenchmark,
    sample,
    *,
    optimal_path: Sequence[PuzzleState] | None = None,
) -> bool:
    puzzle = benchmark.puzzle
    solve_config = sample.solve_config

    if optimal_path is None:
        optimal_path = _reconstruct_optimal_path(benchmark, sample)

    final_state = optimal_path[-1] if optimal_path else sample.state
    return puzzle.is_solved(solve_config, final_state)


def _summarize_dataset(benchmark: RubiksCubeDeepCubeABenchmark) -> None:
    dataset = benchmark.dataset
    times = np.asarray(dataset.get("times", []), dtype=float)
    nodes = np.asarray(dataset.get("num_nodes_generated", []), dtype=float)
    lengths = np.array([len(sol) for sol in dataset.get("solutions", [])], dtype=int)

    print("Dataset summary")
    print("---------------")
    print(f"States: {len(dataset.get('states', []))}")
    if times.size:
        print(f"Times : min={times.min():.4f}s  max={times.max():.4f}s  mean={times.mean():.4f}s")
    if nodes.size:
        print(f"Nodes : min={int(nodes.min())}  max={int(nodes.max())}  mean={nodes.mean():.1f}")
    if lengths.size:
        print(
            "Moves : min="
            f"{lengths.min()}  max={lengths.max()}  mean={lengths.mean():.1f}  median={statistics.median(lengths)}"
        )
    print()


def _preview_samples(
    benchmark: RubiksCubeDeepCubeABenchmark,
    sample_ids: Iterable[int],
    validate: bool,
) -> None:
    for sample_id in sample_ids:
        sample = benchmark.get_sample(sample_id)
        state = sample.state
        solve_config = sample.solve_config
        optimal_action_sequence = sample.optimal_action_sequence

        try:
            optimal_path = _reconstruct_optimal_path(benchmark, sample)
        except KeyError as exc:
            print(f"  optimal path reconstruction error: {exc}")
            optimal_path = tuple()
        print(f"Sample {sample_id}")
        print(f"  state faces shape: {state.faces.shape} dtype={state.faces.dtype}")
        print(state)
        target = solve_config.TargetState
        print(f"  target faces shape: {target.faces.shape} dtype={target.faces.dtype}")
        print(target)
        if optimal_path:
            print(f"  optimal path length: {len(optimal_path)}")
            print(f"  optimal path states: \n{_stack_states(optimal_path)}")
        else:
            print("  optimal path length: 0 (no path provided)")
            print("  optimal path states: <not available>")
        print(f"  optimal action sequence: {optimal_action_sequence}")
        solved = _validate_solution(benchmark, sample, optimal_path=optimal_path)
        if not solved:
            msg = f"Optimal path for sample {sample_id} does not reach the target state."
            print("  validation: FAILED (check move convention mapping)")
            if validate:
                raise RuntimeError(msg)
        elif validate:
            print("  validation: solved")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick sanity-check script for Rubik's Cube DeepCubeA benchmark integration."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional explicit path to the DeepCubeA pickle file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of samples to preview (default: 3).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Verify that the recorded optimal path solves each sampled state.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index offset when previewing samples (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark = RubiksCubeDeepCubeABenchmark(dataset_path=args.dataset)
    _summarize_dataset(benchmark)

    total_states = len(benchmark.dataset["states"])
    offset = max(0, args.offset)
    limit = max(0, args.limit)
    sample_range = range(offset, min(offset + limit, total_states))
    if limit == 0:
        print("No samples requested (--limit 0). Exiting.")
        return
    if offset >= total_states:
        print(f"Offset {offset} exceeds dataset bounds ({total_states} states).")
        return

    _preview_samples(benchmark, sample_range, validate=args.validate)


if __name__ == "__main__":
    main()
