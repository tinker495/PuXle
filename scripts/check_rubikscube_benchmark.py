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

def _stack_states(states: Sequence[PuzzleState]) -> str:
    return xnp.concatenate(states, axis=0)


def _validate_solution(benchmark: RubiksCubeDeepCubeABenchmark, sample_id: int) -> bool:
    sample = benchmark.get_sample(sample_id)
    puzzle = benchmark.puzzle
    solve_config = sample.solve_config
    state = sample.state
    if not sample.optimal_path:
        return puzzle.is_solved(solve_config, state)
    for next_state in sample.optimal_path:
        state = next_state
    return puzzle.is_solved(solve_config, state)


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
        optimal_path = sample.optimal_path
        print(f"Sample {sample_id}")
        print(f"  state faces shape: {state.faces.shape} dtype={state.faces.dtype}")
        print(state)
        target = solve_config.TargetState
        print(f"  target faces shape: {target.faces.shape} dtype={target.faces.dtype}")
        print(target)
        print(f"  optimal path length: {len(optimal_path)}")
        print(f"  optimal path states: \n{_stack_states(optimal_path)}")
        if validate:
            solved = _validate_solution(benchmark, sample_id)
            if solved:
                print("  validation: solved")
            else:
                print("  validation: FAILED (check move convention mapping)")
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
