#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


from puxle.benchmark.lightsout_deepcubea import LightsOutDeepCubeABenchmark


def _summarize_dataset(benchmark: LightsOutDeepCubeABenchmark) -> None:
    dataset = benchmark.dataset
    states = dataset.get("states", [])
    print("Dataset summary")
    print("---------------")
    print(f"States       : {len(states)}")
    print(f"Board size   : {benchmark.puzzle.size} x {benchmark.puzzle.size}")
    print()


def _preview_samples(
    benchmark: LightsOutDeepCubeABenchmark,
    sample_ids: Iterable[int],
) -> None:
    puzzle = benchmark.puzzle
    for sample_id in sample_ids:
        sample = benchmark.get_sample(sample_id)
        print(f"Sample {sample_id}")
        print(f"  puzzle size: {puzzle.size}x{puzzle.size}")
        print("  board:")
        print(sample.state)
        solved = puzzle.is_solved(sample.solve_config, sample.state)
        print(f"  already solved? {'yes' if solved else 'no'}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick sanity-check script for LightsOut DeepCubeA benchmark integration."
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
        "--offset",
        type=int,
        default=0,
        help="Starting index offset when previewing samples (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark = LightsOutDeepCubeABenchmark(dataset_path=args.dataset)
    _summarize_dataset(benchmark)

    total_states = len(benchmark.dataset.get("states", []))
    offset = max(0, args.offset)
    limit = max(0, args.limit)
    if limit == 0:
        print("No samples requested (--limit 0). Exiting.")
        return
    if offset >= total_states:
        print(f"Offset {offset} exceeds dataset bounds ({total_states} states).")
        return

    sample_range = range(offset, min(offset + limit, total_states))
    _preview_samples(benchmark, sample_range)


if __name__ == "__main__":
    main()
