#!/usr/bin/env python3
"""Compare LightsOut neighbour dynamics between PuXle and DeepCubeA conventions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from puxle.benchmark.lightsout_deepcubea import LightsOutDeepCubeABenchmark


def _apply_deepcube_move(board: np.ndarray, action: int, size: int) -> np.ndarray:
    """Emulate the LightsOut move convention used by DeepCubeA."""

    next_board = board.copy()
    x, y = divmod(action, size)
    deltas = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            idx = nx * size + ny
            next_board[idx] = ~next_board[idx]
    return next_board


def _compare_sample(benchmark: LightsOutDeepCubeABenchmark, sample_id: int) -> bool:
    sample = benchmark.get_sample(sample_id)
    puzzle = benchmark.puzzle
    solve_config = sample.solve_config
    size = puzzle.size

    mismatched = False
    initial_board = np.array(sample.state.unpacked.board, dtype=bool)

    for action in range(puzzle.action_size):
        next_state, _ = puzzle.get_actions(
            solve_config, sample.state, jnp.asarray(action), filled=True
        )
        puzzle_board = np.array(next_state.unpacked.board, dtype=bool)
        deepcube_board = _apply_deepcube_move(initial_board, action, size)

        if not np.array_equal(puzzle_board, deepcube_board):
            mismatched = True
            print(f"Mismatch detected for sample {sample_id}, action {action}")
            print("PuXle board:")
            print(puzzle.get_string_parser()(next_state))
            print("DeepCubeA board:")
            deepcube_state = puzzle.State(board=jnp.asarray(deepcube_board, dtype=jnp.bool_)).packed
            print(puzzle.get_string_parser()(deepcube_state))
            print()
    return not mismatched


def _summary(matched: bool, sample_id: int):
    status = "OK" if matched else "FAILED"
    print(f"Sample {sample_id}: {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that PuXle LightsOut dynamics match DeepCubeA conventions."
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
        help="Number of samples to compare (default: 3).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting index offset when comparing samples (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = LightsOutDeepCubeABenchmark(dataset_path=args.dataset)
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
    for sample_id in sample_range:
        matched = _compare_sample(benchmark, sample_id)
        _summary(matched, sample_id)


if __name__ == "__main__":
    main()
