import math
from functools import partial

import chex
import jax
import jax.numpy as jnp
from xtructure import numpy as xnp

from puxle.core.puzzle_base import Puzzle
from puxle.core.trajectory import PuzzleTrajectory


def trajectory_to_transition_dataset(
    trajectory: PuzzleTrajectory, limit: int
) -> tuple[chex.ArrayTree, chex.Array, chex.ArrayTree]:
    return (
        trajectory.states[:-1].flatten()[:limit],
        trajectory.actions.reshape(-1)[:limit],
        trajectory.states[1:].flatten()[:limit],
    )


def trajectory_to_eval_trajectory(
    trajectory: PuzzleTrajectory,
) -> tuple[chex.ArrayTree, chex.Array]:
    return trajectory.states.flatten(), trajectory.actions.reshape(-1)


def get_world_model_dataset_builder(
    puzzle: Puzzle,
    dataset_size: int,
    shuffle_parallel: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
):
    if min(dataset_size, shuffle_parallel, shuffle_length, dataset_minibatch_size) <= 0:
        raise ValueError("dataset sizes and shuffle dimensions must be positive")
    samples_per_step = min(dataset_minibatch_size, shuffle_parallel * shuffle_length)
    steps = math.ceil(dataset_size / samples_per_step)
    create_path = partial(
        create_shuffled_path,
        puzzle,
        shuffle_length,
        shuffle_parallel,
        dataset_minibatch_size,
    )

    def get_datasets(key: chex.PRNGKey):
        datasets = []
        for _ in range(steps):
            key, subkey = jax.random.split(key)
            datasets.append(create_path(subkey))
        flattened = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs)[:dataset_size], *datasets
        )
        assert flattened[1].shape[0] == dataset_size
        return flattened

    return get_datasets


def create_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    dataset_minibatch_size: int,
    key: chex.PRNGKey,
):
    trajectory = puzzle.batched_get_random_trajectory(
        shuffle_length, shuffle_parallel, key
    )
    return trajectory_to_transition_dataset(trajectory, dataset_minibatch_size)


def get_sample_data_builder(puzzle: Puzzle, dataset_size: int, shuffle_parallel: int):
    steps = math.ceil(dataset_size / shuffle_parallel)
    create_samples = partial(create_sample_data, puzzle, shuffle_parallel)

    def get_datasets(key: chex.PRNGKey):
        datasets = []
        for _ in range(steps):
            key, subkey = jax.random.split(key)
            datasets.append(create_samples(subkey))
        flattened = jax.tree_util.tree_map(
            lambda *xs: xnp.concatenate(xs)[:dataset_size], *datasets
        )
        assert flattened[0].shape[0] == dataset_size
        return flattened

    return get_datasets


def create_sample_data(puzzle: Puzzle, shuffle_parallel: int, key: chex.PRNGKey):
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key, shuffle_parallel)
    )
    return solve_configs.GoalSpec, initial_states


def create_eval_trajectory(puzzle: Puzzle, shuffle_length: int, key: chex.PRNGKey):
    trajectory = puzzle.batched_get_random_trajectory(shuffle_length, 1, key)
    return trajectory_to_eval_trajectory(trajectory)
