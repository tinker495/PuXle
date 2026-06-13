"""Puzzle Trajectory Module.

Owns the algorithms that turn a Puzzle's batched (inverse) trajectory generator
into a flat ``PuzzleTrajectory`` record. Consumed directly by PuXle tests and
indirectly by JAxtar's neural training builders through the JAxtar-owned
``trajectory_to_dataset_dict`` adapter.

The Module consumes only the public ``Puzzle`` Interface
(``batched_get_random_trajectory``, ``batched_get_random_inverse_trajectory``,
``batched_hindsight_transform``); it does not reach into ``_get_shuffled_state``
or other private helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp

if TYPE_CHECKING:
    from puxle.core.puzzle_base import Puzzle


@chex.dataclass
class PuzzleTrajectory:
    """
    A unified JAX PyTree dataclass representing extracted puzzle environments,
    shuffled states, paths, and their corresponding algorithmic generation costs.
    """

    solve_configs: chex.Array
    states: chex.Array
    move_costs: chex.Array
    move_costs_tm1: chex.Array
    actions: chex.Array
    action_costs: chex.Array

    # Hindsight wrapper auxiliary indices
    parent_indices: Optional[chex.Array] = None
    trajectory_indices: Optional[chex.Array] = None
    step_indices: Optional[chex.Array] = None


def _repeat_solve_configs_for_steps(solve_configs, k_max: int):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.repeat(leaf[:, jnp.newaxis, ...], k_max, axis=1),
        solve_configs,
    )


def _trajectory_indices(shuffle_parallel: int, k_max: int) -> chex.Array:
    return jnp.broadcast_to(
        jnp.arange(shuffle_parallel, dtype=jnp.int32)[:, jnp.newaxis],
        (shuffle_parallel, k_max),
    )


def _step_indices(shuffle_parallel: int, k_max: int) -> chex.Array:
    return jnp.broadcast_to(
        jnp.arange(k_max, dtype=jnp.int32)[jnp.newaxis, :],
        (shuffle_parallel, k_max),
    )


def _chain_parent_indices(shuffle_parallel: int, k_max: int) -> chex.Array:
    indices = jnp.arange(k_max * shuffle_parallel, dtype=jnp.int32)
    parent_indices = indices - 1
    parent_indices = parent_indices.reshape(shuffle_parallel, k_max)
    return parent_indices.at[:, 0].set(-1)


def _flatten_batch_time_trajectory(
    *,
    solve_configs,
    states,
    move_costs: chex.Array,
    move_costs_tm1: chex.Array,
    actions: chex.Array,
    action_costs: chex.Array,
    parent_indices: chex.Array,
    trajectory_indices: chex.Array,
    step_indices: chex.Array,
) -> PuzzleTrajectory:
    return PuzzleTrajectory(
        solve_configs=solve_configs.flatten(),
        states=states.flatten(),
        move_costs=move_costs.flatten(),
        move_costs_tm1=move_costs_tm1.flatten(),
        actions=actions.flatten(),
        action_costs=action_costs.flatten(),
        parent_indices=parent_indices.flatten(),
        trajectory_indices=trajectory_indices.flatten(),
        step_indices=step_indices.flatten(),
    )


def _flatten_chain_trajectory(
    *,
    solve_configs,
    states,
    move_costs: chex.Array,
    move_costs_tm1: chex.Array,
    actions: chex.Array,
    action_costs: chex.Array,
    shuffle_parallel: int,
    k_max: int,
) -> PuzzleTrajectory:
    return _flatten_batch_time_trajectory(
        solve_configs=solve_configs,
        states=states,
        move_costs=move_costs,
        move_costs_tm1=move_costs_tm1,
        actions=actions,
        action_costs=action_costs,
        parent_indices=_chain_parent_indices(shuffle_parallel, k_max),
        trajectory_indices=_trajectory_indices(shuffle_parallel, k_max),
        step_indices=_step_indices(shuffle_parallel, k_max),
    )


def create_target_shuffled_path(
    puzzle: "Puzzle",
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
) -> PuzzleTrajectory:
    """Generate ``shuffle_parallel * k_max`` (solve_config, state, ...) tuples by
    inverse-walking from each goal. Returns a flat ``PuzzleTrajectory``.
    """
    inverse_trajectory = puzzle.batched_get_random_inverse_trajectory(
        k_max, shuffle_parallel, key, non_backtracking_steps=non_backtracking_steps
    )
    solve_configs = inverse_trajectory.solve_configs
    if include_solved_states:
        states = jax.tree_util.tree_map(
            lambda leaf: leaf[:-1, ...], inverse_trajectory.states
        )
        move_costs = inverse_trajectory.move_costs[:-1, ...]
        move_costs_tm1 = inverse_trajectory.move_costs_tm1[:-1, ...]
    else:
        states = jax.tree_util.tree_map(
            lambda leaf: leaf[1:, ...], inverse_trajectory.states
        )
        move_costs = inverse_trajectory.move_costs[1:, ...]
        move_costs_tm1 = inverse_trajectory.move_costs_tm1[1:, ...]
    inv_actions = inverse_trajectory.actions
    action_costs = inverse_trajectory.action_costs

    states = states.transpose((1, 0))
    move_costs = move_costs.transpose((1, 0))
    move_costs_tm1 = move_costs_tm1.transpose((1, 0))
    inv_actions = inv_actions.transpose((1, 0))
    action_costs = action_costs.transpose((1, 0))

    solve_configs = _repeat_solve_configs_for_steps(solve_configs, k_max)
    return _flatten_chain_trajectory(
        solve_configs=solve_configs,
        states=states,
        move_costs=move_costs,
        move_costs_tm1=move_costs_tm1,
        actions=inv_actions,
        action_costs=action_costs,
        shuffle_parallel=shuffle_parallel,
        k_max=k_max,
    )


def create_hindsight_target_shuffled_path(
    puzzle: "Puzzle",
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
) -> PuzzleTrajectory:
    """Hindsight variant: relabel each trajectory's final state as the target."""
    assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"
    key_traj, key_append = jax.random.split(key, 2)
    trajectory = puzzle.batched_get_random_trajectory(
        k_max,
        shuffle_parallel,
        key_traj,
        non_backtracking_steps=non_backtracking_steps,
    )

    original_solve_configs = trajectory.solve_configs
    states = trajectory.states
    move_costs = trajectory.move_costs
    move_costs_tm1 = trajectory.move_costs_tm1
    actions = trajectory.actions
    action_costs = trajectory.action_costs

    targets = states[-1, ...]
    if include_solved_states:
        states = states[1:, ...]
    else:
        states = states[:-1, ...]

    solve_configs = puzzle.batched_hindsight_transform(original_solve_configs, targets)

    if include_solved_states:
        move_costs = move_costs[-1, ...] - move_costs[1:, ...]
        move_costs_tm1 = move_costs[-1, ...] - move_costs_tm1[1:, ...]
        actions = jnp.concatenate(
            [
                actions[1:],
                jax.random.randint(
                    key_append,
                    (1, shuffle_parallel),
                    minval=0,
                    maxval=puzzle.action_size,
                ),
            ]
        )
        action_costs = jnp.concatenate(
            [action_costs[1:], jnp.zeros((1, shuffle_parallel))]
        )
    else:
        move_costs = move_costs[-1, ...] - move_costs[:-1, ...]
        move_costs_tm1 = move_costs[-1, ...] - move_costs_tm1[:-1, ...]
        move_costs_tm1 = move_costs_tm1.at[0, ...].set(0.0)

    states = states[::-1, ...]
    move_costs = move_costs[::-1, ...]
    move_costs_tm1 = move_costs_tm1[::-1, ...]
    actions = actions[::-1, ...]
    action_costs = action_costs[::-1, ...]

    states = states.transpose((1, 0))
    move_costs = move_costs.transpose((1, 0))
    move_costs_tm1 = move_costs_tm1.transpose((1, 0))
    actions = actions.transpose((1, 0))
    action_costs = action_costs.transpose((1, 0))

    solve_configs = _repeat_solve_configs_for_steps(solve_configs, k_max)
    return _flatten_chain_trajectory(
        solve_configs=solve_configs,
        states=states,
        move_costs=move_costs,
        move_costs_tm1=move_costs_tm1,
        actions=actions,
        action_costs=action_costs,
        shuffle_parallel=shuffle_parallel,
        k_max=k_max,
    )


def create_hindsight_target_triangular_shuffled_path(
    puzzle: "Puzzle",
    k_max: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
) -> PuzzleTrajectory:
    """Hindsight variant with uniform path-length sampling for triangular paths."""
    assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"
    key, subkey = jax.random.split(key)
    trajectory = puzzle.batched_get_random_trajectory(
        k_max,
        shuffle_parallel,
        subkey,
        non_backtracking_steps=non_backtracking_steps,
    )

    original_solve_configs = trajectory.solve_configs
    states = trajectory.states
    move_costs = trajectory.move_costs
    move_costs_tm1 = trajectory.move_costs_tm1
    actions = trajectory.actions
    action_costs = trajectory.action_costs

    key, key_k, key_i = jax.random.split(key, 3)

    minval = 0 if include_solved_states else 1
    k = jax.random.randint(
        key_k, shape=(k_max, shuffle_parallel), minval=minval, maxval=k_max + 1
    )
    random_floats = jax.random.uniform(key_i, shape=(k_max, shuffle_parallel))
    max_start_idx = k_max - k
    start_indices = (random_floats * (max_start_idx + 1)).astype(jnp.int32)

    target_indices = start_indices + k
    parallel_indices = jnp.tile(jnp.arange(shuffle_parallel)[None, :], (k_max, 1))

    start_states = states[start_indices, parallel_indices]
    target_states = states[target_indices, parallel_indices]

    start_move_costs = move_costs[start_indices, parallel_indices]
    target_move_costs = move_costs[target_indices, parallel_indices]
    start_move_costs_tm1 = move_costs_tm1[start_indices, parallel_indices]
    final_move_costs = target_move_costs - start_move_costs
    final_move_costs_tm1 = target_move_costs - start_move_costs_tm1
    final_move_costs_tm1 = jnp.where(start_indices == 0, 0.0, final_move_costs_tm1)

    clamped_start_indices = jnp.clip(start_indices, 0, k_max - 1)
    final_actions = actions[clamped_start_indices, parallel_indices]
    final_action_costs = action_costs[clamped_start_indices, parallel_indices]

    is_goal_state = (k == 0) & include_solved_states
    final_action_costs = jnp.where(is_goal_state, 0.0, final_action_costs)

    tiled_solve_configs = xnp.repeat(
        original_solve_configs[jnp.newaxis, ...], k_max, axis=0
    )
    flat_tiled_sc = tiled_solve_configs.flatten()
    flat_target_states = target_states.flatten()
    final_solve_configs = puzzle.batched_hindsight_transform(
        flat_tiled_sc, flat_target_states
    ).reshape((k_max, shuffle_parallel, -1))

    k_transposed = k.transpose((1, 0))
    sort_indices = jnp.argsort(k_transposed, axis=1)

    def _sort_and_transpose(arr_tree):
        def _op(arr):
            arr_t = jnp.swapaxes(arr, 0, 1)
            indices = sort_indices
            while indices.ndim < arr_t.ndim:
                indices = indices[..., jnp.newaxis]
            return jnp.take_along_axis(arr_t, indices, axis=1)

        return jax.tree_util.tree_map(_op, arr_tree)

    final_solve_configs = _sort_and_transpose(final_solve_configs)
    final_start_states = _sort_and_transpose(start_states)
    final_move_costs = _sort_and_transpose(final_move_costs)
    final_move_costs_tm1 = _sort_and_transpose(final_move_costs_tm1)
    final_actions = _sort_and_transpose(final_actions)
    final_action_costs = _sort_and_transpose(final_action_costs)

    step_indices = jnp.take_along_axis(k_transposed, sort_indices, axis=1)

    parent_indices = jnp.full((shuffle_parallel, k_max), -1, dtype=jnp.int32)
    return _flatten_batch_time_trajectory(
        solve_configs=final_solve_configs,
        states=final_start_states,
        move_costs=final_move_costs,
        move_costs_tm1=final_move_costs_tm1,
        actions=final_actions,
        action_costs=final_action_costs,
        parent_indices=parent_indices,
        trajectory_indices=_trajectory_indices(shuffle_parallel, k_max),
        step_indices=step_indices,
    )


__all__ = [
    "PuzzleTrajectory",
    "create_target_shuffled_path",
    "create_hindsight_target_shuffled_path",
    "create_hindsight_target_triangular_shuffled_path",
]
