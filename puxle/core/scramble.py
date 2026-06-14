"""Random-walk scramble engine for Puzzle implementations.

The functions in this module consume only the Puzzle public interface. Puzzle
keeps compatibility methods that delegate here, while the implementation lives
outside the base interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

from puxle.core.trajectory import PuzzleTrajectory

if TYPE_CHECKING:
    from puxle.core.puzzle_base import Puzzle


def _masked_action_sample_uniform(mask: chex.Array, key: chex.PRNGKey) -> chex.Array:
    mask_bt = mask.T
    logits = jnp.where(
        mask_bt, jnp.array(0.0, dtype=jnp.float32), jnp.array(-1.0e9, dtype=jnp.float32)
    )
    keys = jax.random.split(key, logits.shape[0])
    actions = jax.vmap(lambda k, lg: jax.random.categorical(k, lg, axis=-1))(
        keys, logits
    )
    return actions.astype(jnp.int32)


def _mask_inverse_action(
    previous_action: chex.Array,
    candidate_mask: chex.Array,
    inverse_action_permutation: chex.Array,
    action_size: int,
) -> chex.Array:
    valid_idx = previous_action >= 0
    safe_prev = jnp.where(valid_idx, previous_action, 0)
    inverse_actions = inverse_action_permutation[safe_prev]
    return jnp.where(
        valid_idx[jnp.newaxis, :]
        & (jnp.arange(action_size)[:, jnp.newaxis] == inverse_actions[jnp.newaxis, :]),
        False,
        candidate_mask,
    )


def _gather_by_action(neighbor_states, actions: chex.Array):
    batch_idx = jnp.arange(actions.shape[0], dtype=jnp.int32)

    def _gather(leaf: chex.Array) -> chex.Array:
        return leaf[actions, batch_idx, ...]

    return jax.tree_util.tree_map(_gather, neighbor_states)


def _leafwise_equal(
    candidate_leaf: chex.Array, reference_leaf: chex.Array
) -> chex.Array:
    expanded_ref = reference_leaf[jnp.newaxis, ...]
    eq = jnp.equal(candidate_leaf, expanded_ref)
    if eq.ndim <= 2:
        return eq
    axes = tuple(range(2, eq.ndim))
    return jnp.all(eq, axis=axes)


def _states_equal(candidate_states, reference_state) -> chex.Array:
    equality_tree = jax.tree_util.tree_map(
        _leafwise_equal, candidate_states, reference_state
    )
    leaves, _ = jax.tree_util.tree_flatten(equality_tree)
    if not leaves:
        raise ValueError("State comparison received an empty tree")
    result = leaves[0]
    for leaf in leaves[1:]:
        result = jnp.logical_and(result, leaf)
    return result


def _match_history(candidate_states, history_states) -> chex.Array:
    def _compare(prev_state):
        return _states_equal(candidate_states, prev_state)

    matches = jax.vmap(_compare)(history_states)
    return jnp.any(matches, axis=0)


def _initialize_history(state, history_len: int):
    if history_len <= 0:
        return None

    def _repeat(leaf):
        expanded = leaf[jnp.newaxis, ...]
        return jnp.repeat(expanded, repeats=history_len, axis=0)

    return jax.tree_util.tree_map(_repeat, state)


def _roll_history(history_states, new_state):
    if history_states is None:
        return None
    return jax.tree_util.tree_map(
        lambda h, n: jnp.concatenate([h[1:, ...], n[jnp.newaxis, ...]], axis=0),
        history_states,
        new_state,
    )


def _get_shuffled_state(
    puzzle: "Puzzle",
    solve_config: "Puzzle.SolveConfig",
    init_state: "Puzzle.State",
    key,
    num_shuffle,
):
    """Generate a scrambled state by applying random actions."""
    key, subkey = jax.random.split(key)
    num_shuffle += jax.random.randint(subkey, (), 0, 2)

    if puzzle.is_reversible:
        action_size = puzzle.action_size
        inv_map = puzzle.inverse_action_permutation

        def cond_fun_reversible(loop_state):
            iteration_count, _, _, _ = loop_state
            return iteration_count < num_shuffle

        def body_fun_reversible(loop_state):
            iteration_count, current_state, previous_action, key = loop_state
            key, subkey = jax.random.split(key)

            mask = jnp.ones(action_size, dtype=jnp.float32)

            def mask_inverse(prev_action, m):
                inv_action = inv_map[prev_action]
                return m.at[inv_action].set(0.0)

            valid_mask = jax.lax.cond(
                previous_action >= 0,
                lambda: mask_inverse(previous_action, mask),
                lambda: mask,
            )

            action = jax.random.choice(subkey, action_size, p=valid_mask)
            next_state, _ = puzzle.get_actions(
                solve_config, current_state, action, filled=True
            )
            return (iteration_count + 1, next_state, action, key)

        _, final_state, _, _ = jax.lax.while_loop(
            cond_fun_reversible, body_fun_reversible, (0, init_state, -1, key)
        )
        return final_state

    def cond_fun_irreversible(loop_state):
        iteration_count, _, _, _ = loop_state
        return iteration_count < num_shuffle

    def body_fun_irreversible(loop_state):
        iteration_count, current_state, previous_state, key = loop_state
        neighbor_states, costs = puzzle.get_neighbours(
            solve_config, current_state, filled=True
        )
        old_eq = jax.vmap(lambda x, y: x == y, in_axes=(None, 0))(
            previous_state, neighbor_states
        )
        valid_mask = jnp.where(old_eq, 0.0, 1.0)
        valid_mask_sum = jnp.sum(valid_mask)
        probabilities = jax.lax.cond(
            valid_mask_sum > 0,
            lambda: valid_mask / valid_mask_sum,
            lambda: jnp.ones_like(costs) / costs.shape[0],
        )
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, jnp.arange(costs.shape[0]), p=probabilities)
        next_state = neighbor_states[idx]
        return (iteration_count + 1, next_state, current_state, key)

    _, final_state, _, _ = jax.lax.while_loop(
        cond_fun_irreversible,
        body_fun_irreversible,
        (0, init_state, init_state, key),
    )
    return final_state


def _batched_get_random_trajectory(
    puzzle: "Puzzle",
    k_max: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
):
    key_inits, key_scan = jax.random.split(key, 2)
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key_inits, shuffle_parallel)
    )
    step_keys = jax.random.split(key_scan, k_max)

    if puzzle.is_reversible and non_backtracking_steps == 1:
        action_size = puzzle.action_size
        inv_map = puzzle.inverse_action_permutation

        def _scan_fast(carry, step_key):
            state, move_cost, previous_action = carry
            neighbor_states, cost = puzzle.batched_get_neighbours(
                solve_configs,
                state,
                filleds=jnp.ones_like(move_cost),
                multi_solve_config=True,
            )
            mask = jnp.isfinite(cost)

            final_mask = _mask_inverse_action(
                previous_action, mask, inv_map, action_size
            )
            no_valid = jnp.sum(final_mask, axis=0) == 0
            final_mask = jnp.where(no_valid[jnp.newaxis, :], mask, final_mask)
            actions = _masked_action_sample_uniform(final_mask, step_key)
            next_state = _gather_by_action(neighbor_states, actions)
            batch_idx = jnp.arange(actions.shape[0], dtype=jnp.int32)
            step_cost = cost[actions, batch_idx]
            return (
                (next_state, move_cost + step_cost, actions),
                (state, move_cost, actions, step_cost),
            )

        (
            (last_state, last_move_cost, _),
            (states, move_costs, actions, action_costs),
        ) = jax.lax.scan(
            _scan_fast,
            (
                initial_states,
                jnp.zeros(shuffle_parallel),
                jnp.full((shuffle_parallel,), -1, dtype=jnp.int32),
            ),
            step_keys,
            length=k_max,
        )
    else:
        if non_backtracking_steps < 0:
            raise ValueError("non_backtracking_steps must be non-negative")
        history_states = _initialize_history(
            initial_states, int(non_backtracking_steps)
        )
        use_history = history_states is not None

        def _scan_legacy(carry, step_key):
            history, state, move_cost = carry
            neighbor_states, cost = puzzle.batched_get_neighbours(
                solve_configs,
                state,
                filleds=jnp.ones_like(move_cost),
                multi_solve_config=True,
            )
            action_mask = jnp.isfinite(cost)
            history_block = (
                _match_history(neighbor_states, history)
                if use_history
                else jnp.zeros_like(action_mask)
            )
            same_block = _states_equal(neighbor_states, state)
            backtracking_mask = (~history_block) & (~same_block)
            masked = action_mask & backtracking_mask
            no_valid_backtracking = jnp.sum(masked, axis=0) == 0
            final_mask = jnp.where(
                no_valid_backtracking[jnp.newaxis, :], action_mask, masked
            )
            actions = _masked_action_sample_uniform(final_mask, step_key)
            next_state = _gather_by_action(neighbor_states, actions)
            batch_idx = jnp.arange(actions.shape[0], dtype=jnp.int32)
            step_cost = cost[actions, batch_idx]
            next_history = _roll_history(history, state) if use_history else history
            return (
                (next_history, next_state, move_cost + step_cost),
                (state, move_cost, actions, step_cost),
            )

        (
            (_, last_state, last_move_cost),
            (states, move_costs, actions, action_costs),
        ) = jax.lax.scan(
            _scan_legacy,
            (history_states, initial_states, jnp.zeros(shuffle_parallel)),
            step_keys,
            length=k_max,
        )

    states = jax.tree_util.tree_map(
        lambda s_seq, s_last: jnp.concatenate(
            [s_seq, s_last[jnp.newaxis, ...]], axis=0
        ),
        states,
        last_state,
    )
    move_costs = jnp.concatenate([move_costs, last_move_cost[jnp.newaxis, ...]], axis=0)
    move_costs_tm1 = jnp.concatenate(
        [jnp.zeros_like(move_costs[:1, ...]), move_costs[:-1, ...]], axis=0
    )

    return PuzzleTrajectory(
        solve_configs=solve_configs,
        states=states,
        move_costs=move_costs,
        move_costs_tm1=move_costs_tm1,
        actions=actions,
        action_costs=action_costs,
    )


def _batched_get_random_inverse_trajectory(
    puzzle: "Puzzle",
    k_max: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
    non_backtracking_steps: int = 3,
):
    key_inits, key_targets, key_scan = jax.random.split(key, 3)
    solve_configs, _ = jax.vmap(puzzle.get_inits)(
        jax.random.split(key_inits, shuffle_parallel)
    )
    target_states = jax.vmap(puzzle.solve_config_to_state_transform, in_axes=(0, 0))(
        solve_configs, jax.random.split(key_targets, shuffle_parallel)
    )
    step_keys = jax.random.split(key_scan, k_max)

    if puzzle.is_reversible and non_backtracking_steps == 1:
        action_size = puzzle.action_size
        inv_map = puzzle.inverse_action_permutation

        def _scan_fast(carry, step_key):
            state, move_cost, previous_action = carry
            neighbor_states, cost = puzzle.batched_get_inverse_neighbours(
                solve_configs,
                state,
                filleds=jnp.ones_like(move_cost),
                multi_solve_config=True,
            )
            mask = jnp.isfinite(cost)

            final_mask = _mask_inverse_action(
                previous_action, mask, inv_map, action_size
            )
            no_valid = jnp.sum(final_mask, axis=0) == 0
            final_mask = jnp.where(no_valid[jnp.newaxis, :], mask, final_mask)
            inv_actions = _masked_action_sample_uniform(final_mask, step_key)
            next_state = _gather_by_action(neighbor_states, inv_actions)
            batch_idx = jnp.arange(inv_actions.shape[0], dtype=jnp.int32)
            step_cost = cost[inv_actions, batch_idx]
            return (
                (next_state, move_cost + step_cost, inv_actions),
                (state, move_cost, inv_actions, step_cost),
            )

        (
            (last_state, last_move_cost, _),
            (states, move_costs, inv_actions, action_costs),
        ) = jax.lax.scan(
            _scan_fast,
            (
                target_states,
                jnp.zeros(shuffle_parallel),
                jnp.full((shuffle_parallel,), -1, dtype=jnp.int32),
            ),
            step_keys,
            length=k_max,
        )
    else:
        if non_backtracking_steps < 0:
            raise ValueError("non_backtracking_steps must be non-negative")
        history_states = _initialize_history(target_states, int(non_backtracking_steps))
        use_history = history_states is not None

        def _scan_legacy(carry, step_key):
            history, state, move_cost = carry
            neighbor_states, cost = puzzle.batched_get_inverse_neighbours(
                solve_configs,
                state,
                filleds=jnp.ones_like(move_cost),
                multi_solve_config=True,
            )
            action_mask = jnp.isfinite(cost)
            history_block = (
                _match_history(neighbor_states, history)
                if use_history
                else jnp.zeros_like(action_mask)
            )
            same_block = _states_equal(neighbor_states, state)
            backtracking_mask = (~history_block) & (~same_block)
            masked = action_mask & backtracking_mask
            no_valid_backtracking = jnp.sum(masked, axis=0) == 0
            final_mask = jnp.where(
                no_valid_backtracking[jnp.newaxis, :], action_mask, masked
            )
            inv_actions = _masked_action_sample_uniform(final_mask, step_key)
            next_state = _gather_by_action(neighbor_states, inv_actions)
            batch_idx = jnp.arange(inv_actions.shape[0], dtype=jnp.int32)
            step_cost = cost[inv_actions, batch_idx]
            next_history = _roll_history(history, state) if use_history else history
            return (
                (next_history, next_state, move_cost + step_cost),
                (state, move_cost, inv_actions, step_cost),
            )

        (
            (_, last_state, last_move_cost),
            (states, move_costs, inv_actions, action_costs),
        ) = jax.lax.scan(
            _scan_legacy,
            (history_states, target_states, jnp.zeros(shuffle_parallel)),
            step_keys,
            length=k_max,
        )

    states = jax.tree_util.tree_map(
        lambda s_seq, s_last: jnp.concatenate(
            [s_seq, s_last[jnp.newaxis, ...]], axis=0
        ),
        states,
        last_state,
    )
    move_costs = jnp.concatenate([move_costs, last_move_cost[jnp.newaxis, ...]], axis=0)
    move_costs_tm1 = jnp.concatenate(
        [jnp.zeros_like(move_costs[:1, ...]), move_costs[:-1, ...]], axis=0
    )

    return PuzzleTrajectory(
        solve_configs=solve_configs,
        states=states,
        move_costs=move_costs,
        move_costs_tm1=move_costs_tm1,
        actions=inv_actions,
        action_costs=action_costs,
    )
