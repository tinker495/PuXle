import importlib
import inspect
import pkgutil

import jax
import jax.numpy as jnp
import pytest

from puxle.core.puzzle_base import Puzzle


def get_all_puzzles():
    try:
        import puxle.puzzles
    except ImportError:
        return []

    puzzles = []
    package = puxle.puzzles
    path = package.__path__
    prefix = package.__name__ + "."

    for _, name, _ in pkgutil.iter_modules(path, prefix):
        try:
            module = importlib.import_module(name)
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Puzzle)
                    and obj is not Puzzle
                ):
                    puzzles.append(obj)
        except Exception as e:
            print(f"Skipping module {name}: {e}")
    return puzzles


ALL_PUZZLES = get_all_puzzles()


@pytest.fixture(scope="module", params=ALL_PUZZLES, ids=lambda cls: cls.__name__)
def puzzle_instance(request):
    puzzle_cls = request.param
    try:
        puzzle = puzzle_cls()
        return puzzle
    except Exception as e:
        pytest.skip(f"Could not instantiate {puzzle_cls.__name__}: {e}")


@pytest.fixture
def puzzle_init(puzzle_instance):
    key = jax.random.PRNGKey(0)
    try:
        solve_config, state = puzzle_instance.get_inits(key)
        return solve_config, state
    except Exception as e:
        pytest.skip(f"get_inits failed for {puzzle_instance.__class__.__name__}: {e}")


def test_hindsight_transform_immutability(puzzle_instance, puzzle_init):
    if not puzzle_instance.has_target:
        pytest.skip("Puzzle does not support hindsight_transform")

    solve_config, state = puzzle_init

    original_target_leaves = jax.tree_util.tree_leaves(solve_config.GoalSpec)
    # Deep copy needed for array comparison later
    original_target_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_target_leaves
    ]

    original_id = id(solve_config)

    try:
        output_config = puzzle_instance.hindsight_transform(solve_config, state)
    except NotImplementedError:
        pytest.skip("hindsight_transform not implemented")

    # Require a distinct object instance to catch in-place mutation or returning self unchanged.
    assert id(output_config) != original_id, (
        "hindsight_transform returned the exact same object instance. Use .replace() to return a new object."
    )

    current_target_leaves = jax.tree_util.tree_leaves(solve_config.GoalSpec)

    for old, new in zip(original_target_leaves_copy, current_target_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "Original solve_config.GoalSpec was mutated in place!"
            )
        else:
            assert old == new, "Original solve_config.GoalSpec was mutated in place!"


def test_get_actions_immutability(puzzle_instance, puzzle_init):
    solve_config, state = puzzle_init

    original_state_leaves = jax.tree_util.tree_leaves(state)
    original_state_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_state_leaves
    ]

    action = jnp.array(0)

    try:
        next_state, cost = puzzle_instance.get_actions(solve_config, state, action)
    except Exception as e:
        # Fail rather than skip because every puzzle must support get_actions.
        pytest.fail(f"get_actions raised exception: {e}")

    current_state_leaves = jax.tree_util.tree_leaves(state)

    for old, new in zip(original_state_leaves_copy, current_state_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "Original state was mutated in place by get_actions!"
            )
        else:
            assert old == new, "Original state was mutated in place by get_actions!"


def test_get_neighbours_immutability(puzzle_instance, puzzle_init):
    solve_config, state = puzzle_init

    original_state_leaves = jax.tree_util.tree_leaves(state)
    original_state_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_state_leaves
    ]
    original_config_leaves = jax.tree_util.tree_leaves(solve_config)
    original_config_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_config_leaves
    ]

    try:
        neighbours, costs = puzzle_instance.get_neighbours(solve_config, state)
    except Exception as e:
        pytest.fail(f"get_neighbours raised exception: {e}")

    current_state_leaves = jax.tree_util.tree_leaves(state)
    for old, new in zip(original_state_leaves_copy, current_state_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "Original state was mutated in place by get_neighbours!"
            )
        else:
            assert old == new, "Original state was mutated in place by get_neighbours!"

    current_config_leaves = jax.tree_util.tree_leaves(solve_config)
    for old, new in zip(original_config_leaves_copy, current_config_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "Original solve_config was mutated in place by get_neighbours!"
            )
        else:
            assert old == new, (
                "Original solve_config was mutated in place by get_neighbours!"
            )


def test_get_inverse_neighbours_immutability(puzzle_instance, puzzle_init):
    if not puzzle_instance.is_reversible:
        # Non-reversible puzzles (e.g. Sokoban) may still override get_inverse_neighbours; skip only on NotImplementedError below.
        pass

    solve_config, state = puzzle_init

    original_state_leaves = jax.tree_util.tree_leaves(state)
    original_state_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_state_leaves
    ]
    original_config_leaves = jax.tree_util.tree_leaves(solve_config)
    original_config_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_config_leaves
    ]

    try:
        neighbours, costs = puzzle_instance.get_inverse_neighbours(solve_config, state)
    except NotImplementedError:
        pytest.skip("get_inverse_neighbours not implemented")
    except Exception as e:
        pytest.fail(f"get_inverse_neighbours raised exception: {e}")

    current_state_leaves = jax.tree_util.tree_leaves(state)
    for old, new in zip(original_state_leaves_copy, current_state_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "Original state was mutated in place by get_inverse_neighbours!"
            )
        else:
            assert old == new, (
                "Original state was mutated in place by get_inverse_neighbours!"
            )

    current_config_leaves = jax.tree_util.tree_leaves(solve_config)
    for old, new in zip(original_config_leaves_copy, current_config_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "Original solve_config was mutated in place by get_inverse_neighbours!"
            )
        else:
            assert old == new, (
                "Original solve_config was mutated in place by get_inverse_neighbours!"
            )


def test_is_solved_immutability(puzzle_instance, puzzle_init):
    solve_config, state = puzzle_init

    original_state_leaves = jax.tree_util.tree_leaves(state)
    original_state_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_state_leaves
    ]
    original_config_leaves = jax.tree_util.tree_leaves(solve_config)
    original_config_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_config_leaves
    ]

    try:
        puzzle_instance.is_solved(solve_config, state)
    except Exception as e:
        pytest.fail(f"is_solved raised exception: {e}")

    current_state_leaves = jax.tree_util.tree_leaves(state)
    for old, new in zip(original_state_leaves_copy, current_state_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "Original state was mutated in place by is_solved!"
            )
        else:
            assert old == new, "Original state was mutated in place by is_solved!"

    current_config_leaves = jax.tree_util.tree_leaves(solve_config)
    for old, new in zip(original_config_leaves_copy, current_config_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "Original solve_config was mutated in place by is_solved!"
            )
        else:
            assert old == new, (
                "Original solve_config was mutated in place by is_solved!"
            )


def test_initialization_immutability(puzzle_instance):
    key = jax.random.PRNGKey(0)
    try:
        data = puzzle_instance.get_data(key)
        solve_config = puzzle_instance.get_solve_config(key, data)
    except Exception as e:
        pytest.skip(f"Setup failed: {e}")

    original_config_leaves = jax.tree_util.tree_leaves(solve_config)
    original_config_leaves_copy = [
        jnp.array(x) if hasattr(x, "copy") else x for x in original_config_leaves
    ]

    try:
        puzzle_instance.get_initial_state(solve_config, key, data)
    except Exception as e:
        pytest.fail(f"get_initial_state failed: {e}")

    current_config_leaves = jax.tree_util.tree_leaves(solve_config)
    for old, new in zip(original_config_leaves_copy, current_config_leaves):
        if hasattr(old, "shape") and hasattr(new, "shape"):
            assert jnp.array_equal(old, new), (
                "solve_config was mutated in place by get_initial_state!"
            )
        else:
            assert old == new, "solve_config was mutated in place by get_initial_state!"
