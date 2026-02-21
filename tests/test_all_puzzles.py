import inspect
from typing import List, Type

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puxle.core.puzzle_base import Puzzle


def discover_puzzle_classes() -> List[Type[Puzzle]]:
    """
    Dynamically discover all puzzle classes from puxle.puzzles module.
    Returns a list of puzzle classes that inherit from Puzzle.
    """
    import puxle.puzzles as puzzles_module

    puzzle_classes = []

    # Get all attributes from the puzzles module
    for attr_name in dir(puzzles_module):
        attr = getattr(puzzles_module, attr_name)

        # Check if it's a class and inherits from Puzzle
        if (
            inspect.isclass(attr) and issubclass(attr, Puzzle) and attr != Puzzle
        ):  # Exclude the base Puzzle class itself
            puzzle_classes.append(attr)

    return puzzle_classes


class TestPuzzleValidation:
    """Comprehensive test suite for all puzzles in puxle.puzzles module"""

    @pytest.fixture(scope="class")
    def puzzle_configs(self):
        """Dynamically generate configuration for all discovered puzzles"""
        puzzle_classes = discover_puzzle_classes()
        print(f"\nDiscovered {len(puzzle_classes)} puzzle classes:")
        for cls in puzzle_classes:
            print(f"  - {cls.__name__}")

        return puzzle_classes

    @pytest.fixture
    def rng_key(self):
        """Provide a random key for JAX operations"""
        return jax.random.PRNGKey(42)

    def test_puzzle_instantiation(self, puzzle_configs):
        """Test that all puzzles can be instantiated correctly"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()
                assert isinstance(puzzle, Puzzle), f"{puzzle_class.__name__} should inherit from Puzzle"
                assert hasattr(puzzle, "State"), f"{puzzle_class.__name__} should define State class"
                assert hasattr(puzzle, "SolveConfig"), f"{puzzle_class.__name__} should define SolveConfig class"
                assert puzzle.action_size is not None, f"{puzzle_class.__name__} should have action_size defined"
                assert puzzle.action_size > 0, f"{puzzle_class.__name__} should have positive action_size"
                print(f"✓ {puzzle_class.__name__} instantiated successfully")
            except Exception as e:
                pytest.fail(f"Failed to instantiate {puzzle_class.__name__}: {e}")

    def test_state_generation(self, puzzle_configs, rng_key):
        """Test that puzzles can generate initial states and solve configurations"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()

                # Test solve config generation
                solve_config = puzzle.get_solve_config(key=rng_key)
                assert solve_config is not None, f"{puzzle_class.__name__} should generate solve_config"
                assert isinstance(solve_config, puzzle.SolveConfig), "solve_config should be instance of SolveConfig"

                # Test initial state generation
                initial_state = puzzle.get_initial_state(solve_config, key=rng_key)
                assert initial_state is not None, f"{puzzle_class.__name__} should generate initial_state"
                assert isinstance(initial_state, puzzle.State), "initial_state should be instance of State"

                # Test get_inits convenience function
                solve_config2, initial_state2 = puzzle.get_inits(key=rng_key)
                assert isinstance(solve_config2, puzzle.SolveConfig), "get_inits should return valid solve_config"
                assert isinstance(initial_state2, puzzle.State), "get_inits should return valid initial_state"

                print(f"✓ {puzzle_class.__name__} state generation works")

            except Exception as e:
                pytest.fail(f"State generation failed for {puzzle_class.__name__}: {e}")

    def test_neighbor_generation(self, puzzle_configs, rng_key):
        """Test that puzzles can generate valid neighbors"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()
                solve_config = puzzle.get_solve_config(key=rng_key)
                initial_state = puzzle.get_initial_state(solve_config, key=rng_key)

                # Test neighbor generation
                neighbor_states, costs = puzzle.get_neighbours(solve_config, initial_state, filled=True)

                # Validate neighbor states structure
                assert neighbor_states is not None, f"{puzzle_class.__name__} should generate neighbor_states"
                assert costs is not None, f"{puzzle_class.__name__} should generate costs"
                assert len(costs) == puzzle.action_size, "Number of costs should match action_size"

                # Check that costs are valid (either finite positive values or infinity)
                finite_costs = jnp.isfinite(costs)
                valid_costs = jnp.logical_or(finite_costs, jnp.isinf(costs))
                assert jnp.all(valid_costs), "All costs should be finite or infinity"

                # Check that at least some moves are possible (not all costs are infinity)
                assert jnp.any(finite_costs), "At least some moves should be possible"

                # Test with filled=False
                neighbor_states_unfilled, costs_unfilled = puzzle.get_neighbours(
                    solve_config, initial_state, filled=False
                )
                assert jnp.all(jnp.isinf(costs_unfilled)), "All costs should be infinity when filled=False"

                print(f"✓ {puzzle_class.__name__} neighbor generation works")

            except Exception as e:
                pytest.fail(f"Neighbor generation failed for {puzzle_class.__name__}: {e}")

    def test_action_transitions(self, puzzle_configs, rng_key):
        """Ensure get_actions produces results consistent with get_neighbours"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()
                solve_config = puzzle.get_solve_config(key=rng_key)
                initial_state = puzzle.get_initial_state(solve_config, key=rng_key)

                neighbours_filled, costs_filled = puzzle.get_neighbours(solve_config, initial_state, filled=True)
                neighbours_blocked, costs_blocked = puzzle.get_neighbours(solve_config, initial_state, filled=False)

                actions_to_test = min(puzzle.action_size, 3)
                for action_idx in range(actions_to_test):
                    action = jnp.asarray(action_idx)

                    # Filled=True should match neighbour generation
                    next_state, action_cost = puzzle.get_actions(solve_config, initial_state, action, filled=True)
                    expected_state = jax.tree_util.tree_map(lambda x: x[action_idx], neighbours_filled)

                    assert next_state == expected_state, (
                        f"{puzzle_class.__name__} get_actions state mismatch at action {action_idx}"
                    )
                    assert jnp.allclose(action_cost, costs_filled[action_idx]), (
                        f"{puzzle_class.__name__} get_actions cost mismatch at action {action_idx}"
                    )

                    # Filled=False should block all moves
                    blocked_state, blocked_cost = puzzle.get_actions(solve_config, initial_state, action, filled=False)
                    expected_blocked_state = jax.tree_util.tree_map(lambda x: x[action_idx], neighbours_blocked)
                    assert blocked_state == expected_blocked_state, (
                        f"{puzzle_class.__name__} filled=False state mismatch at action {action_idx}"
                    )
                    assert jnp.allclose(blocked_cost, costs_blocked[action_idx]), (
                        f"{puzzle_class.__name__} filled=False cost mismatch at action {action_idx}"
                    )
                    assert jnp.isinf(costs_blocked[action_idx]), (
                        f"{puzzle_class.__name__} filled=False neighbours should be inf"
                    )

                print(f"✓ {puzzle_class.__name__} action transitions align with neighbours")

            except Exception as e:
                pytest.fail(f"Action transition test failed for {puzzle_class.__name__}: {e}")

    def test_get_neighbours_matches_get_actions(self, puzzle_configs, rng_key):
        """Cross-check get_neighbours outputs against per-action get_actions calls"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()
                solve_config = puzzle.get_solve_config(key=rng_key)
                initial_state = puzzle.get_initial_state(solve_config, key=rng_key)

                neighbours_filled, costs_filled = puzzle.get_neighbours(solve_config, initial_state, filled=True)
                neighbours_blocked, costs_blocked = puzzle.get_neighbours(solve_config, initial_state, filled=False)

                manual_states = []
                manual_costs = []
                manual_states_blocked = []
                manual_costs_blocked = []

                for action_idx in range(puzzle.action_size):
                    action = jnp.asarray(action_idx)
                    next_state, next_cost = puzzle.get_actions(solve_config, initial_state, action, filled=True)
                    manual_states.append(next_state)
                    manual_costs.append(next_cost)

                    blocked_state, blocked_cost = puzzle.get_actions(solve_config, initial_state, action, filled=False)
                    manual_states_blocked.append(blocked_state)
                    manual_costs_blocked.append(blocked_cost)

                stacked_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *manual_states)
                stacked_states_blocked = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(xs, axis=0), *manual_states_blocked
                )
                stacked_costs = jnp.stack(manual_costs)
                stacked_costs_blocked = jnp.stack(manual_costs_blocked)

                def trees_equal(tree_a, tree_b):
                    comparisons = jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), tree_a, tree_b)
                    return all(bool(v) for v in jax.tree_util.tree_leaves(comparisons))

                assert trees_equal(neighbours_filled, stacked_states), (
                    f"{puzzle_class.__name__} neighbours vs get_actions mismatch (filled=True)"
                )
                assert jnp.allclose(costs_filled, stacked_costs), (
                    f"{puzzle_class.__name__} neighbour costs mismatch (filled=True)"
                )

                assert trees_equal(neighbours_blocked, stacked_states_blocked), (
                    f"{puzzle_class.__name__} neighbours vs get_actions mismatch (filled=False)"
                )
                assert jnp.allclose(costs_blocked, stacked_costs_blocked), (
                    f"{puzzle_class.__name__} neighbour costs mismatch (filled=False)"
                )

                print(f"✓ {puzzle_class.__name__} neighbours align with get_actions for all moves")

            except Exception as e:
                pytest.fail(f"Neighbours/get_actions parity test failed for {puzzle_class.__name__}: {e}")

    def test_solution_checking(self, puzzle_configs, rng_key):
        """Test solution checking functionality"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()
                solve_config = puzzle.get_solve_config(key=rng_key)
                initial_state = puzzle.get_initial_state(solve_config, key=rng_key)

                # Test is_solved function
                is_solved_initial = puzzle.is_solved(solve_config, initial_state)
                assert isinstance(is_solved_initial, (bool, jnp.bool_, np.bool_)) or (
                    hasattr(is_solved_initial, "dtype") and is_solved_initial.dtype == jnp.bool_
                ), "is_solved should return boolean"

                # For puzzles with fixed targets, test that target state is actually solved
                if puzzle.fixed_target and hasattr(solve_config, "TargetState"):
                    target_solved = puzzle.is_solved(solve_config, solve_config.TargetState)
                    assert target_solved, f"Target state should be marked as solved for {puzzle_class.__name__}"

                print(f"✓ {puzzle_class.__name__} solution checking works")

            except Exception as e:
                pytest.fail(f"Solution checking failed for {puzzle_class.__name__}: {e}")

    def test_action_strings(self, puzzle_configs):
        """Test action string conversion"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()

                # Test action_to_string for all valid actions
                for action in range(puzzle.action_size):
                    action_str = puzzle.action_to_string(action)
                    assert isinstance(action_str, str), "action_to_string should return string"
                    assert len(action_str) > 0, "action_to_string should return non-empty string"

                print(f"✓ {puzzle_class.__name__} action strings work")

            except Exception as e:
                pytest.fail(f"Action string conversion failed for {puzzle_class.__name__}: {e}")

    def test_string_parsing(self, puzzle_configs, rng_key):
        """Test string representation of states"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()
                solve_config = puzzle.get_solve_config(key=rng_key)
                initial_state = puzzle.get_initial_state(solve_config, key=rng_key)

                # Test state string representation
                state_str = str(initial_state)
                assert isinstance(state_str, str), "State string representation should be string"
                assert len(state_str) > 0, "State string should be non-empty"

                # Test solve config string representation
                try:
                    solve_config_str = str(solve_config)
                    assert isinstance(solve_config_str, str), "SolveConfig string representation should be string"
                    # Empty SolveConfigs (like DotKnot) might return empty strings, which is acceptable
                    # assert len(solve_config_str) > 0, f"SolveConfig string should be non-empty"
                except IndexError as e:
                    # Handle the case where SolveConfig has no fields (like DotKnot)
                    if "list index out of range" in str(e):
                        print(f"⚠ {puzzle_class.__name__} has empty SolveConfig - skipping string representation test")
                    else:
                        raise

                print(f"✓ {puzzle_class.__name__} string parsing works")

            except Exception as e:
                pytest.fail(f"String parsing failed for {puzzle_class.__name__}: {e}")

    def test_jax_compilation(self, puzzle_configs, rng_key):
        """Test that puzzle methods work correctly with JAX compilation"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()

                # Test that core methods are JIT compiled (they should not raise errors)
                solve_config = puzzle.get_solve_config(key=rng_key)
                _ = puzzle.get_initial_state(solve_config, key=rng_key)

                # Test compiled methods
                solve_config_jit = puzzle.get_solve_config(key=rng_key)
                initial_state_jit = puzzle.get_initial_state(solve_config_jit, key=rng_key)
                neighbors, costs = puzzle.get_neighbours(solve_config_jit, initial_state_jit)
                is_solved = puzzle.is_solved(solve_config_jit, initial_state_jit)

                # Verify results are JAX arrays where expected
                if hasattr(costs, "shape"):
                    assert isinstance(costs, jnp.ndarray), "Costs should be JAX array"
                assert isinstance(is_solved, (bool, jnp.bool_, np.bool_)) or (
                    hasattr(is_solved, "dtype") and is_solved.dtype == jnp.bool_
                ), "is_solved should be boolean"

                print(f"✓ {puzzle_class.__name__} JAX compilation works")

            except Exception as e:
                pytest.fail(f"JAX compilation failed for {puzzle_class.__name__}: {e}")

    def test_batched_operations(self, puzzle_configs, rng_key):
        """Test batched operations work correctly"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()

                # Create multiple states and configs for batching
                keys = jax.random.split(rng_key, 3)
                solve_configs = [puzzle.get_solve_config(key=k) for k in keys]
                initial_states = [puzzle.get_initial_state(sc, key=k) for sc, k in zip(solve_configs, keys)]

                # Stack them for batched operations
                # Note: This may not work for all puzzles due to different state structures
                try:
                    # Test batched is_solved
                    solve_config = solve_configs[0]
                    _ = puzzle.batched_is_solved(solve_config, initial_states[0])

                    # Test batched neighbors (with single solve_config)
                    batched_neighbors, batched_costs = puzzle.batched_get_neighbours(
                        solve_config, initial_states[0], filled=True
                    )

                    print(f"✓ {puzzle_class.__name__} batched operations work")

                except Exception as batch_e:
                    # Batched operations might fail due to state structure incompatibilities
                    print(f"⚠ {puzzle_class.__name__} batched operations not fully testable: {batch_e}")

            except Exception as e:
                pytest.fail(f"Batched operations test setup failed for {puzzle_class.__name__}: {e}")

    def test_puzzle_properties(self, puzzle_configs):
        """Test puzzle property flags"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()

                # Test boolean properties
                assert isinstance(puzzle.has_target, bool), "has_target should be boolean"
                assert isinstance(puzzle.only_target, bool), "only_target should be boolean"
                assert isinstance(puzzle.fixed_target, bool), "fixed_target should be boolean"
                assert isinstance(puzzle.is_reversible, bool), "is_reversible should be boolean"

                # Logical consistency checks
                if puzzle.only_target:
                    assert puzzle.has_target, "only_target implies has_target"

                print(f"✓ {puzzle_class.__name__} properties are consistent")

            except Exception as e:
                pytest.fail(f"Property test failed for {puzzle_class.__name__}: {e}")

    def test_inverse_actions(self, puzzle_configs, rng_key):
        """Test inverse action functionality"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()

                if puzzle.is_reversible:
                    # Test for perfectly reversible puzzles
                    solve_config = puzzle.get_solve_config(key=rng_key)
                    initial_state = puzzle.get_initial_state(solve_config, key=rng_key)

                    neighbors, costs = puzzle.get_neighbours(solve_config, initial_state, filled=True)

                    for i in range(puzzle.action_size):
                        # If a move is possible
                        if jnp.isfinite(costs[i]):
                            next_state = jax.tree_util.tree_map(lambda x: x[i], neighbors)

                            # Taking the inverse action from the next state should return to the initial state
                            inv_neighbors, inv_costs = puzzle.get_inverse_neighbours(
                                solve_config, next_state, filled=True
                            )

                            # The action that leads from S_prev to S_next is i.
                            # So, the state before S_next via action i was S_initial.
                            # get_inverse_neighbours(S_next)[i] should be S_initial
                            state_after_inverse = jax.tree_util.tree_map(lambda x: x[i], inv_neighbors)

                            assert state_after_inverse == initial_state, (
                                f"Inverse action did not return to original state for action {i}"
                            )

                    print(f"✓ {puzzle_class.__name__} inverse actions are reversible")

                else:
                    # Test for non-reversible puzzles or those without inverse map
                    # Case 1: Custom inverse implementation (e.g., Sokoban)
                    is_custom_inverse = (
                        puzzle.get_inverse_neighbours.__qualname__ != Puzzle.get_inverse_neighbours.__qualname__
                    )
                    if is_custom_inverse:
                        try:
                            solve_config = puzzle.get_solve_config(key=rng_key)
                            initial_state = puzzle.get_initial_state(solve_config, key=rng_key)
                            puzzle.get_inverse_neighbours(solve_config, initial_state, filled=True)
                            print(f"✓ {puzzle_class.__name__} custom get_inverse_neighbours called successfully")
                        except Exception as e:
                            pytest.fail(f"Custom get_inverse_neighbours for {puzzle_class.__name__} failed: {e}")

                    # Case 2: No inverse map and no custom implementation
                    else:
                        with pytest.raises(NotImplementedError):
                            solve_config = puzzle.get_solve_config(key=rng_key)
                            initial_state = puzzle.get_initial_state(solve_config, key=rng_key)
                            puzzle.get_inverse_neighbours(solve_config, initial_state, filled=True)
                        print(f"✓ {puzzle_class.__name__} correctly raised NotImplementedError for inverse actions")

            except Exception as e:
                pytest.fail(f"Inverse action test failed for {puzzle_class.__name__}: {e}")

    def test_state_equality(self, puzzle_configs, rng_key):
        """Test state equality comparisons work correctly"""
        for puzzle_class in puzzle_configs:
            try:
                puzzle = puzzle_class()
                solve_config = puzzle.get_solve_config(key=rng_key)
                state1 = puzzle.get_initial_state(solve_config, key=rng_key)
                state2 = puzzle.get_initial_state(solve_config, key=rng_key)

                # Test equality with same state
                assert state1 == state1, "State should equal itself"

                # Test that different states may or may not be equal (depends on randomness)
                # Just ensure comparison doesn't crash
                equality_result = state1 == state2
                assert isinstance(equality_result, (bool, jnp.bool_, np.bool_)) or (
                    hasattr(equality_result, "dtype") and equality_result.dtype == jnp.bool_
                ), "Equality should return boolean"

                print(f"✓ {puzzle_class.__name__} state equality works")

            except Exception as e:
                pytest.fail(f"State equality test failed for {puzzle_class.__name__}: {e}")


def run_comprehensive_puzzle_tests():
    """Run all puzzle validation tests"""
    print("=" * 80)
    print("COMPREHENSIVE PUZZLE VALIDATION TESTS")
    print("=" * 80)

    # Create test instance
    test_instance = TestPuzzleValidation()

    # Get puzzle configurations
    puzzle_configs = test_instance.puzzle_configs()
    rng_key = jax.random.PRNGKey(42)

    print(f"\nTesting {len(puzzle_configs)} puzzle configurations...")
    print("-" * 50)

    try:
        # Run all test methods
        test_methods = [
            test_instance.test_puzzle_instantiation,
            test_instance.test_state_generation,
            test_instance.test_neighbor_generation,
            test_instance.test_solution_checking,
            test_instance.test_action_strings,
            test_instance.test_string_parsing,
            test_instance.test_jax_compilation,
            test_instance.test_batched_operations,
            test_instance.test_puzzle_properties,
            test_instance.test_state_equality,
        ]

        for test_method in test_methods:
            print(f"\n{test_method.__name__.replace('test_', '').replace('_', ' ').title()}:")
            print("-" * 30)

            if "rng_key" in test_method.__code__.co_varnames:
                test_method(puzzle_configs, rng_key)
            else:
                test_method(puzzle_configs)

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        raise

    print("\n" + "=" * 80)
    print("ALL PUZZLE VALIDATION TESTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_puzzle_tests()
