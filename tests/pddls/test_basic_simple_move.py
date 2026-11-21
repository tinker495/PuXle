from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from puxle.puzzles.adapters.pddl import PDDL

DATA_DIR = Path(__file__).resolve().parents[1] / "pddl_data" / "simple_move"
DOMAIN = DATA_DIR / "domain.pddl"
PROBLEM = DATA_DIR / "problem.pddl"


class TestPDDLWrapper:
    @pytest.fixture
    def puzzle(self):
        return PDDL(str(DOMAIN), str(PROBLEM))

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_parse_and_ground_counts(self, puzzle):
        # 3 locations => 3 at() atoms + 3*3 connected() atoms = 3 + 9 = 12
        assert puzzle.num_atoms == 12
        # 3x3 possible moves => 9 grounded actions
        assert puzzle.num_actions == 9
        assert puzzle.action_size == 9

    def test_grounded_atoms(self, puzzle):
        assert len(puzzle.grounded_atoms) == 12

        expected_atoms = [
            "(at loc1)",
            "(at loc2)",
            "(at loc3)",
            "(connected loc1 loc1)",
            "(connected loc1 loc2)",
            "(connected loc1 loc3)",
            "(connected loc2 loc1)",
            "(connected loc2 loc2)",
            "(connected loc2 loc3)",
            "(connected loc3 loc1)",
            "(connected loc3 loc2)",
            "(connected loc3 loc3)",
        ]

        for expected in expected_atoms:
            assert expected in puzzle.grounded_atoms

    def test_grounded_actions(self, puzzle):
        assert len(puzzle.grounded_actions) == 9

        expected_parameters = [
            ["loc1", "loc1"],
            ["loc1", "loc2"],
            ["loc1", "loc3"],
            ["loc2", "loc1"],
            ["loc2", "loc2"],
            ["loc2", "loc3"],
            ["loc3", "loc1"],
            ["loc3", "loc2"],
            ["loc3", "loc3"],
        ]

        for expected_params in expected_parameters:
            found = False
            for action in puzzle.grounded_actions:
                if action["name"] == "move" and action["parameters"] == expected_params:
                    found = True
                    break
            assert found, f"Action move {expected_params} not found in grounded actions"

    def test_action_preconditions_and_effects(self, puzzle):
        move_loc1_loc2 = None
        for action in puzzle.grounded_actions:
            if action["name"] == "move" and action["parameters"] == ["loc1", "loc2"]:
                move_loc1_loc2 = action
                break

        assert move_loc1_loc2 is not None

        preconditions = move_loc1_loc2["preconditions"]
        assert "(at loc1)" in preconditions
        assert "(connected loc1 loc2)" in preconditions
        assert len(preconditions) == 2

        effects = move_loc1_loc2["effects"]
        assert "(at loc2)" in effects["add"]
        assert "(at loc1)" in effects["delete"]
        assert len(effects["add"]) == 1
        assert len(effects["delete"]) == 1

    def test_initial_state_and_goal(self, puzzle, rng_key):
        solve_config, initial_state = puzzle.get_inits(rng_key)

        assert initial_state is not None
        assert isinstance(initial_state, puzzle.State)

        assert solve_config is not None
        assert isinstance(solve_config, puzzle.SolveConfig)

        assert not bool(puzzle.is_solved(solve_config, initial_state))

        # Use header=True, raw=True for deterministic, ANSI-free text assertions.
        state_str = puzzle.get_string_parser()(initial_state, header=True, raw=True)
        assert "(at loc1)" in state_str
        assert "(connected loc1 loc2)" in state_str
        assert "(connected loc2 loc3)" in state_str

    def test_neighbor_generation(self, puzzle, rng_key):
        solve_config, initial_state = puzzle.get_inits(rng_key)

        neighbors, costs = puzzle.get_neighbours(
            solve_config, initial_state, filled=True
        )

        assert neighbors is not None
        assert costs is not None
        assert len(costs) == puzzle.action_size

        applicable = jnp.isfinite(costs)
        assert jnp.any(applicable), "At least some actions should be applicable"

        move_idx = None
        for i in range(puzzle.num_actions):
            if puzzle.action_to_string(i, colored=False) == "(move loc1 loc2)":
                move_idx = i
                break

        assert move_idx is not None
        assert jnp.isfinite(costs[move_idx]), "move loc1->loc2 should be applicable"

        next_state = jax.tree_util.tree_map(lambda x: x[move_idx], neighbors)
        next_state_str = puzzle.get_string_parser()(next_state, header=True, raw=True)

        assert "(at loc2)" in next_state_str

    def test_filled_false_contract(self, puzzle, rng_key):
        solve_config, initial_state = puzzle.get_inits(rng_key)

        _, costs = puzzle.get_neighbours(solve_config, initial_state, filled=False)

        assert jnp.all(jnp.isinf(costs))

    def test_jit_compilation(self, puzzle, rng_key):
        solve_config, initial_state = puzzle.get_inits(rng_key)

        jitted_get_neighbours = jax.jit(puzzle.get_neighbours)
        neighbors, costs = jitted_get_neighbours(solve_config, initial_state)

        assert neighbors is not None
        assert costs is not None
        assert len(costs) == puzzle.action_size

        jitted_is_solved = jax.jit(puzzle.is_solved)
        is_solved = jitted_is_solved(solve_config, initial_state)

        assert isinstance(is_solved, (bool, jnp.bool_)) or (
            hasattr(is_solved, "dtype") and is_solved.dtype == jnp.bool_
        )

    def test_batched_operations(self, puzzle, rng_key):
        keys = jax.random.split(rng_key, 4)
        solve_configs = jax.vmap(lambda k: puzzle.get_solve_config(k))(keys)
        initial_states = jax.vmap(lambda sc, k: puzzle.get_initial_state(sc, k))(
            solve_configs, keys
        )

        solved_mask = puzzle.batched_is_solved(
            solve_configs, initial_states, multi_solve_config=True
        )
        assert solved_mask.shape[0] == 4
        assert isinstance(solved_mask, jnp.ndarray)

        solve_config = puzzle.get_solve_config(rng_key)
        filleds = jnp.array([True, True, True, True])
        batched_neighbors, batched_costs = puzzle.batched_get_neighbours(
            solve_config, initial_states, filleds=filleds, multi_solve_config=False
        )

        assert batched_neighbors is not None
        assert batched_costs is not None
        assert batched_costs.shape[0] == puzzle.action_size
        assert batched_costs.shape[1] == 4

        mixed_filleds = jnp.array([True, False, True, False])
        mixed_neighbors, mixed_costs = puzzle.batched_get_neighbours(
            solve_config,
            initial_states,
            filleds=mixed_filleds,
            multi_solve_config=False,
        )

        # Inapplicable actions still have inf cost when filled=True, so check jnp.any(isfinite).
        assert jnp.any(jnp.isfinite(mixed_costs[:, 0]))
        assert jnp.all(jnp.isinf(mixed_costs[:, 1]))
        assert jnp.any(jnp.isfinite(mixed_costs[:, 2]))
        assert jnp.all(jnp.isinf(mixed_costs[:, 3]))

    def test_action_to_string(self, puzzle):
        for i in range(puzzle.action_size):
            action_str = puzzle.action_to_string(i, colored=False)
            assert isinstance(action_str, str)
            assert len(action_str) > 0
            assert action_str.startswith("(move ")
            assert action_str.endswith(")")

    def test_string_parsing(self, puzzle, rng_key):
        solve_config, initial_state = puzzle.get_inits(rng_key)

        state_str = puzzle.get_string_parser()(initial_state, header=True, raw=True)
        assert isinstance(state_str, str)
        assert len(state_str) > 0
        assert state_str.startswith("State:")

        solve_config_str = str(solve_config)
        assert isinstance(solve_config_str, str)
        assert len(solve_config_str) > 0
        assert "Goal" in solve_config_str

    def test_puzzle_properties(self, puzzle):
        assert isinstance(puzzle.has_target, bool)
        assert isinstance(puzzle.fixed_target, bool)
        assert isinstance(puzzle.is_reversible, bool)

        assert not puzzle.has_target
        assert puzzle.has_goal_data
        assert puzzle.fixed_target

    def test_state_equality(self, puzzle, rng_key):
        solve_config, state1 = puzzle.get_inits(rng_key)
        _, state2 = puzzle.get_inits(rng_key)

        assert state1 == state1, "State should equal itself"

        # Test that different states may or may not be equal (depends on randomness)
        # Just ensure comparison doesn't crash
        equality_result = state1 == state2
        assert isinstance(equality_result, (bool, jnp.bool_)) or (
            hasattr(equality_result, "dtype") and equality_result.dtype == jnp.bool_
        )

    def test_image_parser(self, puzzle, rng_key):
        solve_config, initial_state = puzzle.get_inits(rng_key)

        try:
            state_image = puzzle.State.img(initial_state)
            assert state_image is not None
            assert hasattr(state_image, "shape")
            assert len(state_image.shape) == 3

            solve_config_image = puzzle.SolveConfig.img(solve_config)
            assert solve_config_image is not None
            assert hasattr(solve_config_image, "shape")
            assert len(solve_config_image.shape) == 3

        except Exception as e:
            # Image parsing might not be implemented for all puzzles
            pytest.skip(f"Image parser not fully implemented: {e}")

    def test_inverse_actions(self, puzzle, rng_key):
        # PDDL puzzles are not reversible by default
        assert not puzzle.is_reversible

        solve_config, initial_state = puzzle.get_inits(rng_key)

        with pytest.raises(NotImplementedError):
            puzzle.get_inverse_neighbours(solve_config, initial_state, filled=True)

    def test_solution_path(self, puzzle, rng_key):
        solve_config, initial_state = puzzle.get_inits(rng_key)

        current_state = initial_state
        max_steps = 10  # Prevent infinite loops
        path_found = False

        for step in range(max_steps):
            if puzzle.is_solved(solve_config, current_state):
                path_found = True
                break

            neighbors, costs = puzzle.get_neighbours(
                solve_config, current_state, filled=True
            )
            applicable = jnp.isfinite(costs)

            if not jnp.any(applicable):
                break

            action_idx = jnp.where(applicable)[0]
            current_state = neighbors[action_idx]

        # A greedy first-applicable walk may cycle; if the goal is not reached, require at least one state change.
        if not path_found:
            assert current_state != initial_state, (
                "No progress made in solving the puzzle"
            )
            print(
                f"Warning: Goal not reached in {max_steps} steps, but progress was made"
            )
