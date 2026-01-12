import jax
import jax.numpy as jnp
import pytest

from puxle.pddls.pddl import PDDL
from tests.pddls.data_specs import DATA_SPECS


class TestPDDLParametric:
    """Parametric tests across all PDDL domains."""

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_parse_and_ground_counts_param(self, spec):
        """Test that parsing and grounding produces correct counts across all domains."""
        puzzle = PDDL(spec.domain, spec.problem)
        assert (
            puzzle.num_atoms == spec.expected_atoms
        ), f"Expected {spec.expected_atoms} atoms, got {puzzle.num_atoms}"
        assert (
            puzzle.num_actions == spec.expected_actions
        ), f"Expected {spec.expected_actions} actions, got {puzzle.num_actions}"
        assert (
            puzzle.action_size == spec.expected_actions
        ), f"Expected action_size {spec.expected_actions}, got {puzzle.action_size}"

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_grounded_atoms_param(self, spec):
        """Test that atoms are grounded correctly across all domains."""
        puzzle = PDDL(spec.domain, spec.problem)
        assert len(puzzle.grounded_atoms) == spec.expected_atoms

        # Check that all grounded atoms are valid strings
        for atom in puzzle.grounded_atoms:
            assert isinstance(atom, str)
            assert len(atom) > 0
            assert atom.startswith("(") and atom.endswith(")")

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_grounded_actions_param(self, spec):
        """Test that actions are grounded correctly across all domains."""
        puzzle = PDDL(spec.domain, spec.problem)
        assert len(puzzle.grounded_actions) == spec.expected_actions

        # Check that all grounded actions have required structure
        for action in puzzle.grounded_actions:
            assert "name" in action
            assert "parameters" in action
            assert "preconditions" in action
            assert "effects" in action
            assert isinstance(action["name"], str)
            assert isinstance(action["parameters"], list)
            assert isinstance(action["preconditions"], list)
            assert isinstance(action["effects"], dict)
            assert "add" in action["effects"] and "delete" in action["effects"]

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_neighbours_and_filled_contract_param(self, spec):
        """Test neighbor generation and filled contract across all domains."""
        puzzle = PDDL(spec.domain, spec.problem)
        rng_key = jax.random.PRNGKey(42)
        solve_config, initial_state = puzzle.get_inits(rng_key)

        # filled=False => all inf
        _, costs_empty = puzzle.get_neighbours(solve_config, initial_state, filled=False)
        assert jnp.all(
            jnp.isinf(costs_empty)
        ), f"filled=False should return all infinite costs for {spec.name}"

        # filled=True => check applicability based on solvability
        _, costs = puzzle.get_neighbours(solve_config, initial_state, filled=True)
        if not puzzle.is_solved(solve_config, initial_state) and spec.solvable:
            assert jnp.any(
                jnp.isfinite(costs)
            ), f"Solvable domain {spec.name} should have at least one applicable action"

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_jit_and_batched_param(self, spec):
        """Test JIT compilation and batched operations across all domains."""
        puzzle = PDDL(spec.domain, spec.problem)
        rng_key = jax.random.PRNGKey(42)
        solve_config, initial_state = puzzle.get_inits(rng_key)

        # Test JIT compilation
        jitted_get_neighbours = jax.jit(puzzle.get_neighbours)
        neighbors, costs = jitted_get_neighbours(solve_config, initial_state, filled=True)
        assert neighbors is not None and costs is not None
        assert len(costs) == puzzle.action_size

        # Test JIT compiled is_solved
        jitted_is_solved = jax.jit(puzzle.is_solved)
        is_solved = jitted_is_solved(solve_config, initial_state)
        assert isinstance(is_solved, (bool, jnp.bool_)) or (
            hasattr(is_solved, "dtype") and is_solved.dtype == jnp.bool_
        )

        # Test batched operations
        keys = jax.random.split(rng_key, 4)
        solve_configs = jax.vmap(lambda k: puzzle.get_solve_config(k))(keys)
        initial_states = jax.vmap(lambda sc, k: puzzle.get_initial_state(sc, k))(
            solve_configs, keys
        )

        # Test batched is_solved
        solved_mask = puzzle.batched_is_solved(
            solve_configs, initial_states, multi_solve_config=True
        )
        assert solved_mask.shape[0] == 4
        assert isinstance(solved_mask, jnp.ndarray)

        # Test batched neighbors
        filleds = jnp.array([True, True, True, True])
        batched_neighbors, batched_costs = puzzle.batched_get_neighbours(
            solve_configs, initial_states, filleds=filleds, multi_solve_config=True
        )
        assert batched_neighbors is not None and batched_costs is not None
        assert batched_costs.shape[0] == puzzle.action_size
        assert batched_costs.shape[1] == 4

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_solution_path_param(self, spec):
        """Test solution path finding across all domains."""
        puzzle = PDDL(spec.domain, spec.problem)
        rng_key = jax.random.PRNGKey(42)
        solve_config, initial_state = puzzle.get_inits(rng_key)

        current_state = initial_state
        solved = False

        for step in range(spec.max_solution_steps):
            if puzzle.is_solved(solve_config, current_state):
                solved = True
                break

            neighbors, costs = puzzle.get_neighbours(solve_config, current_state, filled=True)
            applicable = jnp.isfinite(costs)

            if not jnp.any(applicable):
                break  # No more applicable actions

            # Take first applicable action
            action_idx = jnp.where(applicable)[0][0]
            current_state = neighbors[action_idx]

        if spec.solvable:
            # For solvable domains, we should either reach the goal or make progress
            assert (
                solved or current_state != initial_state
            ), f"Solvable domain {spec.name} should either reach goal or make progress"
        else:
            # For unsolvable domains, we should never reach the goal
            assert not solved, f"Unsolvable domain {spec.name} should never reach the goal"

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_action_to_string_param(self, spec):
        """Test action string conversion across all domains."""
        puzzle = PDDL(spec.domain, spec.problem)

        for i in range(puzzle.action_size):
            action_str = puzzle.action_to_string(i, colored=False)
            assert isinstance(action_str, str)
            assert len(action_str) > 0
            assert action_str.startswith("(") and action_str.endswith(")")

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_puzzle_properties_param(self, spec):
        """Test puzzle property flags across all domains."""
        puzzle = PDDL(spec.domain, spec.problem)

        # Test boolean properties
        assert isinstance(puzzle.has_target, bool)
        assert isinstance(puzzle.only_target, bool)
        assert isinstance(puzzle.fixed_target, bool)
        assert isinstance(puzzle.is_reversible, bool)

        # PDDL puzzles should have targets (goal masks)
        assert puzzle.has_target
        assert not puzzle.only_target  # Has goal mask, not just target state
        assert puzzle.fixed_target
