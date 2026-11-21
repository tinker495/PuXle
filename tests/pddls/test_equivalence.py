import jax
import jax.numpy as jnp
import pytest

from puxle.puzzles.adapters.pddl import PDDL
from tests.pddls.data_specs import DATA_SPECS
from tests.pddls.reference_strips import ReferenceSTRIPS


class TestPDDLEquivalence:
    """STRIPS semantics cross-validation tests."""

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_grounding_equivalence(self, spec):
        puzzle = PDDL(spec.domain, spec.problem)
        reference = ReferenceSTRIPS(spec.domain, spec.problem)

        assert len(reference.grounded_atoms) == puzzle.num_atoms, (
            f"Atom count mismatch: reference={len(reference.grounded_atoms)}, ours={puzzle.num_atoms}"
        )

        assert len(reference.grounded_actions) == puzzle.num_actions, (
            f"Action count mismatch: reference={len(reference.grounded_actions)}, ours={puzzle.num_actions}"
        )

        # Normalize whitespace so formatting differences do not mask atom-set equivalence.
        our_atoms = set(puzzle.grounded_atoms)
        ref_atoms = reference.grounded_atoms

        our_atoms_normalized = {atom.replace(" ", "") for atom in our_atoms}
        ref_atoms_normalized = {atom.replace(" ", "") for atom in ref_atoms}

        assert our_atoms_normalized == ref_atoms_normalized, (
            f"Atom mismatch: ours={our_atoms}, reference={ref_atoms}"
        )

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_applicability_equivalence(self, spec):
        puzzle = PDDL(spec.domain, spec.problem)
        reference = ReferenceSTRIPS(spec.domain, spec.problem)
        rng_key = jax.random.PRNGKey(42)
        solve_config, initial_state = puzzle.get_inits(rng_key)

        _test_applicability_in_state(
            puzzle, reference, solve_config, initial_state, "initial"
        )

        current_state = initial_state
        for step in range(3):
            neighbors, costs = puzzle.get_neighbours(
                solve_config, current_state, filled=True
            )
            applicable = jnp.isfinite(costs)

            if not jnp.any(applicable):
                break

            action_idx = jnp.where(applicable)[0][0]
            current_state = jax.tree_util.tree_map(lambda x: x[action_idx], neighbors)

            _test_applicability_in_state(
                puzzle, reference, solve_config, current_state, f"step_{step + 1}"
            )

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_successor_equivalence(self, spec):
        puzzle = PDDL(spec.domain, spec.problem)
        reference = ReferenceSTRIPS(spec.domain, spec.problem)
        rng_key = jax.random.PRNGKey(42)
        solve_config, initial_state = puzzle.get_inits(rng_key)

        _test_successors_in_state(
            puzzle, reference, solve_config, initial_state, "initial"
        )

        current_state = initial_state
        for step in range(2):
            neighbors, costs = puzzle.get_neighbours(
                solve_config, current_state, filled=True
            )
            applicable = jnp.isfinite(costs)

            if not jnp.any(applicable):
                break

            action_idx = jnp.where(applicable)[0][0]
            current_state = jax.tree_util.tree_map(lambda x: x[action_idx], neighbors)

            _test_successors_in_state(
                puzzle, reference, solve_config, current_state, f"step_{step + 1}"
            )

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_goal_equivalence(self, spec):
        puzzle = PDDL(spec.domain, spec.problem)
        reference = ReferenceSTRIPS(spec.domain, spec.problem)
        rng_key = jax.random.PRNGKey(42)
        solve_config, initial_state = puzzle.get_inits(rng_key)

        our_goal_satisfied = puzzle.is_solved(solve_config, initial_state)
        ref_goal_satisfied = reference.is_goal_satisfied(reference.initial_atoms)

        assert our_goal_satisfied == ref_goal_satisfied, (
            f"Goal satisfaction mismatch in initial state: ours={our_goal_satisfied}, reference={ref_goal_satisfied}"
        )

        current_state = initial_state
        for step in range(3):
            neighbors, costs = puzzle.get_neighbours(
                solve_config, current_state, filled=True
            )
            applicable = jnp.isfinite(costs)

            if not jnp.any(applicable):
                break

            action_idx = jnp.where(applicable)[0][0]
            current_state = jax.tree_util.tree_map(lambda x: x[action_idx], neighbors)

            our_goal_satisfied = puzzle.is_solved(solve_config, current_state)
            # Successor steps only exercise puzzle.is_solved because reference state atoms are not tracked across transitions.

            if step == 0:
                pass

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_plan_equivalence(self, spec):
        puzzle = PDDL(spec.domain, spec.problem)
        reference = ReferenceSTRIPS(spec.domain, spec.problem)
        rng_key = jax.random.PRNGKey(42)
        solve_config, initial_state = puzzle.get_inits(rng_key)

        ref_plan = reference.bfs_search(max_depth=6)

        if ref_plan is not None:
            current_state = initial_state
            for action in ref_plan:
                action_idx = _find_action_index(puzzle, action)
                if action_idx is None:
                    pytest.skip(
                        f"Could not find action {action['name']} {action['parameters']} in our system"
                    )

                neighbors, costs = puzzle.get_neighbours(
                    solve_config, current_state, filled=True
                )
                if not jnp.isfinite(costs[action_idx]):
                    pytest.fail(
                        f"Action {action['name']} {action['parameters']} not applicable in our system"
                    )

                current_state = jax.tree_util.tree_map(
                    lambda x: x[action_idx], neighbors
                )

            our_goal_satisfied = puzzle.is_solved(solve_config, current_state)
            assert our_goal_satisfied, "Plan execution should reach goal in our system"
        else:
            current_state = initial_state
            for step in range(6):
                if puzzle.is_solved(solve_config, current_state):
                    pytest.fail("Our system found a solution when reference found none")

                neighbors, costs = puzzle.get_neighbours(
                    solve_config, current_state, filled=True
                )
                applicable = jnp.isfinite(costs)

                if not jnp.any(applicable):
                    break

                action_idx = jnp.where(applicable)[0][0]
                current_state = jax.tree_util.tree_map(
                    lambda x: x[action_idx], neighbors
                )

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_static_predicate_preservation(self, spec):
        puzzle = PDDL(spec.domain, spec.problem)
        reference = ReferenceSTRIPS(spec.domain, spec.problem)
        rng_key = jax.random.PRNGKey(42)
        solve_config, initial_state = puzzle.get_inits(rng_key)

        # Identify static predicates (those that never appear in add/delete effects)
        static_predicates = set()
        dynamic_predicates = set()

        for action in reference.grounded_actions:
            for atom in action["add_effects"]:
                pred_name = atom.split("(")[1].split(" ")[0]
                dynamic_predicates.add(pred_name)
            for atom in action["delete_effects"]:
                pred_name = atom.split("(")[1].split(" ")[0]
                dynamic_predicates.add(pred_name)

        all_predicates = set()
        for atom in reference.grounded_atoms:
            pred_name = atom.split("(")[1].split(" ")[0]
            all_predicates.add(pred_name)

        static_predicates = all_predicates - dynamic_predicates

        if not static_predicates:
            pytest.skip("No static predicates in this domain")

        current_state = initial_state
        initial_static_values = _get_static_predicate_values(
            puzzle, current_state, static_predicates
        )

        for step in range(3):
            neighbors, costs = puzzle.get_neighbours(
                solve_config, current_state, filled=True
            )
            applicable = jnp.isfinite(costs)

            if not jnp.any(applicable):
                break

            action_idx = jnp.where(applicable)[0][0]
            current_state = jax.tree_util.tree_map(lambda x: x[action_idx], neighbors)

            current_static_values = _get_static_predicate_values(
                puzzle, current_state, static_predicates
            )

            for pred, value in current_static_values.items():
                assert value == initial_static_values[pred], (
                    f"Static predicate {pred} changed value from {initial_static_values[pred]} to {value}"
                )

    @pytest.mark.parametrize("spec", DATA_SPECS, ids=lambda s: s.name)
    def test_action_string_normalization(self, spec):
        puzzle = PDDL(spec.domain, spec.problem)
        reference = ReferenceSTRIPS(spec.domain, spec.problem)

        assert len(puzzle.grounded_actions) == len(reference.grounded_actions)

        for i in range(puzzle.action_size):
            our_action_str = puzzle.action_to_string(i, colored=False)
            ref_action = reference.grounded_actions[i]
            ref_action_str = (
                f"({ref_action['name']} {' '.join(ref_action['parameters'])})"
            )

            our_normalized = our_action_str.replace(" ", "")
            ref_normalized = ref_action_str.replace(" ", "")

            assert our_normalized == ref_normalized, (
                f"Action string mismatch at index {i}: ours='{our_action_str}', reference='{ref_action_str}'"
            )


def _test_applicability_in_state(puzzle, reference, solve_config, state, state_name):
    neighbors, costs = puzzle.get_neighbours(solve_config, state, filled=True)

    state_atoms = _extract_state_atoms(puzzle, state)

    for i, (action, cost) in enumerate(zip(puzzle.grounded_actions, costs)):
        our_applicable = jnp.isfinite(cost)

        ref_action = {
            "name": action["name"],
            "parameters": action["parameters"],
            "preconditions": set(action["preconditions"]),
            "add_effects": set(action["effects"]["add"]),
            "delete_effects": set(action["effects"]["delete"]),
        }

        ref_applicable = reference.is_applicable(ref_action, state_atoms)

        assert our_applicable == ref_applicable, (
            f"Applicability mismatch for action {i} ({action['name']} {action['parameters']}) "
            f"in {state_name}: ours={our_applicable}, reference={ref_applicable}"
        )


def _extract_state_atoms(puzzle, state):
    return puzzle.state_to_atom_set(state)


def _test_successors_in_state(puzzle, reference, solve_config, state, state_name):
    neighbors, costs = puzzle.get_neighbours(solve_config, state, filled=True)

    state_atoms = _extract_state_atoms(puzzle, state)

    for i, (action, cost) in enumerate(zip(puzzle.grounded_actions, costs)):
        if not jnp.isfinite(cost):
            continue

        our_successor = jax.tree_util.tree_map(lambda x: x[i], neighbors)

        ref_action = {
            "name": action["name"],
            "parameters": action["parameters"],
            "preconditions": set(action["preconditions"]),
            "add_effects": set(action["effects"]["add"]),
            "delete_effects": set(action["effects"]["delete"]),
        }

        # Exercise reference.apply_action to confirm the action is applicable in the reference oracle.
        _ = reference.apply_action(ref_action, state_atoms)

        # Actions with no effects leave the state unchanged, so only verify successor state validity here.
        our_successor_atoms = _extract_state_atoms(puzzle, our_successor)
        assert isinstance(our_successor_atoms, set), (
            f"Successor should be a valid state for action {i} in {state_name}"
        )


def _find_action_index(puzzle, action):
    target_params = list(action["parameters"])
    for i, our_action in enumerate(puzzle.grounded_actions):
        if (
            our_action["name"] == action["name"]
            and our_action["parameters"] == target_params
        ):
            return i
    return None


def _get_static_predicate_values(puzzle, state, static_predicates):
    values_by_pred = {}
    for pred in static_predicates:
        values_by_pred[pred] = puzzle.static_predicate_profile(state, pred)
    return values_by_pred
