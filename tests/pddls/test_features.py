from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from puxle.puzzles.adapters.pddl import PDDL

BASE_DIR = Path(__file__).resolve().parents[1] / "pddl_data"


class TestPDDLFeatures:
    def test_constants(self):
        domain_file = str(BASE_DIR / "constants" / "domain.pddl")
        problem_file = str(BASE_DIR / "constants" / "problem.pddl")

        puzzle = PDDL(domain_file, problem_file)

        # In the constants fixture, c1 is a domain constant and o1 is a problem object.

        atoms = set(puzzle.grounded_atoms)
        assert "(has c1)" in atoms
        assert "(has o1)" in atoms

    def test_negative_preconditions(self):
        domain_file = str(BASE_DIR / "negative" / "domain.pddl")
        problem_file = str(BASE_DIR / "negative" / "problem.pddl")

        puzzle = PDDL(domain_file, problem_file)
        rng_key = jax.random.PRNGKey(0)
        solve_config, init_state = puzzle.get_inits(rng_key)

        # initial state: empty
        # action: toggle-on ?s (pre: (not (on ?s))) -> should be applicable
        # action: toggle-off ?s (pre: (on ?s)) -> should NOT be applicable

        neighbors, costs = puzzle.get_neighbours(solve_config, init_state, filled=True)

        toggle_on_idx = -1
        toggle_off_idx = -1

        for i, action in enumerate(puzzle.grounded_actions):
            if action["name"] == "toggle-on":
                toggle_on_idx = i
            elif action["name"] == "toggle-off":
                toggle_off_idx = i

        assert toggle_on_idx >= 0
        assert toggle_off_idx >= 0

        # If 'not' were stripped into a positive precondition (on ?s), toggle-on would be inapplicable in the empty initial state.

        cost_on = costs[toggle_on_idx]
        assert jnp.isfinite(cost_on), (
            "toggle-on should be applicable in empty state (requires not on)"
        )

    def test_equality(self):
        domain_file = str(BASE_DIR / "equality" / "domain.pddl")
        problem_file = str(BASE_DIR / "equality" / "problem.pddl")

        puzzle = PDDL(domain_file, problem_file)

        # The equality fixture forbids self-loops via (not (= ?from ?to)).

        found_self_loop = False
        for action in puzzle.grounded_actions:
            if action["name"] == "move":
                params = action["parameters"]
                if params[0] == params[1]:
                    found_self_loop = True
                    break

        assert not found_self_loop, (
            "Equality constraint 'not =' should prevent self-loop action"
        )

    def test_zero_arity_formatting(self):
        domain_file = str(BASE_DIR / "zero_arity" / "domain.pddl")
        problem_file = str(BASE_DIR / "zero_arity" / "problem.pddl")

        puzzle = PDDL(domain_file, problem_file)

        assert "(light-on)" in puzzle.grounded_atoms
        assert "(light-on )" not in puzzle.grounded_atoms

    def test_either_type_grounding(self):
        domain_file = str(BASE_DIR / "either_move" / "domain.pddl")
        problem_file = str(BASE_DIR / "either_move" / "problem.pddl")

        puzzle = PDDL(domain_file, problem_file)

        # Union type `(either room hall)` should cover both object sets.
        assert "(at r1)" in set(puzzle.grounded_atoms)
        assert "(at h1)" in set(puzzle.grounded_atoms)

        grounded_action_strs = {
            f"({a['name']} {' '.join(a['parameters'])})"
            for a in puzzle.grounded_actions
        }
        assert "(mark r1)" in grounded_action_strs
        assert "(mark h1)" in grounded_action_strs

    def test_disjunctive_precondition_is_rejected(self):
        domain_file = str(BASE_DIR / "disjunctive" / "domain.pddl")
        problem_file = str(BASE_DIR / "disjunctive" / "problem.pddl")

        with pytest.raises(ValueError, match="Disjunctive preconditions"):
            PDDL(domain_file, problem_file)

    def test_negative_goal_is_rejected(self):
        domain_file = str(BASE_DIR / "negative_goal" / "domain.pddl")
        problem_file = str(BASE_DIR / "negative_goal" / "problem.pddl")

        with pytest.raises(ValueError, match="Negative goals"):
            PDDL(domain_file, problem_file)
