
from pathlib import Path
import pytest
import jax
import jax.numpy as jnp
from puxle.pddls.pddl import PDDL

BASE_DIR = Path(__file__).resolve().parents[1] / "pddl_data"

class TestPDDLFeatures:
    def test_constants(self):
        domain_file = str(BASE_DIR / "constants" / "domain.pddl")
        problem_file = str(BASE_DIR / "constants" / "problem.pddl")
        
        # This should succeed if constants are handled, or fail if not
        puzzle = PDDL(domain_file, problem_file)
        
        # Verify constants are in objects map or grounded atoms
        # c1 is a constant, o1 is an object
        # Atoms should include (has c1) and (has o1)
        
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
        
        # Find actions
        toggle_on_idx = -1
        toggle_off_idx = -1
        
        for i, action in enumerate(puzzle.grounded_actions):
            if action['name'] == 'toggle-on':
                toggle_on_idx = i
            elif action['name'] == 'toggle-off':
                toggle_off_idx = i
                
        assert toggle_on_idx >= 0
        assert toggle_off_idx >= 0
        
        # Check applicability
        # If negative preconditions are NOT supported, toggle-on might have empty preconditions (always applicable)
        # or if `not` is ignored, maybe it sees `(on ?s)` as positive precondition?
        # If `not` is ignored recursively, it sees `(on ?s)`. Then toggle-on requires `(on ?s)`.
        # If that's the case, toggle-on is NOT applicable in empty state.
        
        cost_on = costs[toggle_on_idx]
        assert jnp.isfinite(cost_on), "toggle-on should be applicable in empty state (requires not on)"
        
        # Now emulate state where (on s1) is true
        # We can construct it manually if needed, or take the step if supported
        
    def test_equality(self):
        domain_file = str(BASE_DIR / "equality" / "domain.pddl")
        problem_file = str(BASE_DIR / "equality" / "problem.pddl")
        
        puzzle = PDDL(domain_file, problem_file)
        
        # Grounded actions should NOT include move l1 l1
        # Because (not (= ?from ?to))
        
        found_self_loop = False
        for action in puzzle.grounded_actions:
            if action['name'] == 'move':
                params = action['parameters']
                if params[0] == params[1]:
                    found_self_loop = True
                    break
        
        assert not found_self_loop, "Equality constraint 'not =' should prevent self-loop action"

    def test_zero_arity_formatting(self):
        domain_file = str(BASE_DIR / "zero_arity" / "domain.pddl")
        problem_file = str(BASE_DIR / "zero_arity" / "problem.pddl")
        
        puzzle = PDDL(domain_file, problem_file)
        
        # Atom should be "(light-on)", NOT "(light-on )"
        assert "(light-on)" in puzzle.grounded_atoms
        assert "(light-on )" not in puzzle.grounded_atoms
