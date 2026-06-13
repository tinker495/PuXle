import random
from typing import Dict, List, Optional, Set

from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.logic import Constant, Predicate
from pddl.logic.base import And, Not


class ProblemGenerator:
    """
    Generates solvable problems by simulating valid execution traces.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_problem(
        self,
        domain: Domain,
        num_objects: int = 5,
        walk_length: int = 10,
        problem_name: str = "generated-problem",
    ) -> Problem:
        """
        Generates a random problem for the given domain.

        Args:
            domain: The PDDL domain.
            num_objects: Number of objects to create.
            walk_length: Number of steps to simulate for goal generation.

        Returns:
            A pddl.Problem instance.
        """

        # 1. Create objects, assigning random types when the domain is typed.
        types = list(domain.types) if domain.types else []
        objects = []

        if types:
            for i in range(num_objects):
                obj_type = self.rng.choice(types)
                objects.append(Constant(f"obj{i}", type_tag=str(obj_type)))
        else:
            for i in range(num_objects):
                objects.append(Constant(f"obj{i}"))

        # 2. Generate initial state, falling back to a random sparse state.
        init_state = self._generate_domain_specific_init(domain, objects)
        if not init_state:
            init_state = self._generate_initial_state(domain, objects)

        # 3. Simulate a random applicable-action walk to reach a solvable goal state.
        current_state = set(init_state)

        for _ in range(walk_length):
            found = False
            for _ in range(20):  # attempts to find an applicable grounded action
                action = self.rng.choice(list(domain.actions))
                args = self._sample_args(action, objects)
                if args is None:
                    continue

                var_map = {
                    param.name: arg for param, arg in zip(action.parameters, args)
                }
                if self._check_preconditions(
                    action.precondition, var_map, current_state
                ):
                    current_state = self._apply_effects(
                        action.effect, var_map, current_state
                    )
                    found = True
                    break

            if not found:
                break

        # 4. Goal is a small subset (<=3 atoms) of the reached final state.
        if not current_state:
            goal = And()
        else:
            goal_atoms = self.rng.sample(
                list(current_state), min(3, len(current_state))
            )
            goal = And(*goal_atoms)

        return Problem(
            name=problem_name,
            domain_name=domain.name,
            objects=objects,
            init=init_state,
            goal=goal,
        )

    def _sample_args(
        self, action: Action, objects: List[Constant]
    ) -> Optional[List[Constant]]:
        """Samples objects matching action parameter types."""
        args = []
        for param in action.parameters:
            valid_objs = []
            # Check type tags (pddl lib specific)
            if hasattr(param, "type_tags") and param.type_tags:
                # Assuming single type for simplicity or union
                # We need to match any of tags
                required = param.type_tags
                for obj in objects:
                    if hasattr(obj, "type_tags") and (obj.type_tags & required):
                        valid_objs.append(obj)
                    elif hasattr(obj, "type_tag") and obj.type_tag in required:
                        valid_objs.append(obj)
            else:
                valid_objs = objects

            if not valid_objs:
                return None
            args.append(self.rng.choice(valid_objs))
        return args

    def _sample_args_for_predicate(
        self, predicate: Predicate, objects: List[Constant]
    ) -> Optional[List[Constant]]:
        """Sample objects for a predicate's terms."""
        args = []
        for term in predicate.terms:
            # Check type restrictions
            # term.type_tags might be used if available
            valid_objs = []
            if hasattr(term, "type_tags") and term.type_tags:
                required = term.type_tags
                for obj in objects:
                    # obj.type_tags should match
                    if hasattr(obj, "type_tags") and (obj.type_tags & required):
                        valid_objs.append(obj)
                    elif hasattr(obj, "type_tag") and obj.type_tag in required:
                        valid_objs.append(obj)
            else:
                valid_objs = objects

            if not valid_objs:
                return None
            args.append(self.rng.choice(valid_objs))
        return args

    def _generate_initial_state(
        self, domain: Domain, objects: List[Constant]
    ) -> Set[Predicate]:
        """Generate a random initial state."""
        init_state = set()

        # Pick 20% of possible ground atoms randomly (naive approach)
        # To avoid explosion, we limit number of samples per predicate
        for predicate in domain.predicates:
            # We sample a few times for each predicate
            for _ in range(min(5, len(objects))):
                args = self._sample_args_for_predicate(predicate, objects)
                if args:
                    # Create ground atom
                    atom = Predicate(predicate.name, *args)
                    if self.rng.random() < 0.2:
                        init_state.add(atom)
        return init_state

    def _generate_domain_specific_init(
        self, domain: Domain, objects: List[Constant]
    ) -> Optional[Set[Predicate]]:
        """Try to generate domain-specific initial state based on name heuristics."""
        name = domain.name.lower()
        init_state = set()

        # Simple heuristic for Blocksworld
        if "blocks" in name or "blocksworld" in name:
            # All blocks on table and clear
            # Find predicates matching 'on-table' or 'ontable'
            ontable_p = next(
                (
                    p
                    for p in domain.predicates
                    if "ontable" in p.name.lower() or "on-table" in p.name.lower()
                ),
                None,
            )
            clear_p = next(
                (p for p in domain.predicates if "clear" in p.name.lower()), None
            )
            handempty_p = next(
                (
                    p
                    for p in domain.predicates
                    if "handempty" in p.name.lower() or "hand-empty" in p.name.lower()
                ),
                None,
            )

            if ontable_p and clear_p:
                for obj in objects:
                    init_state.add(Predicate(ontable_p.name, obj))
                    init_state.add(Predicate(clear_p.name, obj))

                if handempty_p:
                    init_state.add(Predicate(handempty_p.name))
                return init_state

        # Simple heuristic for Gripper
        if "gripper" in name:
            # Too complex to guess structure without knowing object types
            pass

        return None

    def _check_preconditions(
        self, precondition, var_map: Dict[str, Constant], state: Set[Predicate]
    ) -> bool:
        """
        Checks if grounded preconditions hold in state.
        """
        if precondition is None:
            return True

        grounded_pre = self._ground_formula(precondition, var_map)

        if isinstance(grounded_pre, And):
            for op in grounded_pre.operands:
                if op not in state:
                    return False
            return True
        elif isinstance(grounded_pre, Predicate):  # Atom
            return grounded_pre in state
        elif isinstance(grounded_pre, Not):
            # Negative precondition
            return grounded_pre.argument not in state

        return False  # Unsupported complex precondition

    def _apply_effects(
        self, effect, var_map: Dict[str, Constant], state: Set[Predicate]
    ) -> Set[Predicate]:
        """
        Applies grounded effects to state.
        """
        new_state = state.copy()

        grounded_eff = self._ground_formula(effect, var_map)

        atoms_to_process = []
        if isinstance(grounded_eff, And):
            atoms_to_process = list(grounded_eff.operands)
        elif grounded_eff:
            atoms_to_process = [grounded_eff]

        for atom in atoms_to_process:
            if isinstance(atom, Not):
                # Delete effect
                target = atom.argument
                if target in new_state:
                    new_state.remove(target)
            else:
                # Add effect
                if isinstance(atom, Predicate):
                    new_state.add(atom)

        return new_state

    def _ground_formula(self, formula, var_map: Dict[str, Constant]):
        """
        Recursively substitutes variables in formula with constants using the provided mapping.
        """
        if isinstance(formula, And):
            return And(*[self._ground_formula(op, var_map) for op in formula.operands])
        elif isinstance(formula, Not):
            return Not(self._ground_formula(formula.argument, var_map))
        elif isinstance(formula, Predicate):
            # Substitute terms
            new_terms = []
            for term in formula.terms:
                if hasattr(term, "name") and term.name in var_map:
                    new_terms.append(var_map[term.name])
                else:
                    # Already a constant or unknown variable
                    new_terms.append(term)
            return Predicate(formula.name, *new_terms)

        return formula
