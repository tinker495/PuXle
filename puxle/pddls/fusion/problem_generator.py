import random
from typing import Dict, List, Optional, Set

from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.logic import Constant, Predicate
from pddl.logic.base import And, Not

from puxle.pddls.fusion.formula_facts import (
    flatten_formula,
    ground_formula,
)
from puxle.pddls.fusion.type_facts import (
    any_match_compatible_candidates,
    build_type_ancestor_map,
    domain_type_parent_map,
)


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
        type_ancestors = build_type_ancestor_map(domain_type_parent_map(domain))

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
            init_state = self._generate_initial_state(
                domain,
                objects,
                type_ancestors,
            )

        # 3. Simulate a random applicable-action walk to reach a solvable goal state.
        current_state = set(init_state)

        for _ in range(walk_length):
            found = False
            for _ in range(20):  # attempts to find an applicable grounded action
                action = self.rng.choice(list(domain.actions))
                args = self._sample_args(action, objects, type_ancestors)
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
        self,
        action: Action,
        objects: List[Constant],
        type_ancestors: Dict[str, Set[str]] | None = None,
    ) -> Optional[List[Constant]]:
        """Samples objects matching action parameter types."""
        args = []
        for param in action.parameters:
            valid_objs = any_match_compatible_candidates(
                objects,
                param,
                type_ancestors,
            )

            if not valid_objs:
                return None
            args.append(self.rng.choice(valid_objs))
        return args

    def _sample_args_for_predicate(
        self,
        predicate: Predicate,
        objects: List[Constant],
        type_ancestors: Dict[str, Set[str]] | None = None,
    ) -> Optional[List[Constant]]:
        """Sample objects for a predicate's terms."""
        args = []
        for term in predicate.terms:
            valid_objs = any_match_compatible_candidates(
                objects,
                term,
                type_ancestors,
            )

            if not valid_objs:
                return None
            args.append(self.rng.choice(valid_objs))
        return args

    def _generate_initial_state(
        self,
        domain: Domain,
        objects: List[Constant],
        type_ancestors: Dict[str, Set[str]] | None = None,
    ) -> Set[Predicate]:
        """Generate a random initial state."""
        init_state = set()

        # Pick 20% of possible ground atoms randomly (naive approach)
        # To avoid explosion, we limit number of samples per predicate
        for predicate in domain.predicates:
            # We sample a few times for each predicate
            for _ in range(min(5, len(objects))):
                args = self._sample_args_for_predicate(
                    predicate,
                    objects,
                    type_ancestors,
                )
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

        grounded_pre = ground_formula(precondition, var_map)

        if isinstance(grounded_pre, And):
            for op in flatten_formula(grounded_pre):
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

        atoms_to_process = flatten_formula(ground_formula(effect, var_map))

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
