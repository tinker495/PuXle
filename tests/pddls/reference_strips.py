from typing import Dict, List, Optional, Set

import pddl
import pddl.logic.base
import pddl.logic.terms


class ReferenceSTRIPS:
    """Reference implementation of STRIPS semantics for cross-validation."""

    def __init__(self, domain_file: str, problem_file: str):
        """Initialize from PDDL domain and problem files."""
        self.domain = pddl.parse_domain(domain_file)
        self.problem = pddl.parse_problem(problem_file)

        # Enumerate all typed objects
        self.objects = {}
        for obj in self.problem.objects:
            # Handle different ways type_tags might be represented
            if hasattr(obj, "type_tags") and obj.type_tags:
                if isinstance(obj.type_tags, (list, tuple)):
                    obj_type = obj.type_tags[0]
                elif isinstance(obj.type_tags, frozenset):
                    obj_type = list(obj.type_tags)[0]
                else:
                    obj_type = str(obj.type_tags)
            else:
                obj_type = "object"

            if obj_type not in self.objects:
                self.objects[obj_type] = []
            self.objects[obj_type].append(str(obj.name))

        # Ground all predicates
        self.grounded_atoms = set()
        for predicate in self.domain.predicates:
            self._ground_predicate(predicate)

        # Ground all actions
        self.grounded_actions = []
        for action in self.domain.actions:
            self._ground_action(action)

        # Extract goal atoms
        self.goal_atoms = set()
        self._extract_goal_atoms(self.problem.goal)

        # Extract initial state atoms
        self.initial_atoms = set()
        for atom in self.problem.init:
            if isinstance(atom, pddl.logic.base.Atomic):
                self.initial_atoms.add(self._atom_to_string(atom))

    def _ground_predicate(self, predicate):
        """Ground a predicate schema."""
        param_types = []
        for term in predicate.terms:
            if hasattr(term, "type_tags") and term.type_tags:
                if isinstance(term.type_tags, (list, tuple)):
                    param_type = term.type_tags[0]
                elif isinstance(term.type_tags, frozenset):
                    param_type = list(term.type_tags)[0]
                else:
                    param_type = str(term.type_tags)
            else:
                param_type = "object"
            param_types.append(param_type)

        # Generate all valid parameter combinations
        param_combinations = self._generate_param_combinations(param_types)

        for params in param_combinations:
            atom_str = f"({predicate.name} {' '.join(params)})"
            self.grounded_atoms.add(atom_str)

    def _ground_action(self, action):
        """Ground an action schema."""
        param_types = []
        for param in action.parameters:
            if hasattr(param, "type_tags") and param.type_tags:
                if isinstance(param.type_tags, (list, tuple)):
                    param_type = param.type_tags[0]
                elif isinstance(param.type_tags, frozenset):
                    param_type = list(param.type_tags)[0]
                else:
                    param_type = str(param.type_tags)
            else:
                param_type = "object"
            param_types.append(param_type)

        # Generate all valid parameter combinations
        param_combinations = self._generate_param_combinations(param_types)

        for params in param_combinations:
            # Create parameter substitution
            param_map = {
                param.name: value for param, value in zip(action.parameters, params)
            }

            # Ground preconditions
            preconditions = set()
            if action.precondition:
                self._ground_formula(action.precondition, param_map, preconditions)

            # Ground effects
            add_effects = set()
            delete_effects = set()
            if action.effect:
                self._ground_effect(
                    action.effect, param_map, add_effects, delete_effects
                )

            grounded_action = {
                "name": action.name,
                "parameters": params,
                "preconditions": preconditions,
                "add_effects": add_effects,
                "delete_effects": delete_effects,
            }
            self.grounded_actions.append(grounded_action)

    def _generate_param_combinations(self, param_types):
        """Generate all valid parameter combinations for given types."""
        if not param_types:
            return [[]]

        type_objects = []
        for param_type in param_types:
            if param_type in self.objects:
                type_objects.append(self.objects[param_type])
            else:
                # Fallback to all objects if type not found
                all_objects = []
                for obj_list in self.objects.values():
                    all_objects.extend(obj_list)
                type_objects.append(all_objects)

        # Generate cartesian product
        import itertools

        return list(itertools.product(*type_objects))

    def _ground_formula(self, formula, param_map, result_set):
        """Ground a formula (precondition) with parameter substitution."""
        if isinstance(formula, pddl.logic.base.Atomic):
            atom_str = self._atom_to_string(formula, param_map)
            result_set.add(atom_str)
        elif isinstance(formula, pddl.logic.base.And):
            for arg in formula.operands:
                self._ground_formula(arg, param_map, result_set)
        elif isinstance(formula, pddl.logic.base.Or):
            # For STRIPS, we only support conjunctive preconditions
            # This is a simplification - in practice, we'd need to handle this differently
            for arg in formula.operands:
                self._ground_formula(arg, param_map, result_set)

    def _ground_effect(self, effect, param_map, add_effects, delete_effects):
        """Ground an effect with parameter substitution."""
        if isinstance(effect, pddl.logic.base.Atomic):
            atom_str = self._atom_to_string(effect, param_map)
            add_effects.add(atom_str)
        elif isinstance(effect, pddl.logic.base.Not):
            if isinstance(effect.argument, pddl.logic.base.Atomic):
                atom_str = self._atom_to_string(effect.argument, param_map)
                delete_effects.add(atom_str)
        elif isinstance(effect, pddl.logic.base.And):
            for arg in effect.operands:
                self._ground_effect(arg, param_map, add_effects, delete_effects)

    def _atom_to_string(self, atom, param_map=None):
        """Convert an atom to string representation."""
        if param_map:
            # Substitute parameters
            args = [param_map.get(arg.name, str(arg)) for arg in atom.terms]
        else:
            # Use original names
            args = [str(arg) for arg in atom.terms]
        return f"({atom.name} {' '.join(args)})"

    def _extract_goal_atoms(self, goal):
        """Extract goal atoms from goal formula."""
        if isinstance(goal, pddl.logic.base.Atomic):
            self.goal_atoms.add(self._atom_to_string(goal))
        elif isinstance(goal, pddl.logic.base.And):
            for arg in goal.operands:
                self._extract_goal_atoms(arg)

    def is_applicable(self, action, state: Set[str]) -> bool:
        """Check if action is applicable in given state."""
        return set(action["preconditions"]).issubset(state)

    def apply_action(self, action, state: Set[str]) -> Set[str]:
        """Apply action to state, returning successor state."""
        if not self.is_applicable(action, state):
            raise ValueError("Action not applicable in state")

        successor = state.copy()
        successor.difference_update(action["delete_effects"])
        successor.update(action["add_effects"])
        return successor

    def is_goal_satisfied(self, state: Set[str]) -> bool:
        """Check if goal is satisfied in given state."""
        return self.goal_atoms.issubset(state)

    def get_applicable_actions(self, state: Set[str]) -> List[Dict]:
        """Get all applicable actions in given state."""
        return [
            action
            for action in self.grounded_actions
            if self.is_applicable(action, state)
        ]

    def bfs_search(self, max_depth: int = 6) -> Optional[List[Dict]]:
        """Breadth-first search for a plan."""
        if self.is_goal_satisfied(self.initial_atoms):
            return []

        queue = [(self.initial_atoms, [])]  # (state, plan)
        visited = {frozenset(self.initial_atoms)}

        for depth in range(max_depth):
            next_queue = []

            for state, plan in queue:
                applicable_actions = self.get_applicable_actions(state)

                for action in applicable_actions:
                    successor = self.apply_action(action, state)
                    successor_frozen = frozenset(successor)

                    if successor_frozen not in visited:
                        visited.add(successor_frozen)
                        new_plan = plan + [action]

                        if self.is_goal_satisfied(successor):
                            return new_plan

                        next_queue.append((successor, new_plan))

            queue = next_queue
            if not queue:
                break

        return None  # No plan found within max_depth
