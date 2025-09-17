from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .type_system import select_most_specific_types


@dataclass(frozen=True)
class GroundLiteral:
    """Representation of a grounded literal with polarity."""

    atom: str
    positive: bool = True


def _get_type_combinations(param_types: List[object], objects_by_type: Dict[str, List[str]]) -> List[List[str]]:
    """Get all valid object combinations for given parameter types.

    param_types may contain strings (single type) or iterables (union of types).
    """
    if not param_types:
        return [[]]

    combinations: List[List[str]] = []
    first_type = param_types[0]
    remaining_types = param_types[1:]

    # Get objects of the first type (supports union types)
    if isinstance(first_type, (list, tuple, set)):
        seen_union: set[str] = set()
        available_objects: list[str] = []
        for t in first_type:
            for o in objects_by_type.get(t, []):
                if o not in seen_union:
                    seen_union.add(o)
                    available_objects.append(o)
    else:
        available_objects = list(objects_by_type.get(first_type, []))

    if not available_objects:
        return []

    sub_combinations = _get_type_combinations(remaining_types, objects_by_type)

    for obj in available_objects:
        for sub_combo in sub_combinations:
            combinations.append([obj] + sub_combo)

    return combinations


def ground_predicates(domain, objects_by_type: Dict[str, List[str]], hierarchy) -> Tuple[List[str], Dict[str, int]]:
    """Ground all predicates from the domain using the object universe."""
    grounded_atoms: List[str] = []
    atom_to_idx: Dict[str, int] = {}

    predicates = getattr(domain, "predicates", [])

    for predicate in predicates:
        pred_name = predicate.name
        # Extract parameter types from terms
        param_types: List[object] = []
        for term in getattr(predicate, "terms", []) or []:
            if hasattr(term, "type_tags") and term.type_tags:
                selected = select_most_specific_types(set(term.type_tags), hierarchy)
                param_types.append(selected[0] if len(selected) == 1 else selected)
            else:
                param_types.append("object")

        # Generate all type-consistent object combinations
        type_combinations = _get_type_combinations(param_types, objects_by_type)

        for obj_combination in type_combinations:
            atom_str = f"({pred_name} {' '.join(obj_combination)})"
            atom_to_idx[atom_str] = len(grounded_atoms)
            grounded_atoms.append(atom_str)

    return grounded_atoms, atom_to_idx


def _ground_formula(formula, param_substitution: List[str], param_names: List[str]) -> List[GroundLiteral]:
    """Ground a (possibly compound) formula into signed literals."""
    if formula is None:
        return []

    # Handle simple atomic formulas
    if hasattr(formula, "name"):
        pred_name = formula.name
        args = [getattr(arg, "name", str(arg)) for arg in getattr(formula, "terms", []) or []]

        substituted_args: List[str] = []
        for arg in args:
            if arg in param_names:
                param_idx = param_names.index(arg)
                if param_idx < len(param_substitution):
                    substituted_args.append(param_substitution[param_idx])
                else:
                    substituted_args.append(arg)
            else:
                substituted_args.append(arg)

        literal = f"({pred_name} {' '.join(substituted_args)})"
        return [GroundLiteral(atom=literal, positive=True)]

    # Handle negation (Not)
    if hasattr(formula, "argument"):
        grounded_inner = _ground_formula(formula.argument, param_substitution, param_names)
        return [GroundLiteral(atom=lit.atom, positive=not lit.positive) for lit in grounded_inner]

    # Handle compound formulas (AND, OR, etc.)
    if hasattr(formula, "parts"):
        literals: List[GroundLiteral] = []
        for part in formula.parts:
            literals.extend(_ground_formula(part, param_substitution, param_names))
        return literals

    # Handle And/Or objects
    if hasattr(formula, "operands"):
        literals = []
        for operand in formula.operands:
            literals.extend(_ground_formula(operand, param_substitution, param_names))
        return literals

    return []


def _ground_effects(effect, param_substitution: List[str], param_names: List[str]) -> Tuple[List[str], List[str]]:
    """Ground effects with parameter substitution, return (add_effects, delete_effects)."""
    add_effects: List[str] = []
    delete_effects: List[str] = []

    if effect is None:
        return add_effects, delete_effects

    # Handle conjunctions (And) represented via parts or operands
    if hasattr(effect, "parts") or hasattr(effect, "operands"):
        parts = getattr(effect, "parts", []) or getattr(effect, "operands", [])
        for part in parts:
            part_add, part_delete = _ground_effects(part, param_substitution, param_names)
            add_effects.extend(part_add)
            delete_effects.extend(part_delete)
        return add_effects, delete_effects

    # Handle negation (Not)
    if hasattr(effect, "argument"):
        grounded = _ground_formula(effect.argument, param_substitution, param_names)
        for literal in grounded:
            delete_effects.append(literal.atom)
        return add_effects, delete_effects

    # Handle atomic positive literals
    if hasattr(effect, "name") and hasattr(effect, "terms"):
        grounded = _ground_formula(effect, param_substitution, param_names)
        for literal in grounded:
            if literal.positive:
                add_effects.append(literal.atom)
            else:
                delete_effects.append(literal.atom)
        return add_effects, delete_effects

    return add_effects, delete_effects


def ground_actions(domain, objects_by_type: Dict[str, List[str]], hierarchy) -> Tuple[List[Dict], Dict[str, int]]:
    """Ground all actions from the domain using the object universe."""
    grounded_actions: List[Dict] = []
    action_to_idx: Dict[str, int] = {}

    for action in getattr(domain, "actions", []) or []:
        action_name = action.name
        param_types: List[object] = []
        for param in getattr(action, "parameters", []) or []:
            if hasattr(param, "type_tags") and param.type_tags:
                selected = select_most_specific_types(set(param.type_tags), hierarchy)
                param_types.append(selected[0] if len(selected) == 1 else selected)
            else:
                param_types.append("object")

        param_combinations = _get_type_combinations(param_types, objects_by_type)

        for param_combo in param_combinations:
            param_names = [param.name for param in getattr(action, "parameters", []) or []]

            pre_literals = _ground_formula(action.precondition, param_combo, param_names)
            positive_pres = [lit.atom for lit in pre_literals if lit.positive]
            negative_pres = [lit.atom for lit in pre_literals if not lit.positive]

            add_effects, delete_effects = _ground_effects(action.effect, param_combo, param_names)

            grounded_action = {
                "name": action_name,
                "parameters": param_combo,
                "preconditions": positive_pres,
                "preconditions_neg": negative_pres,
                "effects": (add_effects, delete_effects),
            }

            action_str = f"({action_name} {' '.join(param_combo)})"
            action_to_idx[action_str] = len(grounded_actions)
            grounded_actions.append(grounded_action)

    return grounded_actions, action_to_idx
