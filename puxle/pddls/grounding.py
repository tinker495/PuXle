"""Predicate and action grounding for PDDL environments.

Generates the universe of grounded atoms and grounded actions by
instantiating every typed predicate / action schema with all valid
object combinations drawn from the problem's typed-object pool.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .type_system import select_most_specific_types


def _normalize_type_options(type_tags, type_hierarchy) -> List[str]:
    """Resolve a term's type tags into a deterministic list of candidate types."""
    tags = set(type_tags) if type_tags else {"object"}
    most_specific = select_most_specific_types(tags, type_hierarchy)
    if not most_specific:
        return ["object"]
    return sorted(set(most_specific))


def _objects_for_type_spec(type_spec, objects_by_type: Dict[str, List[str]]) -> List[str]:
    """Return candidate objects for one parameter type spec.

    ``type_spec`` can be either:
    - ``str``: a single type
    - iterable of ``str``: union type (e.g., ``(either t1 t2)``)
    """
    if isinstance(type_spec, str):
        return list(objects_by_type.get(type_spec, []))

    # Union of types
    candidates: List[str] = []
    seen: set[str] = set()
    for t in type_spec:
        for obj in objects_by_type.get(t, []):
            if obj not in seen:
                seen.add(obj)
                candidates.append(obj)
    return candidates


def _get_type_combinations(param_types: List[object], objects_by_type: Dict[str, List[str]]) -> List[List[str]]:
    """Get all valid object combinations for given parameter types."""
    if not param_types:
        return [[]]

    first_type = param_types[0]
    rest_types = param_types[1:]

    objects = _objects_for_type_spec(first_type, objects_by_type)
    if not objects:
        return []

    rest_combinations = _get_type_combinations(rest_types, objects_by_type)

    combinations: List[List[str]] = []
    for obj in objects:
        for combo in rest_combinations:
            combinations.append([obj] + combo)

    return combinations


def ground_predicates(
    predicates,
    objects_by_type: Dict[str, List[str]],
    type_hierarchy: Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]],
) -> Tuple[List[str], Dict[str, int]]:
    """Grounds all predicates to create the set of all possible atoms."""
    grounded_atoms = []
    atom_to_idx = {}

    for predicate in predicates:
        pred_name = predicate.name
        param_types = []
        if hasattr(predicate, "terms"):
            for term in predicate.terms:
                type_tags = getattr(term, "type_tags", None)
                if not type_tags:
                    single = getattr(term, "type_tag", None)
                    type_tags = {single} if single else {"object"}
                param_types.append(_normalize_type_options(type_tags, type_hierarchy))

        type_combinations = _get_type_combinations(param_types, objects_by_type)

        for obj_combination in type_combinations:
            args = " ".join(obj_combination)
            atom_str = f"({pred_name} {args})" if args else f"({pred_name})"
            atom_to_idx[atom_str] = len(grounded_atoms)
            grounded_atoms.append(atom_str)

    return grounded_atoms, atom_to_idx


def _ground_formula(
    formula, param_substitution: List[str], param_names: List[str]
) -> Tuple[List[str], List[str], bool]:
    """
    Recursively grounds a formula.
    Returns: (positive_atoms, negative_atoms, impossible_flag)
    """
    if formula is None:
        return [], [], False

    # Handle Equality (= ?x ?y)
    formula_type = type(formula).__name__
    if formula_type == "EqualTo":
        left = getattr(formula.left, "name", str(formula.left))
        right = getattr(formula.right, "name", str(formula.right))
        
        # Substitute parameters
        if left in param_names:
            left = param_substitution[param_names.index(left)]
        if right in param_names:
            right = param_substitution[param_names.index(right)]
            
        if left == right:
            return [], [], False  # Satisfied
        else:
            return [], [], True  # Impossible

    if formula_type == "Or":
        parts = []
        if hasattr(formula, "parts"):
            parts = list(formula.parts)
        elif hasattr(formula, "operands"):
            parts = list(formula.operands)
        if not parts:
            # pddl parser may represent empty `()` as `Or()`; treat as tautology.
            return [], [], False
        raise ValueError(
            "Disjunctive preconditions `(or ...)` are not supported in STRIPS mode."
        )

    if formula_type == "OneOf":
        raise ValueError(
            "Non-deterministic preconditions `(oneof ...)` are not supported in STRIPS mode."
        )

    # Handle Negation (Not)
    if formula_type == "Not":
        pos, neg, impossible = _ground_formula(formula.argument, param_substitution, param_names)
        if impossible:
            # Not(Impossible) -> Satisfied
            return [], [], False
        
        # If inner is unconditionally true (empty requirements)
        if not pos and not neg:
            return [], [], True

        # Swap pos and neg
        # Note: PDDL usually only allows neg of atomic/equality in preconditions
        if pos: 
            return neg, pos, False # Move pos to neg
        elif neg:
             return neg, pos, False # Move neg to pos (double negation)
        return [], [], False # Empty

    # Handle compound formulas (conjunctive only)
    parts = []
    if hasattr(formula, "parts"):
        parts = formula.parts
    elif hasattr(formula, "operands"):
        parts = formula.operands
    
    if formula_type == "And" and not parts:
        # Empty conjunction means no constraints.
        return [], [], False

    if parts:
        all_pos = []
        all_neg = []
        for part in parts:
            pos, neg, impossible = _ground_formula(part, param_substitution, param_names)
            if impossible:
                return [], [], True
            all_pos.extend(pos)
            all_neg.extend(neg)
        return all_pos, all_neg, False

    # Handle Atomic Prediction
    if hasattr(formula, "name") and hasattr(formula, "terms"):
        pred_name = formula.name
        obj_combination = []
        for term in formula.terms:
            term_name = getattr(term, "name", str(term))
            if term_name in param_names:
                obj_combination.append(param_substitution[param_names.index(term_name)])
            else:
                obj_combination.append(term_name)
        
        args = " ".join(obj_combination)
        atom_str = f"({pred_name} {args})" if args else f"({pred_name})"
        return [atom_str], [], False

    raise ValueError(
        f"Unsupported formula node `{formula_type}` in STRIPS grounding."
    )


def _ground_effects(
    effect, param_substitution: List[str], param_names: List[str]
) -> Dict[str, List[str]]:
    """Grounds effects into add/delete lists."""
    add_effects = []
    delete_effects = []

    # Flatten effect structure
    effects_list = []
    if hasattr(effect, "parts"):
        effects_list = effect.parts
    elif hasattr(effect, "operands"):
        effects_list = effect.operands
    else:
        effects_list = [effect]

    for eff in effects_list:
        eff_type = type(eff).__name__
        if eff_type in {"Or", "OneOf"}:
            parts = []
            if hasattr(eff, "parts"):
                parts = list(eff.parts)
            elif hasattr(eff, "operands"):
                parts = list(eff.operands)
            if not parts:
                # pddl parser may represent empty `()` as `Or()`; treat as no-op.
                continue
            raise ValueError(
                f"Unsupported effect node `{eff_type}` in STRIPS grounding."
            )

        # Check for negation
        if eff_type == "Not":
            # Negative effect (delete)
            # argument is implicitly atomic in STRIPS
            pos, neg, imp = _ground_formula(eff.argument, param_substitution, param_names)
            if not imp and pos:
                 delete_effects.extend(pos) 
                 # _ground_formula returns list[str], we extend
        else:
            # Positive effect (add)
            pos, neg, imp = _ground_formula(eff, param_substitution, param_names)
            if not imp and pos:
                add_effects.extend(pos)

    return {"add": add_effects, "delete": delete_effects}


def ground_actions(
    actions,
    objects_by_type: Dict[str, List[str]],
    type_hierarchy: Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]],
) -> Tuple[List[Dict], Dict[str, int]]:
    """Grounds all actions."""
    grounded_actions = []
    action_to_idx = {}

    for action in actions:
        action_name = action.name
        param_types = []
        if hasattr(action, "parameters"):
            for param in action.parameters:
                type_tags = getattr(param, "type_tags", None)
                if not type_tags:
                    single = getattr(param, "type_tag", None)
                    type_tags = {single} if single else {"object"}
                param_types.append(_normalize_type_options(type_tags, type_hierarchy))

        param_combinations = _get_type_combinations(param_types, objects_by_type)

        for param_combo in param_combinations:
            param_names = [param.name for param in getattr(action, "parameters", []) or []]
            
            # Ground Preconditions
            # Result: (pos, neg, impossible)
            pre_pos, pre_neg, impossible = _ground_formula(
                action.precondition, param_combo, param_names
            )
            
            if impossible:
                continue

            grounded_action = {
                "name": action_name,
                "parameters": param_combo,
                "preconditions": pre_pos,
                "preconditions_neg": pre_neg,
                "effects": _ground_effects(action.effect, param_combo, param_names),
            }

            args = " ".join(param_combo)
            action_str = f"({action_name} {args})" if args else f"({action_name})"
            action_to_idx[action_str] = len(grounded_actions)
            grounded_actions.append(grounded_action)

    return grounded_actions, action_to_idx
