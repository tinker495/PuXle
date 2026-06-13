"""Shared PDDL formula facts for fusion adapters."""

from __future__ import annotations

from typing import Any, Mapping

from pddl.logic import Predicate
from pddl.logic.base import And, Not


def flatten_formula(formula: Any) -> list[Any]:
    """Return leaf formula nodes from nested conjunctions."""
    if formula is None:
        return []
    if isinstance(formula, And):
        flattened: list[Any] = []
        for operand in formula.operands:
            flattened.extend(flatten_formula(operand))
        return flattened
    return [formula]


def extract_predicates(formula: Any) -> list[Predicate]:
    """Walk a formula and collect every predicate node."""
    if formula is None:
        return []
    if isinstance(formula, Predicate):
        return [formula]
    if isinstance(formula, Not):
        return extract_predicates(formula.argument)
    if isinstance(formula, And):
        predicates: list[Predicate] = []
        for operand in formula.operands:
            predicates.extend(extract_predicates(operand))
        return predicates
    return []


def ground_formula(formula: Any, var_map: Mapping[str, Any]) -> Any:
    """Substitute variables in a formula with mapped terms."""
    if isinstance(formula, And):
        return And(*(ground_formula(operand, var_map) for operand in formula.operands))
    if isinstance(formula, Not):
        return Not(ground_formula(formula.argument, var_map))
    if isinstance(formula, Predicate):
        grounded_terms = []
        for term in formula.terms:
            term_name = getattr(term, "name", None)
            grounded_terms.append(var_map.get(term_name, term))
        return Predicate(formula.name, *grounded_terms)
    return formula
