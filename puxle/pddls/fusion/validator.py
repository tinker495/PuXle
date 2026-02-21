from typing import List, Tuple

from pddl.core import Action, Domain, Formula
from pddl.logic import Predicate
from pddl.logic.base import And, Not


class DomainValidator:
    """Validates fused domains for structural and logical correctness."""

    def validate(self, domain: Domain) -> Tuple[bool, List[str]]:
        """
        Validates the domain.
        Returns: (is_valid, list_of_error_messages)
        """
        errors = []

        # 0. Build predicate lookup
        defined_predicates = {p.name: p.arity for p in domain.predicates}

        # 1. Check all predicates in actions exist in domain
        for action in domain.actions:
            action_preds = self._extract_predicates(
                action.precondition
            ) + self._extract_predicates(action.effect)

            for pred in action_preds:
                if pred.name not in defined_predicates:
                    errors.append(
                        f"Action '{action.name}' uses undefined predicate '{pred.name}'"
                    )
                elif defined_predicates[pred.name] != pred.arity:
                    errors.append(
                        f"Action '{action.name}' uses predicate '{pred.name}' with arity "
                        f"{pred.arity}, expected {defined_predicates[pred.name]}"
                    )

        # 2. Check for contradictions (P and ~P in same effect)
        for action in domain.actions:
            if not self._validate_action_consistency(action):
                errors.append(f"Action '{action.name}' has contradictory effects")
            if not self._validate_type_consistency(action, domain):
                errors.append(f"Action '{action.name}' uses undefined parameter types")

        return (len(errors) == 0, errors)

    def _validate_type_consistency(self, action: Action, domain: Domain) -> bool:
        """Check if action parameter types are defined in the domain."""
        # domain.types usually contains all Type objects.
        # We need to extract names.
        if not domain.types:
            return True  # Untyped

        defined_types = {str(t) for t in domain.types}
        defined_types.add("object")  # implicit root

        for param in action.parameters:
            if hasattr(param, "type_tags") and param.type_tags:
                for tag in param.type_tags:
                    if tag not in defined_types:
                        # Check strictness: if tag is not in types, invalid?
                        # PDDL might allow implicit types but usually strict matching.
                        return False
            elif hasattr(param, "type_tag") and param.type_tag:
                if param.type_tag not in defined_types:
                    return False

        return True

    def _extract_predicates(self, formula: Formula) -> List[Predicate]:
        """Recursively extract predicates from a formula."""
        if formula is None:
            return []

        preds = []
        if isinstance(formula, Predicate):
            preds.append(formula)
        elif isinstance(formula, Not):
            preds.extend(self._extract_predicates(formula.argument))
        elif isinstance(formula, And):  # Or other composite
            for op in formula.operands:
                preds.extend(self._extract_predicates(op))
        # Add other types if needed (Or, Imply, etc.)
        return preds

    def _validate_action_consistency(self, action: Action) -> bool:
        """Check if action effects contain direct contradictions."""
        # This is a simple check: can't add and delete the same atom.
        # But 'same atom' means same predicate and same variables.

        # We need to flatten effects into lists of atomic effects
        effects = self._flatten_effects(action.effect)

        adds = set()
        dels = set()

        for eff in effects:
            if isinstance(eff, Not):
                # We convert to string representation for comparison
                dels.add(str(eff.argument))
            else:
                adds.add(str(eff))

        # Intersection means we try to Add P and Del P at same time.
        # In PDDL, usually Del happens before Add or undefined. PDDLFuse paper mentions consistency check.
        # If intersection is non-empty, it's ambiguous or contradictory.
        if not adds.isdisjoint(dels):
            return False

        return True

    def _flatten_effects(self, formula: Formula) -> List[Formula]:
        """Flatten nested Ands."""
        if formula is None:
            return []
        if isinstance(formula, And):
            res = []
            for op in formula.operands:
                res.extend(self._flatten_effects(op))
            return res
        return [formula]
