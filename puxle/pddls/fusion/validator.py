from typing import List, Tuple

from pddl.core import Action, Domain
from pddl.logic.base import Not

from puxle.pddls.fusion.formula_facts import (
    extract_predicates,
    flatten_formula,
)
from puxle.pddls.fusion.type_facts import (
    normalise_type_tags,
)


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
            action_preds = extract_predicates(action.precondition) + extract_predicates(
                action.effect
            )

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
        if not domain.types:
            return True  # Untyped

        defined_types = {str(t) for t in domain.types}
        defined_types.add("object")  # implicit root

        for param in action.parameters:
            for tag in normalise_type_tags(param):
                if tag not in defined_types:
                    return False

        return True

    def _validate_action_consistency(self, action: Action) -> bool:
        """Check if action effects contain direct contradictions.

        An action is inconsistent if it both adds and deletes the same atom
        (same predicate and arguments).
        """
        effects = flatten_formula(action.effect)

        adds = set()
        dels = set()
        for eff in effects:
            if isinstance(eff, Not):
                dels.add(str(eff.argument))
            else:
                adds.add(str(eff))

        return adds.isdisjoint(dels)
