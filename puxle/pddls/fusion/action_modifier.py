import random
from typing import List, Set, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import copy

import pddl
from pddl.logic import Predicate, Constant, Variable
from pddl.core import Action, Formula
from pddl.logic.base import And, Or, Not, OneOf
from pddl.logic.predicates import Predicate

@dataclass
class FusionParams:
    prob_add_pre: float = 0.1
    prob_add_eff: float = 0.1
    prob_rem_pre: float = 0.05
    prob_rem_eff: float = 0.05
    prob_neg: float = 0.1
    seed: int = 42

class ActionModifier:
    """
    Modifies actions by stochastically adding preconditions and effects.
    """
    def __init__(self, params: FusionParams):
        self.params = params
        self.rng = random.Random(params.seed)

    def modify_actions(
        self, 
        actions: List[Action], 
        all_predicates: List[Predicate], 
        types_map: Dict[str, Any]
    ) -> List[Action]:
        """
        Apply stochastic modifications to a list of actions.
        """
        modified_actions = []
        for action in actions:
            modified = self._modify_single_action(action, all_predicates, types_map)
            modified_actions.append(modified)
        return modified_actions

    def _modify_single_action(
        self, 
        action: Action, 
        all_predicates: List[Predicate], 
        types_map: Dict[str, Any]
    ) -> Action:
        # PDDL library objects are often immutable, so we might need to recreate them.
        # Strategically, we will construct new precondition and effect lists.

        # 1. Parse current preconditions/effects into mutable lists
        preconditions = self._extract_atomic_conditions(action.precondition)
        effects = self._extract_atomic_conditions(action.effect)
        
        # Action parameters (variables)
        params = action.parameters # Tuple of Variable

        # 2. Possibly add preconditions
        if self.rng.random() < self.params.prob_add_pre:
            new_pre = self._sample_predicate_for_action(all_predicates, params, types_map)
            if new_pre:
                # Optionally negate
                if self.rng.random() < self.params.prob_neg:
                    new_pre = Not(new_pre)
                preconditions.append(new_pre)

        # 2.5. Possibly remove preconditions
        # We only remove if there are preconditions to remove
        if preconditions and self.rng.random() < self.params.prob_rem_pre:
            # Pick one random index to remove
            idx = self.rng.randint(0, len(preconditions) - 1)
            preconditions.pop(idx)

        # 3. Possibly add effects
        if self.rng.random() < self.params.prob_add_eff:
            new_eff = self._sample_predicate_for_action(all_predicates, params, types_map)
            if new_eff:
                # Check for consistency (don't add P and ~P)
                # Simple check: if Not(new_eff) in effects, don't add
                # or if new_eff in effects but we want to add Not(new_eff)
                
                is_neg = self.rng.random() < self.params.prob_neg
                term_to_add = Not(new_eff) if is_neg else new_eff
                
                if self._is_consistent(term_to_add, effects):
                    effects.append(term_to_add)

        # 3.5. Possibly remove effects
        if effects and self.rng.random() < self.params.prob_rem_eff:
            idx = self.rng.randint(0, len(effects) - 1)
            effects.pop(idx)

        # Reconstruct Action
        # Note: We need to combine lists back into And/Formula
        new_precondition = self._list_to_formula(preconditions)
        new_effect = self._list_to_formula(effects)

        return Action(
            name=action.name,
            parameters=action.parameters,
            precondition=new_precondition,
            effect=new_effect
        )

    def _sample_predicate_for_action(
        self, 
        all_predicates: List[Predicate], 
        action_params: Tuple[Variable, ...], 
        types_map: Dict[str, Any] # Unused for now, but useful for type checking
    ) -> Optional[Predicate]:
        """
        Samples a predicate and grounds it with action parameters.
        Constraints:
        - Predicate arity <= len(action_params)
        - Variable types must match (if typed)
        """
        candidates = []
        for p in all_predicates:
            if p.arity <= len(action_params):
                candidates.append(p)
        
        if not candidates:
            return None
            
        chosen_pred = self.rng.choice(candidates)
        
        # Sample variables for the predicate terms
        # Need to match types eventually, but for now randomly pick from params
        # (Assuming untyped or loosely typed for simple fusion, strictly typed needs checks)
        # TODO: Implement type checking logic
        
        # Simple sampling (allows duplicates, e.g. P(x, x))
        # If predicate needs k args, pick k from action_params
        if len(action_params) < chosen_pred.arity:
            return None # Should be covered by candidate filter but safe check
            
        chosen_vars = self.rng.sample(action_params, chosen_pred.arity)
        
        # Create new Predicate instance with variables
        return Predicate(chosen_pred.name, *chosen_vars)

    def _extract_atomic_conditions(self, formula: Formula) -> List[Formula]:
        """Unpack And(a, b, c) to [a, b, c]. Handle single atom too."""
        if formula is None:
            return []
        if isinstance(formula, And):
            return list(formula.operands)
        return [formula]

    def _list_to_formula(self, dry_list: List[Formula]) -> Optional[Formula]:
        if not dry_list:
            return And() # Return empty And instead of None
        if len(dry_list) == 1:
            return dry_list[0]
        return And(*dry_list)

    def _is_consistent(self, term: Formula, current_effects: List[Formula]) -> bool:
        # Simple contradiction check: P vs Not(P)
        # We need a way to check equality of atoms. 
        # pddl library objects implement equality typically.
        
        # Construct the negation of the term we want to add
        if isinstance(term, Not):
             contradiction = term.argument
        else:
             contradiction = Not(term)
             
        return contradiction not in current_effects
