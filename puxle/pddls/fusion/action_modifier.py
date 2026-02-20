import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from pddl.core import Action, Formula
from pddl.logic import Variable
from pddl.logic.base import And, Not
from pddl.logic.predicates import Predicate

@dataclass
class FusionParams:
    prob_add_pre: float = 0.1
    prob_add_eff: float = 0.1
    prob_rem_pre: float = 0.05
    prob_rem_eff: float = 0.05
    prob_neg: float = 0.1
    rev_flag: bool = True # Ensures reversibility
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
        type_ancestors = self._build_type_ancestor_map(types_map)
        modified_actions = []
        for action in actions:
            modified = self._modify_single_action(action, all_predicates, type_ancestors)
            modified_actions.append(modified)
            
        if self.params.rev_flag:
            modified_actions = self._enforce_reversibility(modified_actions, all_predicates, types_map, type_ancestors)
            
        return modified_actions

    def _modify_single_action(
        self, 
        action: Action, 
        all_predicates: List[Predicate], 
        type_ancestors: Dict[str, Set[str]],
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
            new_pre = self._sample_predicate_for_action(all_predicates, params, type_ancestors)
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
            new_eff = self._sample_predicate_for_action(all_predicates, params, type_ancestors)
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

    def _enforce_reversibility(
        self,
        actions: List[Action],
        all_predicates: List[Predicate],
        types_map: Dict[str, Any],
        type_ancestors: Dict[str, Set[str]]
    ) -> List[Action]:
        """
        Ensures that for every predicate P, if an action deletes P, there is an action that adds P.
        """
        # 1. Identify predicates that are deleted but never added
        deleted_preds = set()
        added_preds = set()
        
        for action in actions:
            effects = self._extract_atomic_conditions(action.effect)
            for eff in effects:
                if isinstance(eff, Not):
                    # Deletion
                    # We store the name only, as strict reversibility might require matching args,
                    # but paper usually implies predicate-level or general existence.
                    # "Ensures predicate reversibility" - likely predicate level.
                    # Or specific grounding? Symbolic planning usually checks lifted level.
                    # If action A(x) deletes P(x), we need action B(x) adding P(x).
                    # We will track by Predicate Name check.
                    if hasattr(eff.argument, 'name'):
                        predicate_name = eff.argument.name
                        deleted_preds.add(predicate_name)
                elif isinstance(eff, Predicate):
                    # Addition
                    added_preds.add(eff.name)
        
        missing_adds = deleted_preds - added_preds
        
        if not missing_adds:
            return actions
            
        # 2. Fix missing additions
        # For each missing predicate, find a compatible action and add it as an effect.
        # Or pick a random action that *could* support it.
        
        # We need to map Name back to Predicate Object to know arity/types
        pred_map = {p.name: p for p in all_predicates}
        
        # We might need to modify actions in place or create new list.
        # Since actions are immutable-ish, we replace them in the list.
        
        # Strategy: scan missing_adds, pick random action that is compatible, add effect.
        
        new_actions = list(actions)
        
        for pred_name in missing_adds:
            target_pred = pred_map.get(pred_name)
            if not target_pred:
                continue
                
            # Find candidate actions
            candidates = []
            for i, action in enumerate(new_actions):
                # Check if action parameters can support this predicate
                # We need to see if we can bind predicate terms to action params
                
                # Check if all terms of predicate can be mapped to action params
                # This is similar to _sample_predicate_for_action logic
                
                # We reuse the logic:
                term_candidate_vars_list = []
                possible = True
                for term in target_pred.terms:
                    vars_for_term = self._compatible_action_params_for_term(
                        term, action.parameters, type_ancestors
                    )
                    if not vars_for_term:
                        possible = False
                        break
                    term_candidate_vars_list.append(vars_for_term)
                
                if possible:
                    # PROACTIVE CHECK: Can we validly add this predicate to this action's effects?
                    # Since we don't know the exact grounding variables yet (logic below picks them random),
                    # we can only check if the predicate ITSELF is fundamentally inconsistent (e.g. if we add P, does it have Not(P))?
                    # But P vs Not(P) check requires variables for strict equality if grounded.
                    # However, at lifted level, if effect is Not(P(x,y)), adding P(x,y) is contradiction.
                    # If effect is Not(P(?a, ?b)), adding P(?u, ?v) might be valid if ?a!=?u.
                    # For simplicty, let's assume if the action unconditionally deletes a matching predicate type, avoid it.
                    # OR better: iterate candidates in random order and try until satisfied.
                    
                     # Actually, standard PDDL allows P(x) and Not(P(y)).
                     # So strict consistency check depends on grounding.
                     # We will filter candidates later.
                     candidates.append((i, term_candidate_vars_list))
            
            if not candidates:
                continue
            
            # Shuffle candidates to try random ones
            self.rng.shuffle(candidates)
            
            success = False
            for idx, vars_list in candidates:
                # Construct the new effect term
                # Try a few groundings?
                for _ in range(5): # retry grounding a few times
                    chosen_vars = [self.rng.choice(v_pool) for v_pool in vars_list]
                    new_effect_term = Predicate(pred_name, *chosen_vars)
                    
                    # Modify the action
                    original_action = new_actions[idx]
                    current_effects = self._extract_atomic_conditions(original_action.effect)
                    
                    # Avoid adding if it contradicts? (adding P and ~P same time)
                    if self._is_consistent(new_effect_term, current_effects):
                        current_effects.append(new_effect_term)
                        
                        # Rebuild action
                        new_eff_formula = self._list_to_formula(current_effects)
                        new_action = Action(
                            name=original_action.name,
                            parameters=original_action.parameters,
                            precondition=original_action.precondition,
                            effect=new_eff_formula
                        )
                        new_actions[idx] = new_action
                        success = True
                        break
                if success:
                    break
                
        return new_actions

    def _sample_predicate_for_action(
        self, 
        all_predicates: List[Predicate], 
        action_params: Tuple[Variable, ...], 
        type_ancestors: Dict[str, Set[str]],
    ) -> Optional[Predicate]:
        """
        Samples a predicate and grounds it with action parameters.
        Constraints:
        - Predicate arity <= len(action_params)
        - Variable types must match (if typed)
        """
        candidates: List[Tuple[Predicate, List[List[Variable]]]] = []
        for p in all_predicates:
            if p.arity <= len(action_params):
                if p.arity == 0:
                    candidates.append((p, []))
                    continue
                term_candidate_vars: List[List[Variable]] = []
                for term in p.terms:
                    compatible_vars = self._compatible_action_params_for_term(
                        term, action_params, type_ancestors
                    )
                    if not compatible_vars:
                        term_candidate_vars = []
                        break
                    term_candidate_vars.append(compatible_vars)
                if term_candidate_vars:
                    candidates.append((p, term_candidate_vars))
        
        if not candidates:
            return None
            
        chosen_pred, term_candidate_vars = self.rng.choice(candidates)
        
        if chosen_pred.arity == 0:
            return Predicate(chosen_pred.name)

        # Choose one compatible action parameter per term.
        chosen_vars = [self.rng.choice(var_pool) for var_pool in term_candidate_vars]
        
        # Create new Predicate instance with variables
        return Predicate(chosen_pred.name, *chosen_vars)

    def _normalize_type_tags(self, typed_obj: Any) -> Set[str]:
        tags = getattr(typed_obj, "type_tags", None)
        if tags:
            return {str(t) for t in tags}
        tag = getattr(typed_obj, "type_tag", None)
        if tag:
            return {str(tag)}
        return {"object"}

    def _build_type_ancestor_map(self, types_map: Dict[str, Any]) -> Dict[str, Set[str]]:
        parent: Dict[str, str] = {}
        for child_raw, par_raw in (types_map or {}).items():
            child = str(child_raw)
            if par_raw is None:
                continue
            par = str(par_raw)
            if not par or par.lower() == "none":
                continue
            if child != par:
                parent[child] = par

        ancestors: Dict[str, Set[str]] = {}
        all_types = set(parent.keys()) | set(parent.values()) | {"object"}
        for t in all_types:
            chain: Set[str] = set()
            seen: Set[str] = set()
            cur = parent.get(t)
            while cur is not None and cur not in seen:
                chain.add(cur)
                seen.add(cur)
                cur = parent.get(cur)
            ancestors[t] = chain

        return ancestors

    def _is_subtype(self, candidate_type: str, required_type: str, type_ancestors: Dict[str, Set[str]]) -> bool:
        if required_type == "object":
            return True
        if candidate_type == required_type:
            return True
        return required_type in type_ancestors.get(candidate_type, set())

    def _is_variable_compatible_with_required_tags(
        self,
        variable: Variable,
        required_tags: Set[str],
        type_ancestors: Dict[str, Set[str]],
    ) -> bool:
        variable_tags = self._normalize_type_tags(variable)
        # Variable type must be equal to, or more specific than, at least one required tag.
        for v_tag in variable_tags:
            if not any(
                self._is_subtype(v_tag, req_tag, type_ancestors) for req_tag in required_tags
            ):
                return False
        return True

    def _compatible_action_params_for_term(
        self,
        term: Any,
        action_params: Tuple[Variable, ...],
        type_ancestors: Dict[str, Set[str]],
    ) -> List[Variable]:
        required_tags = self._normalize_type_tags(term)
        return [
            param
            for param in action_params
            if self._is_variable_compatible_with_required_tags(
                param, required_tags, type_ancestors
            )
        ]

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
