import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from pddl.core import Action, Formula
from pddl.logic import Variable
from pddl.logic.base import And, Not
from pddl.logic.predicates import Predicate

from puxle.pddls.fusion.formula_facts import flatten_formula
from puxle.pddls.fusion.type_facts import (
    build_type_ancestor_map,
    strict_compatible_candidates,
)


@dataclass
class FusionParams:
    prob_add_pre: float = 0.1
    prob_add_eff: float = 0.1
    prob_rem_pre: float = 0.05
    prob_rem_eff: float = 0.05
    prob_neg: float = 0.1
    rev_flag: bool = True  # Ensures reversibility
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
        types_map: Dict[str, Any],
    ) -> List[Action]:
        """
        Apply stochastic modifications to a list of actions.
        """
        type_ancestors = build_type_ancestor_map(types_map)
        modified_actions = []
        for action in actions:
            modified = self._modify_single_action(
                action, all_predicates, type_ancestors
            )
            modified_actions.append(modified)

        if self.params.rev_flag:
            modified_actions = self._enforce_reversibility(
                modified_actions, all_predicates, types_map, type_ancestors
            )

        return modified_actions

    def _modify_single_action(
        self,
        action: Action,
        all_predicates: List[Predicate],
        type_ancestors: Dict[str, Set[str]],
    ) -> Action:
        # PDDL library objects are immutable, so we rebuild precondition/effect lists.
        preconditions = flatten_formula(action.precondition)
        effects = flatten_formula(action.effect)
        params = action.parameters

        if self.rng.random() < self.params.prob_add_pre:
            new_pre = self._sample_predicate_for_action(
                all_predicates, params, type_ancestors
            )
            if new_pre:
                if self.rng.random() < self.params.prob_neg:
                    new_pre = Not(new_pre)
                preconditions.append(new_pre)

        if preconditions and self.rng.random() < self.params.prob_rem_pre:
            idx = self.rng.randint(0, len(preconditions) - 1)
            preconditions.pop(idx)

        if self.rng.random() < self.params.prob_add_eff:
            new_eff = self._sample_predicate_for_action(
                all_predicates, params, type_ancestors
            )
            if new_eff:
                is_neg = self.rng.random() < self.params.prob_neg
                term_to_add = Not(new_eff) if is_neg else new_eff
                if self._is_consistent(term_to_add, effects):
                    effects.append(term_to_add)

        if effects and self.rng.random() < self.params.prob_rem_eff:
            idx = self.rng.randint(0, len(effects) - 1)
            effects.pop(idx)

        new_precondition = self._list_to_formula(preconditions)
        new_effect = self._list_to_formula(effects)

        return Action(
            name=action.name,
            parameters=action.parameters,
            precondition=new_precondition,
            effect=new_effect,
        )

    def _enforce_reversibility(
        self,
        actions: List[Action],
        all_predicates: List[Predicate],
        types_map: Dict[str, Any],
        type_ancestors: Dict[str, Set[str]],
    ) -> List[Action]:
        """
        Ensures that for every predicate P, if an action deletes P, there is an action that adds P.

        Reversibility is enforced at the lifted predicate-name level: a predicate
        deleted by some action but never added gets a compatible add-effect grafted
        onto a randomly chosen action whose parameters can bind the predicate's terms.
        """
        deleted_preds = set()
        added_preds = set()

        for action in actions:
            effects = flatten_formula(action.effect)
            for eff in effects:
                if isinstance(eff, Not):
                    if hasattr(eff.argument, "name"):
                        deleted_preds.add(eff.argument.name)
                elif isinstance(eff, Predicate):
                    added_preds.add(eff.name)

        missing_adds = deleted_preds - added_preds

        if not missing_adds:
            return actions

        pred_map = {p.name: p for p in all_predicates}
        new_actions = list(actions)

        for pred_name in missing_adds:
            target_pred = pred_map.get(pred_name)
            if not target_pred:
                continue

            # Candidate actions are those whose parameters can bind every term of
            # the target predicate. Grounding consistency is checked per attempt below.
            candidates = []
            for i, action in enumerate(new_actions):
                term_candidate_vars_list = []
                possible = True
                for term in target_pred.terms:
                    vars_for_term = strict_compatible_candidates(
                        action.parameters, term, type_ancestors
                    )
                    if not vars_for_term:
                        possible = False
                        break
                    term_candidate_vars_list.append(vars_for_term)

                if possible:
                    candidates.append((i, term_candidate_vars_list))

            if not candidates:
                continue

            self.rng.shuffle(candidates)

            success = False
            for idx, vars_list in candidates:
                for _ in range(5):  # retry grounding a few times
                    chosen_vars = [self.rng.choice(v_pool) for v_pool in vars_list]
                    new_effect_term = Predicate(pred_name, *chosen_vars)

                    original_action = new_actions[idx]
                    current_effects = flatten_formula(original_action.effect)

                    if self._is_consistent(new_effect_term, current_effects):
                        current_effects.append(new_effect_term)

                        new_eff_formula = self._list_to_formula(current_effects)
                        new_action = Action(
                            name=original_action.name,
                            parameters=original_action.parameters,
                            precondition=original_action.precondition,
                            effect=new_eff_formula,
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
                    compatible_vars = strict_compatible_candidates(
                        action_params, term, type_ancestors
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

    def _list_to_formula(self, dry_list: List[Formula]) -> Optional[Formula]:
        if not dry_list:
            return And()  # Return empty And instead of None
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
