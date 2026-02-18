from typing import Any, Dict, List, Set
from pddl.logic import Constant, Predicate
from pddl.core import Domain, Action
from pddl.requirements import Requirements

class DomainFusion:
    """
    Engine for fusing multiple PDDL domains into a single domain.
    """
    def __init__(self):
        pass

    def fuse_domains(self, domains: List[Domain], name: str = "fused-domain") -> Domain:
        """
        Fuses a list of domains into a single domain using disjoint union.
        
        Args:
            domains: List of pddl.Domain objects to fuse.
            name: Name of the resulting domain.
            
        Returns:
            A new pddl.Domain object containing the disjoint union of components.
        """
        if not domains:
            raise ValueError("At least one domain must be provided for fusion.")

        # 1. Merge Requirements
        requirements = self._merge_requirements(domains)

        # 2. Merge Types
        # Types are merged by name (shared type system assumption)
        types = self._merge_types(domains)

        # 3. Merge Constants
        # Constants are also merged by name.
        constants = self._merge_constants(domains)

        # 4. Merge Predicates (Disjoint)
        predicates = self._merge_predicates_disjoint(domains)

        # 5. Merge Actions (Disjoint)
        actions = self._merge_actions_disjoint(domains)

        # TODO: derived predicates if any

        return Domain(
            name=name,
            requirements=requirements,
            types=types,
            constants=constants,
            predicates=predicates,
            actions=actions
        )

    def _merge_requirements(self, domains: List[Domain]) -> Set[Requirements]:
        requirements = set()
        for d in domains:
            if hasattr(d, "requirements"):
                 requirements.update(d.requirements)
        return requirements

    def _merge_types(self, domains: List[Domain]) -> Dict[Any, Any]:
        """
        Merges types from multiple domains.
        Returns a dictionary {type: parent} if supported, or a collection of types.
        """
        merged_types = {} 
        
        for d in domains:
            domain_types = d.types
            
            if isinstance(domain_types, dict):
                for t, parent in domain_types.items():
                    t_key = str(t)
                    
                    if t_key not in merged_types:
                        merged_types[t_key] = (t, parent)
                    else:
                        existing_t, existing_parent = merged_types[t_key]
                        if str(existing_parent) == "object" and str(parent) != "object":
                             merged_types[t_key] = (t, parent)
            else:
                 for t in domain_types:
                     t_key = str(t)
                     if t_key not in merged_types:
                         merged_types[t_key] = (t, "object")
        
        result = {}
        for t_key, (t, parent) in merged_types.items():
            result[t] = parent
            
        return result

    def _merge_constants(self, domains: List[Domain]) -> List[Constant]:
        merged_constants = {}
        for d in domains:
            for c in d.constants:
                if c.name not in merged_constants:
                    merged_constants[c.name] = c
        return list(merged_constants.values())

    def _merge_predicates_disjoint(self, domains: List[Domain]) -> List[Predicate]:
        """
        Merges predicates by prefixing them with domain index/name to ensure disjointness.
        """
        merged_predicates = []
        
        for idx, d in enumerate(domains):
            prefix = f"dom{idx}_"
            for p in d.predicates:
                new_name = f"{prefix}{p.name}"
                # Reconstruct predicate with new name
                new_p = Predicate(new_name, *p.terms)
                merged_predicates.append(new_p)
                
        return merged_predicates

    def _merge_actions_disjoint(self, domains: List[Domain]) -> List[Action]:
        """
        Merges actions by prefixing them with domain index/name to ensure disjointness.
        Also updates their preconditions/effects to refer to the prefixed predicates.
        """
        merged_actions = []
        
        for idx, d in enumerate(domains):
            prefix = f"dom{idx}_"
            for a in d.actions:
                new_name = f"{prefix}{a.name}"
                
                # We must also rename all predicates appearing in precond/effect
                # to match the disjoint predicate names.
                new_pre = self._rename_predicates_in_formula(a.precondition, prefix)
                new_eff = self._rename_predicates_in_formula(a.effect, prefix)
                
                renamed_action = Action(
                    name=new_name,
                    parameters=a.parameters,
                    precondition=new_pre,
                    effect=new_eff
                )
                merged_actions.append(renamed_action)

        return merged_actions

    def _rename_predicates_in_formula(self, formula, prefix: str):
        """
        Recursively renames predicates in a formula with the given prefix.
        """
        if formula is None:
            return None
            
        # Handle And, Not, Or, Imply, etc.
        # Assuming minimal support for And/Not/Predicate for now based on previous code.
        
        if hasattr(formula, "operands"): # And, Or
            # Reconstruct same type
            new_ops = [self._rename_predicates_in_formula(op, prefix) for op in formula.operands]
            return type(formula)(*new_ops)
            
        if hasattr(formula, "argument"): # Not
            return type(formula)(self._rename_predicates_in_formula(formula.argument, prefix))
            
        if hasattr(formula, "name") and hasattr(formula, "terms"): # Predicate
            # Rename!
            new_name = f"{prefix}{formula.name}"
            return type(formula)(new_name, *formula.terms)
            
        return formula
