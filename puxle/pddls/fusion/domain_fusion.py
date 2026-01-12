from typing import List, Set, Dict, Optional, Union, Any
import pddl
from pddl.logic import Predicate, Constant, Variable
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
        Fuses a list of domains into a single domain.
        
        Args:
            domains: List of pddl.Domain objects to fuse.
            name: Name of the resulting domain.
            
        Returns:
            A new pddl.Domain object containing the union of all components.
        """
        if not domains:
            raise ValueError("At least one domain must be provided for fusion.")

        # 1. Merge Requirements
        requirements = self._merge_requirements(domains)

        # 2. Merge Types
        types = self._merge_types(domains)

        # 3. Merge Constants
        constants = self._merge_constants(domains)

        # 4. Merge Predicates
        predicates = self._merge_predicates(domains)

        # 5. Merge Actions
        actions = self._merge_actions(domains)

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
            # pddl library requirements are usually a set-like object or list
            if hasattr(d, "requirements"):
                 requirements.update(d.requirements)
        return requirements

    def _merge_types(self, domains: List[Domain]) -> Dict[Any, Any]:
        """
        Merges types from multiple domains.
        Returns a dictionary {type: parent} if supported, or a collection of types.
        """
        # We need to construct a unified type hierarchy.
        # Assuming pddl library uses a dictionary {type_name: parent_name} for types.
        merged_types = {} 
        
        for d in domains:
            domain_types = d.types
            
            if isinstance(domain_types, dict):
                for t, parent in domain_types.items():
                    # t and parent might be strings or 'name' objects
                    # We store them as found.
                    # Conflict resolution: if same type exists...
                    # If existing has parent 'object' (None) and new has specific parent, overwrite.
                    t_key = str(t)
                    
                    if t_key not in merged_types:
                        merged_types[t_key] = (t, parent)
                    else:
                        existing_t, existing_parent = merged_types[t_key]
                        # Check strictness of parent
                        # If existing parent is None or 'object', and new is specific, take new.
                        # Assuming 'object' is default root.
                        if str(existing_parent) == "object" and str(parent) != "object":
                             merged_types[t_key] = (t, parent)
            else:
                 # List or Set of types (legacy or different structure)
                 # If we don't have parents here, just merge names.
                 for t in domain_types:
                     t_key = str(t) # Should handle 'name' objects
                     if t_key not in merged_types:
                         merged_types[t_key] = (t, "object")
        
        # Now reconstruct the return value.
        # If the input was predominantly dict, return dict.
        # But we must return what Domain() constructor expects.
        # We can try returning a dict {type: parent}.
        # Unpack stored tuples.
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
                # Check for type conflicts? For now assume same name = same constant
        return list(merged_constants.values())

    def _merge_predicates(self, domains: List[Domain]) -> List[Predicate]:
        merged_predicates = {}
        for d in domains:
            for p in d.predicates:
                # Keying by name and arity might be safer, 
                # but pddl allows overloading? Standard STRIPS usually doesn't.
                # We assume unique signatures for simplicity.
                # Actually, same predicate signature (name + args) is fine to merge.
                sig = (p.name, tuple(getattr(arg, "type_tags", tuple()) for arg in p.terms)) 
                # p.terms usually gives variables. variable.type_tags gives types.
                # Using string representation for signature check might be easier/safer
                if p.name not in merged_predicates:
                    merged_predicates[p.name] = p
        return list(merged_predicates.values())

    def _merge_actions(self, domains: List[Domain]) -> List[Action]:
        merged_actions = []
        seen_names = {} # {name: count}
        
        for d in domains:
            for a in d.actions:
                name = a.name
                
                if name in seen_names:
                    # Rename with suffix
                    seen_names[name] += 1
                    suffix = seen_names[name]
                    new_name = f"{name}_{suffix}"
                    
                    # Create new Action with renamed name. 
                    # Assuming Action constructor signature from pddl library:
                    # Action(name, parameters, precondition, effect)
                    # We reuse everything except name.
                    renamed_action = Action(
                        name=new_name,
                        parameters=a.parameters,
                        precondition=a.precondition,
                        effect=a.effect
                    )
                    merged_actions.append(renamed_action)
                else:
                    seen_names[name] = 0
                    merged_actions.append(a)

        return merged_actions
