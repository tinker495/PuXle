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
        merged_predicates: Dict[str, Predicate] = {}
        predicate_signatures: Dict[str, tuple[int, tuple[tuple[str, ...], ...]]] = {}
        for d in domains:
            for p in d.predicates:
                name = str(p.name)
                signature = self._predicate_signature(p)
                if name not in merged_predicates:
                    merged_predicates[name] = p
                    predicate_signatures[name] = signature
                    continue
                if predicate_signatures[name] != signature:
                    raise ValueError(
                        "Predicate name collision with incompatible signatures: "
                        f"'{name}' {predicate_signatures[name]} vs {signature}. "
                        "Rename colliding predicates before fusion."
                    )
        return list(merged_predicates.values())

    def _predicate_signature(self, predicate: Predicate) -> tuple[int, tuple[tuple[str, ...], ...]]:
        arity = int(getattr(predicate, "arity", len(getattr(predicate, "terms", ()))))
        term_type_signature: list[tuple[str, ...]] = []
        for term in getattr(predicate, "terms", ()) or ():
            term_type_signature.append(self._term_type_signature(term))
        return arity, tuple(term_type_signature)

    def _term_type_signature(self, term) -> tuple[str, ...]:
        type_tags = getattr(term, "type_tags", None)
        if type_tags:
            return tuple(sorted(str(t) for t in type_tags))
        type_tag = getattr(term, "type_tag", None)
        if type_tag:
            return (str(type_tag),)
        return ("object",)

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
