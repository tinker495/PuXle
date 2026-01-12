from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple


def collect_type_hierarchy(domain) -> Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Extract best-effort type hierarchy from a PDDL domain object.

    Returns (parent, ancestors, descendants).
    - parent: type -> immediate parent
    - ancestors: type -> transitive set of ancestors
    - descendants: type -> transitive set of descendants
    """
    parent: dict[str, str] = {}
    try:
        types_obj = getattr(domain, "types", None)
        if isinstance(types_obj, dict):
            for sub_t, par_t in types_obj.items():
                if sub_t and par_t and sub_t != par_t:
                    parent[str(sub_t)] = str(par_t)
        if not parent:
            th = getattr(domain, "type_hierarchy", None)
            if isinstance(th, dict):
                for sub_t, par_t in th.items():
                    if sub_t and par_t and sub_t != par_t:
                        parent[str(sub_t)] = str(par_t)
    except Exception:
        parent = {}

    ancestors: dict[str, set[str]] = {}
    descendants: dict[str, set[str]] = {}
    all_types: set[str] = set(parent.keys()) | set(parent.values())

    for t in all_types:
        ancestors[t] = set()
        p = parent.get(t)
        while p is not None and p not in ancestors[t]:
            ancestors[t].add(p)
            p = parent.get(p)

    for t in all_types:
        descendants[t] = set()
    for t, ancs in ancestors.items():
        for a in ancs:
            descendants.setdefault(a, set()).add(t)

    return parent, ancestors, descendants


def select_most_specific_types(
    type_tags: Iterable[str],
    hierarchy: Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]],
) -> list[str]:
    """Keep the most specific types using the hierarchy.

    Drops 'object' when more specific tags are present and removes any tag
    that is an ancestor of another tag in the same set.
    """
    tags_set = set(type_tags)
    if not tags_set:
        return ["object"]

    _, _ancestors, descendants = hierarchy

    if "object" in tags_set and len(tags_set) > 1:
        tags_set.discard("object")

    result: list[str] = []
    for t in tags_set:
        desc = descendants.get(t, set())
        if not any((other in desc) for other in tags_set if other != t):
            result.append(t)
    if not result:
        result = sorted(tags_set)
    return result


def extract_objects_by_type(
    problem, hierarchy: Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]], domain=None
) -> Dict[str, list[str]]:
    """Extract objects grouped by types, respecting hierarchy if available."""
    objects_by_type: dict[str, list[str]] = {}
    direct_by_type: dict[str, list[str]] = {}

    parent, ancestors, descendants = hierarchy
    _ = parent  # Unused directly but kept for clarity and symmetry

    # Handle untyped objects
    if not hasattr(problem, "objects") or not problem.objects:
        objects_by_type["object"] = []
    else:
        # Handle different object container types
        if isinstance(problem.objects, dict):
            # If objects is a dict mapping type -> object list
            for obj_type, obj_list in problem.objects.items():
                if obj_type not in direct_by_type:
                    direct_by_type[obj_type] = []
                for obj in obj_list:
                    obj_name = getattr(obj, "name", str(obj))
                    direct_by_type[obj_type].append(obj_name)
                    direct_by_type.setdefault("object", []).append(obj_name)
        else:
            # If objects is a set-like container (frozenset, set, list)
            for obj in problem.objects:
                obj_name = getattr(obj, "name", str(obj))
                if hasattr(obj, "type_tags") and obj.type_tags:
                    tags = set(obj.type_tags)
                elif hasattr(obj, "type_tag") and obj.type_tag:
                    tags = {obj.type_tag}
                else:
                    tags = set()
                
                if tags:
                    for t in select_most_specific_types(tags, hierarchy):
                        direct_by_type.setdefault(t, []).append(obj_name)
                    direct_by_type.setdefault("object", []).append(obj_name)
                else:
                    direct_by_type.setdefault("object", []).append(obj_name)

    # Handle domain constants
    if domain is not None and hasattr(domain, "constants") and domain.constants:
        for obj in domain.constants:
            obj_name = getattr(obj, "name", str(obj))
            if hasattr(obj, "type_tags") and obj.type_tags:
                tags = set(obj.type_tags)
            elif hasattr(obj, "type_tag") and obj.type_tag:
                tags = {obj.type_tag}
            else:
                tags = set()

            if tags:
                for t in select_most_specific_types(tags, hierarchy):
                    direct_by_type.setdefault(t, []).append(obj_name)
                direct_by_type.setdefault("object", []).append(obj_name)
            else:
                direct_by_type.setdefault("object", []).append(obj_name)

    # Expand direct to include descendants under supertypes
    if direct_by_type:
        # Initialize with direct entries (deduped)
        for t, objs in direct_by_type.items():
            seen = set()
            deduped = []
            for o in objs:
                if o not in seen:
                    seen.add(o)
                    deduped.append(o)
            objects_by_type[t] = deduped

        all_types = set(direct_by_type.keys()) | set(ancestors.keys()) | set(descendants.keys())
        for t in sorted(all_types):
            base = list(objects_by_type.get(t, []))
            seen = set(base)
            for d in sorted(descendants.get(t, set())):
                for o in direct_by_type.get(d, []):
                    if o not in seen:
                        seen.add(o)
                        base.append(o)
            objects_by_type[t] = base

    return objects_by_type
