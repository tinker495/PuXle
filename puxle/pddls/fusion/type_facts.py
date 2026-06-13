"""Shared PDDL type facts and compatibility policies for fusion adapters."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Set


def normalise_type_tags(typed_obj: Any) -> Set[str]:
    """Return a stable set of type tags for pddl objects, variables, or terms."""
    tags = getattr(typed_obj, "type_tags", None)
    if tags:
        if isinstance(tags, str):
            return {tags}
        return {str(tag) for tag in tags}

    tag = getattr(typed_obj, "type_tag", None)
    if tag:
        return {str(tag)}

    return {"object"}


def build_type_ancestor_map(types_map: Mapping[Any, Any] | None) -> dict[str, Set[str]]:
    """Build a type -> ancestors map from a best-effort child->parent mapping."""
    parent: dict[str, str] = {}
    for child_raw, parent_raw in (types_map or {}).items():
        child = str(child_raw)
        if parent_raw is None:
            continue
        parent_name = str(parent_raw)
        if not parent_name or parent_name.lower() == "none":
            continue
        if child != parent_name:
            parent[child] = parent_name

    ancestors: dict[str, Set[str]] = {}
    all_types = set(parent.keys()) | set(parent.values()) | {"object"}
    for type_name in all_types:
        chain: Set[str] = set()
        seen: Set[str] = set()
        current = parent.get(type_name)
        while current is not None and current not in seen:
            chain.add(current)
            seen.add(current)
            current = parent.get(current)
        ancestors[type_name] = chain

    return ancestors


def domain_type_parent_map(domain: Any) -> dict[str, str]:
    """Return {type_name: parent_type_name} for a PDDL domain."""
    mapping: dict[str, str] = {}
    domain_types = getattr(domain, "types", None)
    if isinstance(domain_types, Mapping):
        for type_name, parent in domain_types.items():
            name = str(type_name)
            parent_name = "object" if parent is None else str(parent)
            mapping[name] = parent_name
    elif domain_types:
        for type_name in domain_types:
            mapping[str(type_name)] = "object"
    return mapping


def is_subtype(
    candidate_type: str,
    required_type: str,
    type_ancestors: Mapping[str, Set[str]] | None = None,
) -> bool:
    """Return whether candidate_type can stand in for required_type."""
    if required_type == "object":
        return True
    if candidate_type == required_type:
        return True
    return required_type in (type_ancestors or {}).get(candidate_type, set())


def strict_compatible_terms(
    candidate: Any,
    required: Any,
    type_ancestors: Mapping[str, Set[str]] | None = None,
) -> bool:
    """Return whether every candidate tag can bind at least one required tag."""
    required_tags = normalise_type_tags(required)
    candidate_tags = normalise_type_tags(candidate)

    return all(
        any(
            is_subtype(candidate_tag, required_tag, type_ancestors)
            for required_tag in required_tags
        )
        for candidate_tag in candidate_tags
    )


def strict_compatible_candidates(
    candidates: Sequence[Any],
    required: Any,
    type_ancestors: Mapping[str, Set[str]] | None = None,
) -> list[Any]:
    """Filter candidates with strict all-candidate-tags compatibility."""
    return [
        candidate
        for candidate in candidates
        if strict_compatible_terms(candidate, required, type_ancestors)
    ]


def any_match_compatible_term(
    candidate: Any,
    required: Any,
    type_ancestors: Mapping[str, Set[str]] | None = None,
) -> bool:
    """Return whether any candidate tag can bind any required tag."""
    required_tags = normalise_type_tags(required)
    candidate_tags = normalise_type_tags(candidate)

    return any(
        is_subtype(candidate_tag, required_tag, type_ancestors)
        for candidate_tag in candidate_tags
        for required_tag in required_tags
    )


def any_match_compatible_candidates(
    candidates: Sequence[Any],
    required: Any,
    type_ancestors: Mapping[str, Set[str]] | None = None,
) -> list[Any]:
    """Filter candidates with generator-style any-match compatibility."""
    return [
        candidate
        for candidate in candidates
        if any_match_compatible_term(candidate, required, type_ancestors)
    ]
