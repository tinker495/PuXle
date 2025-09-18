from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pddl

from puxle.pddls.pddl import PDDL
from puxle.pddls.validation import bfs_validate
from .type_system import collect_type_hierarchy


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    """Configuration describing a fusion run."""

    domain_a: str
    problem_a: str
    domain_b: str
    problem_b: str
    name: str = "fused-domain"
    prob_add_pre: float = 0.0
    prob_add_eff: float = 0.0
    prob_rem_pre: float = 0.0
    prob_rem_eff: float = 0.0
    prob_negate: float = 0.0
    ensure_reversible: bool = False
    num_objects: Optional[int] = None
    rollout_depth: int = 5
    goal_sample_size: Optional[int] = None
    seed: Optional[int] = None
    validation_depth: Optional[int] = None

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type: Optional[str] = None


@dataclass(frozen=True)
class PredicateSchema:
    name: str
    parameters: Tuple[ParameterSpec, ...]

    @property
    def arity(self) -> int:
        return len(self.parameters)


@dataclass(frozen=True)
class SchematicLiteral:
    name: str
    arguments: Tuple[str, ...]
    positive: bool = True

    def signature(self) -> Tuple[str, Tuple[str, ...], bool]:
        return (self.name, self.arguments, self.positive)

    def to_pddl(self) -> str:
        inner = f"({self.name}{(' ' + ' '.join(self.arguments)) if self.arguments else ''})"
        return inner if self.positive else f"(not {inner})"


@dataclass
class ActionSchema:
    name: str
    parameters: Tuple[ParameterSpec, ...]
    preconditions: List[SchematicLiteral] = field(default_factory=list)
    effects: List[SchematicLiteral] = field(default_factory=list)

    def ensure_effect(self) -> None:
        if not self.effects:
            raise ValueError(f"Action '{self.name}' must have at least one effect")


# ---------------------------------------------------------------------------
# Helpers for manipulating PDDL objects
# ---------------------------------------------------------------------------


def _prefixed(new_name: str, existing: Iterable[str]) -> str:
    existing_set = set(existing)
    if new_name not in existing_set:
        return new_name
    base = new_name
    suffix = 1
    while f"{base}-{suffix}" in existing_set:
        suffix += 1
    return f"{base}-{suffix}"


def _parameter_name(raw_name: str) -> str:
    return raw_name if raw_name.startswith("?") else f"?{raw_name}"


def _coerce_type_tag(raw) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return _coerce_type_tag(raw[0] if raw else None)
    if isinstance(raw, (set, frozenset)):
        return _coerce_type_tag(next(iter(raw)) if raw else None)
    value = str(raw)
    return value if value and value != "object" else None


def _parameter_spec(item) -> ParameterSpec:
    name = getattr(item, "name", str(item))
    spec_name = _parameter_name(name)
    param_type: Optional[str] = None
    if hasattr(item, "type_tags") and item.type_tags:
        param_type = _coerce_type_tag(item.type_tags)
    return ParameterSpec(spec_name, param_type)


def _rename_formula_symbols(formula, rename_map: Dict[str, str]) -> None:
    if formula is None:
        return
    if hasattr(formula, "name") and formula.name in rename_map:
        formula.name = rename_map[formula.name]
    if hasattr(formula, "argument"):
        _rename_formula_symbols(formula.argument, rename_map)
    if hasattr(formula, "parts"):
        for part in getattr(formula, "parts", []) or []:
            _rename_formula_symbols(part, rename_map)
    if hasattr(formula, "operands"):
        for operand in getattr(formula, "operands", []) or []:
            _rename_formula_symbols(operand, rename_map)


def _rename_problem_atoms(problem, rename_map: Dict[str, str]) -> None:
    for atom in getattr(problem, "init", []) or []:
        if atom.name in rename_map:
            atom.name = rename_map[atom.name]
    goal = getattr(problem, "goal", None)
    if goal is not None:
        _rename_formula_symbols(goal, rename_map)


def _rename_predicates(domain, problem, existing: Dict[str, str], prefix: str) -> Dict[str, str]:
    rename_map: Dict[str, str] = {}
    for predicate in getattr(domain, "predicates", []) or []:
        name = predicate.name
        if name in existing:
            new_name = _prefixed(f"{prefix}-{name}", existing)
            rename_map[name] = new_name
            predicate.name = new_name
        existing[predicate.name] = predicate.name
    if rename_map:
        for action in getattr(domain, "actions", []) or []:
            _rename_formula_symbols(action.precondition, rename_map)
            _rename_formula_symbols(action.effect, rename_map)
        _rename_problem_atoms(problem, rename_map)
    return rename_map


def _rename_actions(domain, existing: Dict[str, str], prefix: str) -> None:
    for action in getattr(domain, "actions", []) or []:
        name = action.name
        if name in existing:
            action.name = _prefixed(f"{prefix}-{name}", existing)
        existing[action.name] = action.name


def _extract_literals(formula, parameterize: bool = True) -> List[SchematicLiteral]:
    if formula is None:
        return []
    literals: List[SchematicLiteral] = []

    def _visit(node, positive: bool = True) -> None:
        if node is None:
            return
        if hasattr(node, "name") and hasattr(node, "terms"):
            args = []
            for term in getattr(node, "terms", []) or []:
                raw_name = getattr(term, "name", str(term))
                arg = _parameter_name(raw_name) if parameterize else str(raw_name)
                args.append(arg)
            literals.append(SchematicLiteral(node.name, tuple(args), positive))
            return
        if hasattr(node, "argument"):
            _visit(node.argument, not positive)
        if hasattr(node, "parts"):
            for part in getattr(node, "parts", []) or []:
                _visit(part, positive)
        if hasattr(node, "operands"):
            for operand in getattr(node, "operands", []) or []:
                _visit(operand, positive)

    _visit(formula)
    return literals


def _collect_predicates(domain) -> List[PredicateSchema]:
    predicates: List[PredicateSchema] = []
    for predicate in getattr(domain, "predicates", []) or []:
        params = tuple(_parameter_spec(term) for term in getattr(predicate, "terms", []) or [])
        predicates.append(PredicateSchema(predicate.name, params))
    return predicates


def _collect_actions(domain) -> List[ActionSchema]:
    actions: List[ActionSchema] = []
    for action in getattr(domain, "actions", []) or []:
        parameters = tuple(_parameter_spec(param) for param in getattr(action, "parameters", []) or [])
        preconditions = _extract_literals(getattr(action, "precondition", None), parameterize=True)
        effects = _extract_literals(getattr(action, "effect", None), parameterize=True)
        actions.append(ActionSchema(action.name, parameters, preconditions, effects))
    return actions


def _render_parameter_list(parameters: Sequence[ParameterSpec]) -> str:
    pieces: List[str] = []
    for param in parameters:
        if param.type and param.type != "object":
            pieces.append(f"{param.name} - {param.type}")
        else:
            pieces.append(param.name)
    return " ".join(pieces)


def _literal_block(literals: Sequence[SchematicLiteral]) -> str:
    if not literals:
        return "()"
    rendered = [lit.to_pddl() for lit in literals]
    if len(rendered) == 1:
        return rendered[0]
    return "(and " + " ".join(rendered) + ")"


def _build_types_lines(type_parents: Dict[str, Optional[str]]) -> List[str]:
    if not type_parents:
        return []
    parent_to_children: Dict[Optional[str], List[str]] = defaultdict(list)
    all_types = set(type_parents.keys()) | {p for p in type_parents.values() if p is not None}
    for typ in all_types:
        parent_to_children.setdefault(type_parents.get(typ), [])
    for typ, parent in type_parents.items():
        parent_to_children[parent].append(typ)
    lines = ["  (:types"]
    for parent, children in sorted(parent_to_children.items(), key=lambda item: (item[0] is not None, item[0] or "")):
        if not children:
            continue
        child_list = " ".join(sorted(set(children)))
        if parent:
            lines.append(f"    {child_list} - {parent}")
        else:
            lines.append(f"    {child_list}")
    lines.append("  )")
    return lines


def _domain_to_pddl(
    name: str,
    predicates: Sequence[PredicateSchema],
    actions: Sequence[ActionSchema],
    type_parents: Dict[str, Optional[str]],
) -> str:
    predicate_lines = []
    for predicate in predicates:
        params = _render_parameter_list(predicate.parameters)
        predicate_lines.append(f"    ({predicate.name}{(' ' + params) if params else ''})")

    action_blocks = "\n".join(
        "\n".join(
            [
                f"  (:action {action.name}",
                f"    :parameters ({_render_parameter_list(action.parameters)})" if action.parameters else "    :parameters ()",
                f"    :precondition {_literal_block(action.preconditions)}",
                f"    :effect {_literal_block(action.effects)}",
                "  )",
            ]
        )
        for action in actions
    )

    requirements = {":strips"}
    if any(not lit.positive for action in actions for lit in action.preconditions):
        requirements.add(":negative-preconditions")
    if any(param.type and param.type != "object" for predicate in predicates for param in predicate.parameters):
        requirements.add(":typing")
    requirement_line = "  (:requirements " + " ".join(sorted(requirements)) + ")"

    lines = [f"(define (domain {name})", requirement_line]
    if ":typing" in requirements:
        lines.extend(_build_types_lines(type_parents))
    lines.extend([
        "  (:predicates",
        *predicate_lines,
        "  )",
        action_blocks,
        ")",
    ])
    return "\n".join(lines)


def _problem_to_pddl(
    name: str,
    domain_name: str,
    objects: Sequence[str],
    init_atoms: Sequence[str],
    goal_atoms: Sequence[str],
) -> str:
    object_line = " ".join(objects)
    init_lines = "\n".join(f"    {atom}" for atom in sorted(init_atoms))
    if not goal_atoms:
        goal_block = "()"
    elif len(goal_atoms) == 1:
        goal_block = goal_atoms[0]
    else:
        goal_block = "(and " + " ".join(goal_atoms) + ")"
    return "\n".join(
        [
            f"(define (problem {name})",
            f"  (:domain {domain_name})",
            f"  (:objects {object_line})" if object_line else "  (:objects)",
            "  (:init",
            init_lines,
            "  )",
            f"  (:goal {goal_block})",
            ")",
        ]
    )


def _collect_problem_objects(problem) -> List[str]:
    objects: List[str] = []
    for obj in getattr(problem, "objects", []) or []:
        objects.append(getattr(obj, "name", str(obj)))
    return objects


def _collect_problem_atoms(problem) -> List[str]:
    atoms: List[str] = []
    for atom in getattr(problem, "init", []) or []:
        args = [getattr(arg, "name", str(arg)) for arg in getattr(atom, "terms", []) or []]
        atoms.append(f"({atom.name}{(' ' + ' '.join(args)) if args else ''})")
    return atoms


def _collect_goal_atoms(problem) -> List[str]:
    return [lit.to_pddl() for lit in _extract_literals(getattr(problem, "goal", None), parameterize=False) if lit.positive]


def _parameter_names(parameters: Sequence[ParameterSpec | str]) -> List[str]:
    result: List[str] = []
    for param in parameters:
        if isinstance(param, ParameterSpec):
            result.append(param.name)
        else:
            result.append(str(param))
    return result


def _dedup_literals(literals: Sequence[SchematicLiteral]) -> List[SchematicLiteral]:
    seen: set[Tuple[str, Tuple[str, ...], bool]] = set()
    result: List[SchematicLiteral] = []
    for lit in literals:
        sig = lit.signature()
        if sig in seen:
            continue
        seen.add(sig)
        result.append(lit)
    return result


def _sample_literal(
    predicate_pool: Sequence[PredicateSchema],
    parameters: Sequence[ParameterSpec | str],
    rng: np.random.Generator,
    positive: bool,
) -> Optional[SchematicLiteral]:
    if not predicate_pool:
        return None
    predicate = rng.choice(predicate_pool)
    param_names = _parameter_names(parameters)
    if predicate.arity > 0 and not param_names:
        return None
    if predicate.arity == 0:
        return SchematicLiteral(predicate.name, tuple(), positive)
    args = [rng.choice(param_names) for _ in range(predicate.arity)]
    return SchematicLiteral(predicate.name, tuple(args), positive)


def _mutate_literals(
    literals: List[SchematicLiteral],
    predicate_pool: Sequence[PredicateSchema],
    parameters: Sequence[ParameterSpec | str],
    rng: np.random.Generator,
    prob_add: float,
    prob_remove: float,
    prob_negate: float,
    min_size: int = 0,
) -> List[SchematicLiteral]:
    result = list(literals)
    if prob_remove > 0 and result:
        filtered: List[SchematicLiteral] = []
        for lit in result:
            if rng.random() < prob_remove:
                continue
            filtered.append(lit)
        result = filtered
    if prob_add > 0 and rng.random() < prob_add:
        positive = rng.random() >= prob_negate
        candidate = _sample_literal(predicate_pool, parameters, rng, positive)
        if candidate is not None and candidate.signature() not in {lit.signature() for lit in result}:
            result.append(candidate)
    if min_size and len(result) < min_size:
        while len(result) < min_size:
            candidate = _sample_literal(predicate_pool, parameters, rng, True)
            if candidate is None or candidate.signature() in {lit.signature() for lit in result}:
                break
            result.append(candidate)
    return result


# ---------------------------------------------------------------------------
# Core fusion routine
# ---------------------------------------------------------------------------


def fuse_domains(config: FusionConfig, output_dir: Path) -> Tuple[Path, Path, Dict[str, object]]:
    rng = config.rng()
    domain_a = pddl.parse_domain(config.domain_a)
    problem_a = pddl.parse_problem(config.problem_a)
    domain_b = pddl.parse_domain(config.domain_b)
    problem_b = pddl.parse_problem(config.problem_b)

    existing_predicates: Dict[str, str] = {predicate.name: predicate.name for predicate in getattr(domain_a, "predicates", []) or []}
    predicate_mapping = _rename_predicates(domain_b, problem_b, existing_predicates, "d2")

    existing_actions: Dict[str, str] = {action.name: action.name for action in getattr(domain_a, "actions", []) or []}
    _rename_actions(domain_b, existing_actions, "d2")

    predicates_dict: Dict[str, PredicateSchema] = {}
    for predicate in _collect_predicates(domain_a) + _collect_predicates(domain_b):
        predicates_dict[predicate.name] = predicate
    predicates = sorted(predicates_dict.values(), key=lambda p: p.name)

    actions_raw = _collect_actions(domain_a) + _collect_actions(domain_b)
    actions_raw = sorted(actions_raw, key=lambda act: act.name)

    type_parents_a, _, _ = collect_type_hierarchy(domain_a)
    type_parents_b, _, _ = collect_type_hierarchy(domain_b)
    type_parents: Dict[str, Optional[str]] = {}
    type_parents.update(type_parents_a)
    for typ, parent in type_parents_b.items():
        type_parents.setdefault(typ, parent)
    for predicate in predicates:
        for param in predicate.parameters:
            if param.type and param.type != "object":
                type_parents.setdefault(param.type, "object")

    processed_actions: List[ActionSchema] = []
    inverse_actions: List[ActionSchema] = []

    for action in actions_raw:
        action.preconditions = _dedup_literals(
            _mutate_literals(
                action.preconditions,
                predicates,
                action.parameters,
                rng,
                config.prob_add_pre,
                config.prob_rem_pre,
                config.prob_negate,
                min_size=0,
            )
        )
        action.effects = _dedup_literals(
            _mutate_literals(
                action.effects,
                predicates,
                action.parameters,
                rng,
                config.prob_add_eff,
                config.prob_rem_eff,
                config.prob_negate,
                min_size=1,
            )
        )
        action.ensure_effect()
        processed_actions.append(action)

        for param in action.parameters:
            if param.type and param.type != "object":
                type_parents.setdefault(param.type, "object")

        if config.ensure_reversible:
            inverse_pre = _dedup_literals(action.preconditions + action.effects)
            inverse_eff = _dedup_literals(
                [SchematicLiteral(lit.name, lit.arguments, not lit.positive) for lit in action.effects]
            )
            inverse_name = _prefixed(f"{action.name}-inverse", [a.name for a in processed_actions] + [a.name for a in inverse_actions])
            inverse_action = ActionSchema(
                name=inverse_name,
                parameters=action.parameters,
                preconditions=inverse_pre,
                effects=inverse_eff,
            )
            inverse_action.ensure_effect()
            inverse_actions.append(inverse_action)

    actions = processed_actions + inverse_actions

    domain_name = config.name
    domain_path = output_dir / f"{domain_name}.pddl"
    problem_name = f"{domain_name}-problem"
    problem_path = output_dir / f"{problem_name}.pddl"

    domain_str = _domain_to_pddl(domain_name, predicates, actions, type_parents)

    objects = sorted(set(_collect_problem_objects(problem_a) + _collect_problem_objects(problem_b)))
    if config.num_objects is not None:
        if config.num_objects < len(objects):
            selection = rng.choice(objects, size=config.num_objects, replace=False)
            objects = sorted(list(np.atleast_1d(selection).tolist()))
        elif config.num_objects > len(objects):
            counter = 0
            while len(objects) < config.num_objects:
                candidate = f"obj_{counter}"
                counter += 1
                if candidate in objects:
                    continue
                objects.append(candidate)
            objects = sorted(objects)

    init_atoms = sorted(set(_collect_problem_atoms(problem_a) + _collect_problem_atoms(problem_b)))
    placeholder_goal = sorted(set(_collect_goal_atoms(problem_a) + _collect_goal_atoms(problem_b)))
    problem_str_initial = _problem_to_pddl(problem_name, domain_name, objects, init_atoms, placeholder_goal)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        temp_domain = tmp_dir / "domain.pddl"
        temp_problem = tmp_dir / "problem.pddl"
        temp_domain.write_text(domain_str, encoding="utf-8")
        temp_problem.write_text(problem_str_initial, encoding="utf-8")

        env = PDDL(str(temp_domain), str(temp_problem))
        solve_config, state = env.get_inits(jax.random.PRNGKey(rng.integers(0, 1 << 32)))
        current_state = state
        for _ in range(max(config.rollout_depth, 0)):
            neighbours, costs = env.get_neighbours(solve_config, current_state, filled=True)
            applicable = np.array(jnp.where(jnp.isfinite(costs))[0])
            if applicable.size == 0:
                break
            chosen = int(rng.choice(applicable))
            current_state = jax.tree_util.tree_map(lambda x: x[chosen], neighbours)

        final_atoms = env.state_to_atom_set(current_state)
        initial_atom_set = set(init_atoms)
        candidate_goals = list(final_atoms - initial_atom_set)
        if not candidate_goals:
            candidate_goals = list(final_atoms)
        if config.goal_sample_size is not None and config.goal_sample_size < len(candidate_goals):
            goal_choice = rng.choice(candidate_goals, size=config.goal_sample_size, replace=False)
            goal_atoms = list(np.atleast_1d(goal_choice).tolist())
        else:
            goal_atoms = candidate_goals
        if not goal_atoms:
            goal_atoms = init_atoms[:1]

    problem_str_final = _problem_to_pddl(problem_name, domain_name, objects, init_atoms, sorted(goal_atoms))

    domain_path.write_text(domain_str, encoding="utf-8")
    problem_path.write_text(problem_str_final, encoding="utf-8")

    metadata: Dict[str, object] = {
        "predicate_renames": predicate_mapping,
        "objects": objects,
        "init_atoms": init_atoms,
        "goal_atoms": sorted(goal_atoms),
        "ensure_reversible": config.ensure_reversible,
        "config": {
            "domain_a": config.domain_a,
            "problem_a": config.problem_a,
            "domain_b": config.domain_b,
            "problem_b": config.problem_b,
            "probabilities": {
                "prob_add_pre": config.prob_add_pre,
                "prob_add_eff": config.prob_add_eff,
                "prob_rem_pre": config.prob_rem_pre,
                "prob_rem_eff": config.prob_rem_eff,
                "prob_negate": config.prob_negate,
            },
            "ensure_reversible": config.ensure_reversible,
            "num_objects": config.num_objects,
            "rollout_depth": config.rollout_depth,
            "goal_sample_size": config.goal_sample_size,
            "seed": config.seed,
            "validation_depth": config.validation_depth,
        },
    }
    if config.validation_depth is not None and config.validation_depth >= 0:
        validation = bfs_validate(domain_path, problem_path, depth_limit=config.validation_depth)
        metadata["validation"] = {
            "passed": validation.passed,
            "explored_states": validation.explored_states,
            "depth_limit": validation.depth_limit,
        }
    else:
        metadata["validation"] = None

    meta_path = output_dir / f"{problem_name}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return domain_path, problem_path, metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fuse two PDDL domains into a generated domain/problem pair")
    parser.add_argument("--domain-a", required=True)
    parser.add_argument("--problem-a", required=True)
    parser.add_argument("--domain-b", required=True)
    parser.add_argument("--problem-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="fused-domain")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prob-add-pre", type=float, default=0.0)
    parser.add_argument("--prob-add-eff", type=float, default=0.0)
    parser.add_argument("--prob-rem-pre", type=float, default=0.0)
    parser.add_argument("--prob-rem-eff", type=float, default=0.0)
    parser.add_argument("--prob-neg", type=float, default=0.0)
    parser.add_argument("--ensure-reversible", action="store_true")
    parser.add_argument("--num-objects", type=int, default=None)
    parser.add_argument("--rollout-depth", type=int, default=5)
    parser.add_argument("--goal-size", type=int, default=None)
    parser.add_argument("--validate-depth", type=int, default=6)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FusionConfig(
        domain_a=args.domain_a,
        problem_a=args.problem_a,
        domain_b=args.domain_b,
        problem_b=args.problem_b,
        name=args.name,
        prob_add_pre=args.prob_add_pre,
        prob_add_eff=args.prob_add_eff,
        prob_rem_pre=args.prob_rem_pre,
        prob_rem_eff=args.prob_rem_eff,
        prob_negate=args.prob_neg,
        ensure_reversible=args.ensure_reversible,
        num_objects=args.num_objects,
        rollout_depth=args.rollout_depth,
        goal_sample_size=args.goal_size,
        seed=args.seed,
        validation_depth=None if args.skip_validation else args.validate_depth,
    )

    domain_path, problem_path, _ = fuse_domains(config, output_dir)
    print(json.dumps({"domain": str(domain_path), "problem": str(problem_path)}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
