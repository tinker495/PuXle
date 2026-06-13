from pddl.core import Action
from pddl.logic import Constant, Predicate, Variable
from pddl.logic.base import And, Not

import puxle.pddls.fusion.problem_generator as problem_generator_module
from puxle.pddls.fusion.action_modifier import ActionModifier, FusionParams
from puxle.pddls.fusion.formula_facts import (
    extract_predicates,
    flatten_formula,
    ground_formula,
)
from puxle.pddls.fusion.problem_generator import ProblemGenerator
from puxle.pddls.fusion.type_facts import (
    any_match_compatible_candidates,
    any_match_compatible_term,
    build_type_ancestor_map,
    normalise_type_tags,
    strict_compatible_terms,
)


class TypedFake:
    def __init__(self, name, type_tags):
        self.name = name
        self.type_tags = type_tags


def test_formula_walk_extracts_and_flattens_nested_nodes():
    p = Predicate("p")
    q = Predicate("q")
    r = Predicate("r")
    formula = And(p, And(Not(q), r))

    assert flatten_formula(formula) == [p, Not(q), r]
    assert [predicate.name for predicate in extract_predicates(formula)] == [
        "p",
        "q",
        "r",
    ]


def test_ground_formula_substitutes_terms_under_not_and_and():
    x = Variable("x")
    y = Variable("y")
    a = Constant("a")
    b = Constant("b")

    grounded = ground_formula(
        And(Predicate("at", x), Not(Predicate("clear", y))),
        {"x": a, "y": b},
    )

    assert grounded == And(Predicate("at", a), Not(Predicate("clear", b)))


def test_type_tags_are_normalised_and_compared_with_ancestry():
    truck = Variable("truck-param", type_tags={"truck"})
    vehicle = Variable("vehicle-param", type_tags={"vehicle"})
    required_truck = Variable("required-truck", type_tags={"truck"})
    required_vehicle = Variable("required-vehicle", type_tags={"vehicle"})
    untyped = Variable("anything")
    ancestors = build_type_ancestor_map({"vehicle": "object", "truck": "vehicle"})

    assert normalise_type_tags(untyped) == {"object"}
    assert strict_compatible_terms(truck, required_vehicle, ancestors)
    assert not strict_compatible_terms(vehicle, required_truck, ancestors)


def test_any_match_compatible_terms_preserve_generator_multi_tag_matching():
    multi_tag_candidate = TypedFake("obj", {"truck", "vehicle"})
    required_vehicle = Variable("required-vehicle", type_tags={"vehicle"})

    assert not strict_compatible_terms(multi_tag_candidate, required_vehicle)
    assert any_match_compatible_term(multi_tag_candidate, required_vehicle)
    assert any_match_compatible_candidates(
        [multi_tag_candidate],
        required_vehicle,
    ) == [multi_tag_candidate]


def test_problem_generator_sampling_uses_any_match_type_policy():
    generator = ProblemGenerator()
    multi_tag_candidate = TypedFake("obj", {"truck", "vehicle"})
    required_vehicle = Variable("x", type_tags={"vehicle"})
    action = Action(
        "move",
        parameters=[required_vehicle],
        precondition=Predicate("ready"),
        effect=Predicate("ready"),
    )
    predicate = Predicate("at", required_vehicle)

    assert generator._sample_args(action, [multi_tag_candidate]) == [
        multi_tag_candidate
    ]
    assert generator._sample_args_for_predicate(
        predicate,
        [multi_tag_candidate],
    ) == [multi_tag_candidate]


def test_problem_generator_sampling_uses_type_ancestry_for_subtypes():
    generator = ProblemGenerator()
    truck_obj = TypedFake("truck1", {"truck"})
    required_vehicle = Variable("x", type_tags={"vehicle"})
    type_ancestors = build_type_ancestor_map({"truck": "vehicle", "vehicle": "object"})
    action = Action(
        "drive",
        parameters=[required_vehicle],
        precondition=Predicate("ready"),
        effect=Predicate("ready"),
    )
    predicate = Predicate("parked", required_vehicle)

    assert generator._sample_args(action, [truck_obj], type_ancestors) == [truck_obj]
    assert generator._sample_args_for_predicate(
        predicate,
        [truck_obj],
        type_ancestors,
    ) == [truck_obj]


def test_formula_facts_cover_shared_formula_operations():
    p = Predicate("p")
    q = Predicate("q")
    formula = And(p, And(Not(q)))

    assert flatten_formula(formula) == [p, Not(q)]
    assert [predicate.name for predicate in extract_predicates(formula)] == [
        "p",
        "q",
    ]

    x = Variable("x")
    obj = Constant("obj")
    assert ground_formula(Predicate("bound", x), {"x": obj}) == Predicate("bound", obj)


def test_problem_generator_formula_call_paths_use_shared_grounding(monkeypatch):
    calls = []
    sentinel_state = {Predicate("sentinel")}
    x = Variable("x")
    obj = Constant("obj")
    generator = ProblemGenerator()

    def fake_ground(formula, var_map):
        calls.append(("ground", formula, var_map))
        return Predicate("ready", obj)

    monkeypatch.setattr(problem_generator_module, "ground_formula", fake_ground)

    assert generator._check_preconditions(
        Predicate("ready", x),
        {"x": obj},
        {Predicate("ready", obj)},
    )
    assert generator._apply_effects(
        Predicate("ready", x),
        {"x": obj},
        sentinel_state,
    ) == {Predicate("sentinel"), Predicate("ready", obj)}
    assert [call[0] for call in calls] == ["ground", "ground"]


def test_action_modifier_keeps_subtype_binding_rules_through_shared_interface():
    params = FusionParams(
        prob_add_pre=1.0,
        prob_add_eff=0.0,
        prob_rem_pre=0.0,
        prob_rem_eff=0.0,
        prob_neg=0.0,
        seed=42,
    )
    modifier = ActionModifier(params)
    requires_vehicle = Predicate(
        "requires-vehicle",
        Variable("x", type_tags={"vehicle"}),
    )
    action = Action(
        "move-truck",
        parameters=[Variable("t", type_tags={"truck"})],
        precondition=Predicate("ready"),
        effect=Predicate("ready"),
    )

    modified = modifier.modify_actions(
        [action],
        [requires_vehicle],
        {"vehicle": "object", "truck": "vehicle"},
    )[0]

    assert "requires-vehicle" in str(modified.precondition)
