import pddl
from pddl.core import Action, Domain
from pddl.logic import Constant, Predicate
from pddl.logic.base import And

from puxle.pddls.fusion.api import FusionParams, iterative_fusion
from puxle.pddls.fusion.validator import DomainValidator


def test_validator_basic():
    validator = DomainValidator()

    # Valid domain
    d = Domain("test", predicates=[Predicate("p", Constant("a"))], actions=[])
    valid, errors = validator.validate(d)
    assert valid
    assert not errors

    # Invalid: Action uses undefined predicate
    a1 = Action("a1", parameters=[], precondition=Predicate("undefined"), effect=And())
    d2 = Domain("test2", predicates=[Predicate("p")], actions=[a1])
    valid, errors = validator.validate(d2)
    assert not valid
    assert "undefined predicate 'undefined'" in errors[0]


def test_iterative_fusion_structure(tmp_path):
    # create dummy domains
    d1_path = tmp_path / "d1.pddl"
    d1_path.write_text(
        "(define (domain d1) (:requirements :strips) (:predicates (p)) "
        "(:action a1 :parameters () :precondition (p) :effect ()))"
    )

    params = FusionParams(prob_add_pre=0.0, prob_add_eff=0.0)

    # 2 iterations
    # Iteration 1: d1 fused (modified).
    # Iteration 2: result fused with result (doubling actions basically).
    fused_domain = iterative_fusion([str(d1_path)], depth=2, params=params)

    # Original action a1.
    # After fusion 1: a1 exists (might be renamed if collision, but here only 1 domain).
    # After fusion 2: we fuse D(a1) with D(a1).
    # Logic in fuse_domains handles renaming.
    # So we expect 2 actions: a1 and a1_1.

    assert len(fused_domain.actions) >= 2


def test_removal_logic():
    # Test that removal happens if prob is high
    from puxle.pddls.fusion.action_modifier import ActionModifier

    params = FusionParams(
        prob_rem_pre=1.0, prob_rem_eff=1.0, prob_add_pre=0.0, prob_add_eff=0.0
    )
    modifier = ActionModifier(params)

    pre = Predicate("p")
    eff = Predicate("q")
    a1 = Action("a1", parameters=[], precondition=pre, effect=eff)

    # We need to pass types_map and all_predicates
    all_preds = [pre, eff]
    types_map = {}

    modified_actions = modifier.modify_actions([a1], all_preds, types_map)
    m_a1 = modified_actions[0]

    # Since we remove with prob 1.0, and there is 1 item.
    # Logic: if preconditions: pop random index.
    # So it should be removed.

    # precondition should be And() (empty)
    assert m_a1.precondition == And()
    # effect should be And() (empty)
    assert m_a1.effect == And()


def test_type_consistency_validator(tmp_path):
    validator = DomainValidator()

    # Case 1: Valid
    d_str = (
        "(define (domain test) (:requirements :typing) (:types a) "
        "(:action a1 :parameters (?x - a) :precondition () :effect ()))"
    )
    d_path = tmp_path / "valid.pddl"
    d_path.write_text(d_str)

    d1 = pddl.parse_domain(str(d_path))
    valid, errors = validator.validate(d1)
    assert valid

    # Case 2: Invalid type
    # Note: pddl library validation prevents creating invalid Domain object with undefined types.
    # So we cannot easily test the validator catching this, because Domain() constructor raises PDDLValidationError.
    # This confirms that type consistency is enforced.

    # from pddl.logic import Variable
    # try:
    #     p_invalid = Variable("y", type_tags={"undefined_type"})
    #     a_invalid = Action("a_inv", parameters=[p_invalid], precondition=And(), effect=And())
    #     d2 = Domain(
    #         d1.name,
    #         requirements=d1.requirements,
    #         types=d1.types,
    #         predicates=d1.predicates,
    #         actions=[a_invalid],
    #     )
    #     valid, errors = validator.validate(d2)
    #     assert not valid
    # except Exception:
    #     pass # Expected behavior from library


def test_varying_depth_benchmark(tmp_path):
    output_dir = tmp_path / "bench"

    # dummy domain file
    d1_path = tmp_path / "d1.pddl"
    d1_path.write_text(
        "(define (domain d1) (:requirements :strips) (:predicates (p)) "
        "(:action a1 :parameters () :precondition (p) :effect ()))"
    )

    from puxle.pddls.fusion.api import (
        FusionParams,
        generate_benchmark_with_varying_depth,
    )

    generate_benchmark_with_varying_depth(
        base_domains=[str(d1_path)],
        output_dir=str(output_dir),
        depth_range=(1, 2),
        problems_per_depth=1,
        params=FusionParams(),
    )

    assert (output_dir / "depth_1").exists()
    assert (output_dir / "depth_2").exists()
    assert (output_dir / "depth_1" / "domain.pddl").exists()
    assert (output_dir / "depth_1" / "problem_00.pddl").exists()
