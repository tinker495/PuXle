import pytest
from pddl.core import Action
from pddl.logic import Predicate, Variable

from puxle import PDDL
from puxle.pddls.fusion.action_modifier import ActionModifier, FusionParams
from puxle.pddls.fusion.api import fuse_and_load
from puxle.pddls.fusion.domain_fusion import DomainFusion

# Mock data
DOM1_TEXT = """(define (domain d1)
  (:requirements :strips :typing)
  (:types a b)
  (:predicates (p1 ?x - a) (p2 ?y - b))
  (:action act1
    :parameters (?x - a)
    :precondition (p1 ?x)
    :effect (not (p1 ?x)))
)"""

DOM2_TEXT = """(define (domain d2)
  (:requirements :strips :typing)
  (:types b c)
  (:predicates (p2 ?y - b) (p3 ?z - c))
  (:action act2
    :parameters (?z - c)
    :precondition (p3 ?z)
    :effect (not (p3 ?z)))
)"""


@pytest.fixture
def domain_files(tmp_path):
    d1 = tmp_path / "d1.pddl"
    d1.write_text(DOM1_TEXT)
    d2 = tmp_path / "d2.pddl"
    d2.write_text(DOM2_TEXT)
    return str(d1), str(d2)


def test_domain_fusion_merge(domain_files):
    # Use the fixture to get paths
    d1_path, d2_path = domain_files

    import pddl

    d1 = pddl.parse_domain(d1_path)
    d2 = pddl.parse_domain(d2_path)

    fusion = DomainFusion()
    fused = fusion.fuse_domains([d1, d2], name="fused")

    assert fused.name == "fused"
    # Requirements union
    # set comparison
    reqs = {str(r) for r in fused.requirements}
    assert ":strips" in reqs
    assert ":typing" in reqs

    # Types union (a, b, c) -> b is shared
    type_names = {str(t) for t in fused.types}
    assert type_names == {"a", "b", "c"}

    # Predicates union (disjoint)
    pred_names = {p.name for p in fused.predicates}
    assert pred_names == {"dom0_p1", "dom0_p2", "dom1_p2", "dom1_p3"}

    # Actions union (disjoint)
    act_names = {a.name for a in fused.actions}
    assert act_names == {"dom0_act1", "dom1_act2"}


def test_action_modifier():
    # Setup simple data
    # Use a new predicate "p_new" not in action to ensure modification is visible
    preds = [Predicate("p_new")]
    act = Action("act", parameters=[], precondition=Predicate("p1"), effect=Predicate("p1"))

    # Force addition of precondition
    params = FusionParams(prob_add_pre=1.0, prob_add_eff=0.0, prob_neg=0.0, seed=42)
    mod = ActionModifier(params)

    # We force addition.
    modified = mod.modify_actions([act], preds, {})
    assert len(modified) == 1
    m_act = modified[0]

    assert m_act != act
    # Check that p_new is in precondition
    # String representation might help debugging if direct object check fails
    assert "p_new" in str(m_act.precondition)


def test_action_modifier_type_compatibility():
    vehicle_var = Variable("v", type_tags={"vehicle"})
    truck_var = Variable("t", type_tags={"truck"})

    requires_truck = Predicate("requires-truck", Variable("x", type_tags={"truck"}))
    requires_vehicle = Predicate("requires-vehicle", Variable("x", type_tags={"vehicle"}))

    params = FusionParams(
        prob_add_pre=1.0,
        prob_add_eff=0.0,
        prob_rem_pre=0.0,
        prob_rem_eff=0.0,
        prob_neg=0.0,
        seed=42,
    )
    modifier = ActionModifier(params)
    types_map = {"vehicle": "object", "truck": "vehicle"}

    # Supertype variable should NOT be used where a subtype is required.
    act_vehicle = Action("act-veh", parameters=[vehicle_var], precondition=Predicate("p1"), effect=Predicate("p1"))
    m_vehicle = modifier.modify_actions([act_vehicle], [requires_truck], types_map)[0]
    assert "requires-truck" not in str(m_vehicle.precondition)

    # Subtype variable can be used where a supertype is required.
    act_truck = Action("act-truck", parameters=[truck_var], precondition=Predicate("p1"), effect=Predicate("p1"))
    m_truck = modifier.modify_actions([act_truck], [requires_vehicle], types_map)[0]
    assert "requires-vehicle" in str(m_truck.precondition)


def test_fuse_and_load_api(domain_files, tmp_path):
    d1, d2 = domain_files

    # Use API to fuse
    env = fuse_and_load([d1, d2], name="test-fused")

    assert isinstance(env, PDDL)
    assert env.domain.name == "test-fused"
    # Validates that PuXle initialized correctly with in-memory objects
    assert env.num_actions >= 2  # act1 + act2
    assert env.num_atoms > 0


def test_n_domain_fusion(tmp_path):
    # Create 3 simple domain files
    d1_path = tmp_path / "d1_n.pddl"
    d1_path.write_text(
        "(define (domain d1) (:requirements :strips) (:action a1 :parameters () :precondition () :effect ()))"
    )

    d2_path = tmp_path / "d2_n.pddl"
    d2_path.write_text(
        "(define (domain d2) (:requirements :strips) (:action a2 :parameters () :precondition () :effect ()))"
    )

    d3_path = tmp_path / "d3_n.pddl"
    d3_path.write_text(
        "(define (domain d3) (:requirements :strips) (:action a3 :parameters () :precondition () :effect ()))"
    )

    import pddl

    d1 = pddl.parse_domain(str(d1_path))
    d2 = pddl.parse_domain(str(d2_path))
    d3 = pddl.parse_domain(str(d3_path))

    fusion = DomainFusion()
    fused = fusion.fuse_domains([d1, d2, d3], name="tri-fusion")

    assert len(fused.actions) == 3
    names = {a.name for a in fused.actions}
    assert names == {"dom0_a1", "dom1_a2", "dom2_a3"}


def test_domain_fusion_predicate_signature_collision(tmp_path):
    d1_path = tmp_path / "d1_sig.pddl"
    d1_path.write_text(
        "(define (domain d1_sig) (:requirements :strips :typing) (:types a) (:predicates (shared ?x - a)))"
    )

    d2_path = tmp_path / "d2_sig.pddl"
    d2_path.write_text(
        "(define (domain d2_sig) (:requirements :strips :typing) (:types b) (:predicates (shared ?x - b)))"
    )

    import pddl

    d1 = pddl.parse_domain(str(d1_path))
    d2 = pddl.parse_domain(str(d2_path))

    fusion = DomainFusion()
    fused = fusion.fuse_domains([d1, d2], name="good-fused")

    pred_names = {p.name for p in fused.predicates}
    assert "dom0_shared" in pred_names
    assert "dom1_shared" in pred_names
