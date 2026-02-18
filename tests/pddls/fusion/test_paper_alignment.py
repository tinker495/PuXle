import pytest
from pddl.core import Domain, Action
from pddl.logic import Predicate, Constant
from pddl.logic.base import Not, And
from puxle.pddls.fusion.domain_fusion import DomainFusion
from puxle.pddls.fusion.action_modifier import ActionModifier, FusionParams

def test_disjoint_fusion():
    # Create two domains with overlapping predicate names
    d1 = Domain(
        name="d1",
        predicates=[Predicate("on", Constant("x"), Constant("y"))],
        actions=[Action("pick", parameters=[], precondition=And(), effect=And())]
    )
    d2 = Domain(
        name="d2",
        predicates=[Predicate("on", Constant("a"), Constant("b"))],
        actions=[Action("pick", parameters=[], precondition=And(), effect=And())]
    )
    
    fusion = DomainFusion()
    fused = fusion.fuse_domains([d1, d2], name="fused")
    
    # Check that we have disjoint predicates/actions
    # The exact renaming strategy depends on implementation, but they should be distinct
    # and not merged into one "on" predicate if we enforce disjointness.
    
    pred_names = {p.name for p in fused.predicates}
    action_names = {a.name for a in fused.actions}
    
    # We expect 2 predicates (renamed) and 2 actions (renamed)
    assert len(pred_names) == 2
    assert len(action_names) == 2
    
    # Ensure original "on" is not present (it should be renamed)
    assert "on" not in pred_names
    assert "pick" not in action_names

def test_reversibility():
    # Setup a scenario where an action deletes P but no action adds P
    p = Predicate("P")
    
    # Action deletes P
    a1 = Action(
        name="delete_P",
        parameters=[],
        precondition=None,
        effect=And(Not(p))
    )
    
    # Action a2 is neutral
    a2 = Action(
        name="neutral",
        parameters=[],
        precondition=And(),
        effect=And()
    )
    
    # No action adds P initially
    actions = [a1, a2]
    all_preds = [p]
    types_map = {}
    
    params = FusionParams(
        prob_add_pre=0.0,
        prob_add_eff=0.0,
        prob_rem_pre=0.0,
        prob_rem_eff=0.0,
        prob_neg=0.0,
        rev_flag=True, # Enable reversibility Check
        seed=42
    )
    
    modifier = ActionModifier(params)
    modified_actions = modifier.modify_actions(actions, all_preds, types_map)
    
    # Now we expect that P is added back by some action
    # Since we only have one action, it must be added to a1 or a new action?
    # The prompt says "inject P into effects of a random compatible action".
    
    has_p_adder = False
    for a in modified_actions:
        effs = modifier._extract_atomic_conditions(a.effect)
        for eff in effs:
            if isinstance(eff, Predicate) and eff.name == "P":
                has_p_adder = True
                break
    
    assert has_p_adder, "Reversibility check failed: No action adds P back."
