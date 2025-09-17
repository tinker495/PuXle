from pathlib import Path
from typing import Dict, Tuple

import pytest

from puxle.pddls.fusion import FusionConfig, fuse_domains
from puxle.pddls.pddl import PDDL


@pytest.mark.slow
def test_fuse_domains_generates_valid_pddl(tmp_path):
    base = Path(__file__).resolve().parents[1] / "pddl_data"
    config = FusionConfig(
        domain_a=str(base / "simple_move" / "domain.pddl"),
        problem_a=str(base / "simple_move" / "problem.pddl"),
        domain_b=str(base / "toggle" / "domain.pddl"),
        problem_b=str(base / "toggle" / "problem.pddl"),
        name="simple-toggle",
        prob_add_pre=0.2,
        prob_add_eff=0.2,
        prob_rem_pre=0.1,
        prob_rem_eff=0.1,
        prob_negate=0.3,
        rollout_depth=2,
        seed=123,
        validation_depth=4,
    )

    domain_path, problem_path, metadata = fuse_domains(config, tmp_path)

    assert domain_path.exists()
    assert problem_path.exists()
    meta_path = tmp_path / "simple-toggle-problem.json"
    assert meta_path.exists()

    env = PDDL(str(domain_path), str(problem_path))
    assert env.num_atoms > 0
    assert env.num_actions > 0

    # Ensure metadata contains goal atoms and init atoms used by generator
    assert metadata["goal_atoms"], "Goal atoms should not be empty"
    assert metadata["init_atoms"], "Initial atoms should not be empty"
    assert metadata["validation"] is not None
    assert metadata["validation"]["depth_limit"] == 4

    # Grounded actions expose negative precondition container (may be empty)
    assert all("preconditions_neg" in action for action in env.grounded_actions)


def test_ensure_reversible_adds_complementary_effects(tmp_path):
    base = Path(__file__).resolve().parents[1] / "pddl_data"
    config = FusionConfig(
        domain_a=str(base / "simple_move" / "domain.pddl"),
        problem_a=str(base / "simple_move" / "problem.pddl"),
        domain_b=str(base / "toggle" / "domain.pddl"),
        problem_b=str(base / "toggle" / "problem.pddl"),
        name="reversible",
        ensure_reversible=True,
        seed=42,
        validation_depth=None,
    )

    domain_path, problem_path, _ = fuse_domains(config, tmp_path)
    env = PDDL(str(domain_path), str(problem_path))

    inverse_actions = [action for action in env.grounded_actions if "-inverse" in action["name"]]
    assert inverse_actions, "Expected inverse actions to be generated"

    by_name_params: Dict[Tuple[str, Tuple[str, ...]], dict] = {}
    for action in env.grounded_actions:
        key = (action["name"], tuple(action["parameters"]))
        by_name_params[key] = action

    for inverse in inverse_actions:
        inv_name = inverse["name"]
        base_name = inv_name[: inv_name.index("-inverse")]
        key = (base_name, tuple(inverse["parameters"]))
        assert key in by_name_params, f"Missing base action for {inv_name}"
        base_action = by_name_params[key]
        base_add, base_del = base_action["effects"]
        inv_add, inv_del = inverse["effects"]
        assert set(base_add) == set(inv_del)
        assert set(base_del) == set(inv_add)
