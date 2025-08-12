import pytest

from puxle.pddls.pddl import PDDL


class TestPDDLPresets:
    @pytest.mark.parametrize(
        "domain, problem_basename",
        [
            ("blocksworld", "bw-S-01"),
            ("gripper", "gr-S-01"),
            ("logistics", "lg-S-01"),
            ("rovers", "rv-S-01"),
            ("satellite", "st-S-01"),
            # Harder instances
            ("blocksworld", "bw-S-04"),
            ("gripper", "gr-S-04"),
            ("logistics", "lg-S-04"),
            ("rovers", "rv-S-04"),
            ("satellite", "st-S-04"),
        ],
    )
    def test_from_preset_loads_and_runs(self, domain, problem_basename):
        env = PDDL.from_preset(domain=domain, problem_basename=problem_basename)
        # Basic integrity checks
        assert env.num_atoms > 0
        assert env.num_actions > 0

        sc = env.get_solve_config()
        st = env.get_initial_state(sc)
        ns, costs = env.get_neighbours(sc, st, filled=True)

        # At least one action should be considered (may be inapplicable depending on instance)
        assert ns is not None and costs is not None
        assert costs.shape[0] == env.action_size
