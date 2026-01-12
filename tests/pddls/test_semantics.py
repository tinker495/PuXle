from puxle.pddls.pddl import PDDL
from tests.pddls.data_specs import DATA_SPECS


class TestPDDLSemantics:
    """Domain-specific semantic tests."""

    def test_toggle_domain_semantics(self):
        """Test specific semantics of the toggle domain."""
        spec = next(s for s in DATA_SPECS if s.name == "toggle")
        puzzle = PDDL(spec.domain, spec.problem)

        # Find flip-on and flip-off actions
        flip_on_action = None
        flip_off_action = None

        for action in puzzle.grounded_actions:
            if action["name"] == "flip-on":
                flip_on_action = action
            elif action["name"] == "flip-off":
                flip_off_action = action

        assert flip_on_action is not None
        assert flip_off_action is not None

        # Check flip-on preconditions and effects
        assert any("off" in p for p in flip_on_action["preconditions"])
        effects = flip_on_action["effects"]
        assert any("on" in e for e in effects["add"])
        assert any("off" in e for e in effects["delete"])

        # Check flip-off preconditions and effects
        assert any("on" in p for p in flip_off_action["preconditions"])
        effects = flip_off_action["effects"]
        assert any("off" in e for e in effects["add"])
        assert any("on" in e for e in effects["delete"])

    def test_door_move_domain_semantics(self):
        """Test specific semantics of the door-move domain."""
        spec = next(s for s in DATA_SPECS if s.name == "door-move")
        puzzle = PDDL(spec.domain, spec.problem)

        # Find open action for r->hall
        open_action = None
        move_action = None

        for action in puzzle.grounded_actions:
            if action["name"] == "open" and action["parameters"] == ["r", "hall"]:
                open_action = action
            elif action["name"] == "move" and action["parameters"] == ["r", "hall"]:
                move_action = action

        assert open_action is not None
        assert move_action is not None

        # Check open action preconditions
        preconditions = open_action["preconditions"]
        assert any("at r" in p for p in preconditions)
        assert any("connected r hall" in p for p in preconditions)
        assert any("has-key" in p for p in preconditions)

        # Check open action effects
        effects = open_action["effects"]
        assert any("open r hall" in e for e in effects["add"])
        assert len(effects["delete"]) == 0

        # Check move action preconditions (requires door to be open)
        preconditions = move_action["preconditions"]
        assert any("at r" in p for p in preconditions)
        assert any("connected r hall" in p for p in preconditions)
        assert any("open r hall" in p for p in preconditions)

    def test_typed_move_domain_semantics(self):
        """Test specific semantics of the typed-move domain."""
        spec = next(s for s in DATA_SPECS if s.name == "typed-move")
        puzzle = PDDL(spec.domain, spec.problem)

        # Check that type restrictions are respected
        move_rh_actions = [a for a in puzzle.grounded_actions if a["name"] == "move-rh"]
        move_hr_actions = [a for a in puzzle.grounded_actions if a["name"] == "move-hr"]

        # Should have 2 move-rh actions (r1->h1, r2->h1) and 2 move-hr actions (h1->r1, h1->r2)
        assert len(move_rh_actions) == 2
        assert len(move_hr_actions) == 2

        # Check that move-rh only has room->hall parameters
        for action in move_rh_actions:
            params = action["parameters"]
            assert len(params) == 2
            # Parameters should be room and hall (we can't easily check types here,
            # but we can check that the parameter combinations make sense)

        # Check that move-hr only has hall->room parameters
        for action in move_hr_actions:
            params = action["parameters"]
            assert len(params) == 2
            # Parameters should be hall and room
