"""Tests for PDDL formatting utilities.

Tests the pretty-printing utilities in puxle.puzzles.adapters.pddl.formatting without requiring
actual PDDL parsing or JAX dependencies where possible.
"""

from unittest.mock import Mock

from puxle.puzzles.adapters.pddl.formatting import (
    action_to_string,
    build_label_color_maps,
    split_atom,
)


class TestSplitAtom:
    def test_parenthesized_atom(self):
        label, args = split_atom("(on a b)")
        assert label == "on"
        assert args == ["a", "b"]

    def test_parenthesized_single_arg(self):
        label, args = split_atom("(clear a)")
        assert label == "clear"
        assert args == ["a"]

    def test_parenthesized_no_args(self):
        label, args = split_atom("(ready)")
        assert label == "ready"
        assert args == []

    def test_no_parentheses(self):
        label, args = split_atom("clear a")
        assert label == "clear"
        assert args == ["a"]

    def test_no_parentheses_multiple_args(self):
        label, args = split_atom("at robot location1")
        assert label == "at"
        assert args == ["robot", "location1"]

    def test_empty_string(self):
        label, args = split_atom("")
        assert label == ""
        assert args == []

    def test_empty_parentheses(self):
        label, args = split_atom("()")
        assert label == ""
        assert args == []

    def test_whitespace_only(self):
        label, args = split_atom("   ")
        assert label == ""
        assert args == []

    def test_parenthesized_whitespace(self):
        label, args = split_atom("(   )")
        assert label == ""
        assert args == []

    def test_extra_whitespace(self):
        label, args = split_atom("(on   a   b)")
        assert label == "on"
        assert args == ["a", "b"]


class TestBuildLabelColorMaps:
    def test_with_actions_and_predicates(self):
        domain = Mock()
        action1 = Mock()
        action1.name = "move"
        action2 = Mock()
        action2.name = "pickup"
        pred1 = Mock()
        pred1.name = "on"
        pred2 = Mock()
        pred2.name = "clear"

        domain.actions = [action1, action2]
        domain.predicates = [pred1, pred2]

        rich_map, tc_map = build_label_color_maps(domain)

        assert "move" in rich_map
        assert "pickup" in rich_map
        assert "on" in rich_map
        assert "clear" in rich_map
        assert "default" in rich_map

        assert "move" in tc_map
        assert "pickup" in tc_map
        assert "on" in tc_map
        assert "clear" in tc_map
        assert "default" in tc_map

        assert rich_map["default"] == "white"
        assert tc_map["default"] == "white"

    def test_actions_only(self):
        domain = Mock()
        action1 = Mock()
        action1.name = "move"
        action2 = Mock()
        action2.name = "pickup"

        domain.actions = [action1, action2]
        domain.predicates = []

        rich_map, tc_map = build_label_color_maps(domain)

        assert "move" in rich_map
        assert "pickup" in rich_map
        assert len([k for k in rich_map.keys() if k != "default"]) == 2

    def test_predicates_only(self):
        domain = Mock()
        pred1 = Mock()
        pred1.name = "on"
        pred2 = Mock()
        pred2.name = "clear"

        domain.actions = []
        domain.predicates = [pred1, pred2]

        rich_map, tc_map = build_label_color_maps(domain)

        assert "on" in rich_map
        assert "clear" in rich_map
        assert len([k for k in rich_map.keys() if k != "default"]) == 2

    def test_empty_domain(self):
        domain = Mock()
        domain.actions = []
        domain.predicates = []

        rich_map, tc_map = build_label_color_maps(domain)

        assert rich_map == {"default": "white"}
        assert tc_map == {"default": "white"}

    def test_none_attributes(self):
        domain = Mock()
        domain.actions = None
        domain.predicates = None

        rich_map, tc_map = build_label_color_maps(domain)

        assert rich_map == {"default": "white"}
        assert tc_map == {"default": "white"}

    def test_missing_attributes(self):
        domain = Mock(spec=[])

        rich_map, tc_map = build_label_color_maps(domain)

        assert rich_map == {"default": "white"}
        assert tc_map == {"default": "white"}

    def test_duplicate_names(self):
        domain = Mock()
        action1 = Mock()
        action1.name = "move"
        action2 = Mock()
        action2.name = "move"
        pred1 = Mock()
        pred1.name = "move"

        domain.actions = [action1, action2]
        domain.predicates = [pred1]

        rich_map, tc_map = build_label_color_maps(domain)

        assert "move" in rich_map
        assert len([k for k in rich_map.keys() if k != "default"]) == 1

    def test_sorted_labels(self):
        domain = Mock()
        action_z = Mock()
        action_z.name = "zebra"
        action_a = Mock()
        action_a.name = "alpha"
        action_m = Mock()
        action_m.name = "middle"

        domain.actions = [action_z, action_a, action_m]
        domain.predicates = []

        rich_map, _ = build_label_color_maps(domain)

        labels = [k for k in rich_map.keys() if k != "default"]
        assert labels == sorted(labels)

    def test_action_without_name(self):
        domain = Mock()
        action_good = Mock()
        action_good.name = "move"
        action_bad = Mock(spec=[])

        domain.actions = [action_good, action_bad]
        domain.predicates = []

        rich_map, tc_map = build_label_color_maps(domain)

        assert "move" in rich_map
        assert len([k for k in rich_map.keys() if k != "default"]) == 1


class TestActionToString:
    def test_valid_action_colored(self):
        actions = [{"name": "move", "parameters": ["a", "b"]}]
        color_map = {"move": "cyan"}
        result = action_to_string(actions, 0, color_map, colored=True)

        assert "move" in result
        assert "a" in result
        assert "b" in result

    def test_valid_action_uncolored(self):
        actions = [{"name": "move", "parameters": ["a", "b"]}]
        color_map = {"move": "cyan"}
        result = action_to_string(actions, 0, color_map, colored=False)

        assert result == "(move a b)"

    def test_action_no_parameters(self):
        actions = [{"name": "noop", "parameters": []}]
        color_map = {"noop": "white"}
        result = action_to_string(actions, 0, color_map, colored=False)

        assert result == "(noop )"

    def test_action_single_parameter(self):
        actions = [{"name": "pickup", "parameters": ["block1"]}]
        color_map = {"pickup": "green"}
        result = action_to_string(actions, 0, color_map, colored=False)

        assert result == "(pickup block1)"

    def test_action_multiple_parameters(self):
        actions = [{"name": "stack", "parameters": ["a", "b", "c", "d"]}]
        color_map = {}
        result = action_to_string(actions, 0, color_map, colored=False)

        assert result == "(stack a b c d)"

    def test_out_of_range_positive(self):
        actions = [{"name": "move", "parameters": ["a", "b"]}]
        result = action_to_string(actions, 5, {})

        assert result == "action_5"

    def test_out_of_range_negative(self):
        actions = [{"name": "move", "parameters": ["a", "b"]}]
        result = action_to_string(actions, -1, {})

        assert result == "action_-1"

    def test_empty_actions_list(self):
        result = action_to_string([], 0, {})

        assert result == "action_0"

    def test_color_fallback(self):
        actions = [{"name": "unknown", "parameters": ["x"]}]
        color_map = {"other": "cyan"}
        result = action_to_string(actions, 0, color_map, colored=True)

        assert "unknown" in result
        assert "x" in result

    def test_multiple_actions_first(self):
        actions = [
            {"name": "move", "parameters": ["a", "b"]},
            {"name": "pickup", "parameters": ["c"]},
            {"name": "drop", "parameters": ["d"]},
        ]
        result = action_to_string(actions, 0, {}, colored=False)

        assert result == "(move a b)"

    def test_multiple_actions_middle(self):
        actions = [
            {"name": "move", "parameters": ["a", "b"]},
            {"name": "pickup", "parameters": ["c"]},
            {"name": "drop", "parameters": ["d"]},
        ]
        result = action_to_string(actions, 1, {}, colored=False)

        assert result == "(pickup c)"

    def test_multiple_actions_last(self):
        actions = [
            {"name": "move", "parameters": ["a", "b"]},
            {"name": "pickup", "parameters": ["c"]},
            {"name": "drop", "parameters": ["d"]},
        ]
        result = action_to_string(actions, 2, {}, colored=False)

        assert result == "(drop d)"
