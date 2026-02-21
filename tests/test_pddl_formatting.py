"""Tests for PDDL formatting utilities.

Tests the pretty-printing utilities in puxle.pddls.formatting without requiring
actual PDDL parsing or JAX dependencies where possible.
"""

from unittest.mock import Mock

from puxle.pddls.formatting import (
    action_to_string,
    build_label_color_maps,
    split_atom,
)


class TestSplitAtom:
    """Test atom string splitting into label and arguments."""

    def test_parenthesized_atom(self):
        """Test atom with parentheses like (on a b)."""
        label, args = split_atom("(on a b)")
        assert label == "on"
        assert args == ["a", "b"]

    def test_parenthesized_single_arg(self):
        """Test atom with single argument like (clear a)."""
        label, args = split_atom("(clear a)")
        assert label == "clear"
        assert args == ["a"]

    def test_parenthesized_no_args(self):
        """Test predicate with no arguments like (ready)."""
        label, args = split_atom("(ready)")
        assert label == "ready"
        assert args == []

    def test_no_parentheses(self):
        """Test atom without parentheses like 'clear a'."""
        label, args = split_atom("clear a")
        assert label == "clear"
        assert args == ["a"]

    def test_no_parentheses_multiple_args(self):
        """Test multiple arguments without parens."""
        label, args = split_atom("at robot location1")
        assert label == "at"
        assert args == ["robot", "location1"]

    def test_empty_string(self):
        """Test empty string input."""
        label, args = split_atom("")
        assert label == ""
        assert args == []

    def test_empty_parentheses(self):
        """Test empty parentheses."""
        label, args = split_atom("()")
        assert label == ""
        assert args == []

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        label, args = split_atom("   ")
        assert label == ""
        assert args == []

    def test_parenthesized_whitespace(self):
        """Test parentheses with whitespace."""
        label, args = split_atom("(   )")
        assert label == ""
        assert args == []

    def test_extra_whitespace(self):
        """Test atom with extra whitespace."""
        label, args = split_atom("(on   a   b)")
        assert label == "on"
        assert args == ["a", "b"]


class TestBuildLabelColorMaps:
    """Test color map generation for predicates and actions."""

    def test_with_actions_and_predicates(self):
        """Test domain with both actions and predicates."""
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

        # Should have entries for all labels
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

        # Default color should be white
        assert rich_map["default"] == "white"
        assert tc_map["default"] == "white"

    def test_actions_only(self):
        """Test domain with only actions."""
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
        """Test domain with only predicates."""
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
        """Test domain with no actions or predicates."""
        domain = Mock()
        domain.actions = []
        domain.predicates = []

        rich_map, tc_map = build_label_color_maps(domain)

        # Should only have default
        assert rich_map == {"default": "white"}
        assert tc_map == {"default": "white"}

    def test_none_attributes(self):
        """Test domain with None actions/predicates."""
        domain = Mock()
        domain.actions = None
        domain.predicates = None

        rich_map, tc_map = build_label_color_maps(domain)

        assert rich_map == {"default": "white"}
        assert tc_map == {"default": "white"}

    def test_missing_attributes(self):
        """Test domain object without actions/predicates attributes."""
        domain = Mock(spec=[])  # Empty spec, no attributes

        rich_map, tc_map = build_label_color_maps(domain)

        assert rich_map == {"default": "white"}
        assert tc_map == {"default": "white"}

    def test_duplicate_names(self):
        """Test domain with duplicate predicate/action names."""
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

        # Should only have one entry for "move"
        assert "move" in rich_map
        assert len([k for k in rich_map.keys() if k != "default"]) == 1

    def test_sorted_labels(self):
        """Test that labels are sorted consistently."""
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

        # Sorted order should be: alpha, middle, zebra
        labels = [k for k in rich_map.keys() if k != "default"]
        assert labels == sorted(labels)

    def test_action_without_name(self):
        """Test handling of action objects without name attribute."""
        domain = Mock()
        action_good = Mock()
        action_good.name = "move"
        action_bad = Mock(spec=[])  # No name attribute

        domain.actions = [action_good, action_bad]
        domain.predicates = []

        rich_map, tc_map = build_label_color_maps(domain)

        # Should only include the good action
        assert "move" in rich_map
        assert len([k for k in rich_map.keys() if k != "default"]) == 1


class TestActionToString:
    """Test action formatting to string."""

    def test_valid_action_colored(self):
        """Test formatting valid action with color."""
        actions = [{"name": "move", "parameters": ["a", "b"]}]
        color_map = {"move": "cyan"}
        result = action_to_string(actions, 0, color_map, colored=True)

        assert "move" in result
        assert "a" in result
        assert "b" in result

    def test_valid_action_uncolored(self):
        """Test formatting valid action without color."""
        actions = [{"name": "move", "parameters": ["a", "b"]}]
        color_map = {"move": "cyan"}
        result = action_to_string(actions, 0, color_map, colored=False)

        assert result == "(move a b)"

    def test_action_no_parameters(self):
        """Test action with empty parameters."""
        actions = [{"name": "noop", "parameters": []}]
        color_map = {"noop": "white"}
        result = action_to_string(actions, 0, color_map, colored=False)

        assert result == "(noop )"

    def test_action_single_parameter(self):
        """Test action with single parameter."""
        actions = [{"name": "pickup", "parameters": ["block1"]}]
        color_map = {"pickup": "green"}
        result = action_to_string(actions, 0, color_map, colored=False)

        assert result == "(pickup block1)"

    def test_action_multiple_parameters(self):
        """Test action with multiple parameters."""
        actions = [{"name": "stack", "parameters": ["a", "b", "c", "d"]}]
        color_map = {}
        result = action_to_string(actions, 0, color_map, colored=False)

        assert result == "(stack a b c d)"

    def test_out_of_range_positive(self):
        """Test index beyond array length."""
        actions = [{"name": "move", "parameters": ["a", "b"]}]
        result = action_to_string(actions, 5, {})

        assert result == "action_5"

    def test_out_of_range_negative(self):
        """Test negative index."""
        actions = [{"name": "move", "parameters": ["a", "b"]}]
        result = action_to_string(actions, -1, {})

        assert result == "action_-1"

    def test_empty_actions_list(self):
        """Test with empty actions list."""
        result = action_to_string([], 0, {})

        assert result == "action_0"

    def test_color_fallback(self):
        """Test color fallback when action not in map."""
        actions = [{"name": "unknown", "parameters": ["x"]}]
        color_map = {"other": "cyan"}
        result = action_to_string(actions, 0, color_map, colored=True)

        # Should use white as fallback color
        assert "unknown" in result
        assert "x" in result

    def test_multiple_actions_first(self):
        """Test accessing first of multiple actions."""
        actions = [
            {"name": "move", "parameters": ["a", "b"]},
            {"name": "pickup", "parameters": ["c"]},
            {"name": "drop", "parameters": ["d"]},
        ]
        result = action_to_string(actions, 0, {}, colored=False)

        assert result == "(move a b)"

    def test_multiple_actions_middle(self):
        """Test accessing middle of multiple actions."""
        actions = [
            {"name": "move", "parameters": ["a", "b"]},
            {"name": "pickup", "parameters": ["c"]},
            {"name": "drop", "parameters": ["d"]},
        ]
        result = action_to_string(actions, 1, {}, colored=False)

        assert result == "(pickup c)"

    def test_multiple_actions_last(self):
        """Test accessing last of multiple actions."""
        actions = [
            {"name": "move", "parameters": ["a", "b"]},
            {"name": "pickup", "parameters": ["c"]},
            {"name": "drop", "parameters": ["d"]},
        ]
        result = action_to_string(actions, 2, {}, colored=False)

        assert result == "(drop d)"
