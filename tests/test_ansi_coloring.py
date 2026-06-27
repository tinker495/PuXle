import importlib.abc
import sys


class _BlockTermcolor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] == "termcolor":
            raise ImportError(f"blocked termcolor import: {fullname}")
        return None


def test_named_ansi_coloring_without_termcolor(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "termcolor" or module_name.startswith("termcolor."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        if module_name in {"puxle.puzzles.lightsout", "puxle.pddls.formatting"}:
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    sys.meta_path.insert(0, _BlockTermcolor())
    try:
        from puxle.pddls.formatting import action_to_string
        from puxle.puzzles.lightsout import action_to_char
        from puxle.utils.util import colored_str

        action = action_to_string(
            [{"name": "move", "parameters": ["a", "b"]}],
            0,
            {"move": "red"},
            colored=True,
        )
    finally:
        sys.meta_path = [
            finder
            for finder in sys.meta_path
            if not isinstance(finder, _BlockTermcolor)
        ]

    assert colored_str("x", "red") == "\x1b[31mx\x1b[0m"
    assert action_to_char(0).startswith("\x1b[93m")
    assert action.startswith("\x1b[31mmove\x1b[0m")
    assert action.endswith(" a b")
