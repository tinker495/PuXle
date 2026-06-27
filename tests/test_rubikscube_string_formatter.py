import importlib.abc
import sys

import jax


class _BlockTabulate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] == "tabulate":
            raise ImportError(f"blocked tabulate import: {fullname}")
        return None


def test_rubikscube_string_formatter_uses_stdlib_padding(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "tabulate" or module_name.startswith("tabulate."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        if module_name == "puxle.puzzles.rubikscube":
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    sys.meta_path.insert(0, _BlockTabulate())
    try:
        from puxle.puzzles.rubikscube import RubiksCube

        cube = RubiksCube(
            size=2, initial_shuffle=0, color_embedding=True, metric="UQTM"
        )
        solve_config, state = cube.get_inits(jax.random.PRNGKey(0))
        rendered = state.str(solve_config=solve_config)
    finally:
        sys.meta_path = [
            finder for finder in sys.meta_path if not isinstance(finder, _BlockTabulate)
        ]

    assert "up" in rendered
    assert "front" in rendered
    assert "down" in rendered
    assert "┏" in rendered
    assert "┗" in rendered
