"""PDDL planning domain adapter for PuXle."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__all__ = ["PDDL", "PDDLPresets"]

_EXPORTS = {
    "PDDL": (".pddl", "PDDL"),
    "PDDLPresets": (".presets", "PDDLPresets"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
