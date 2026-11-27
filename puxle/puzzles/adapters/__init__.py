"""External-ecosystem and learned-checkpoint puzzle adapters for PuXle."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "CayleyPuzzle",
    "CayleyPresets",
    "PDDL",
    "PDDLPresets",
    "WorldModelPuzzle",
    "WorldModelPresets",
]

_EXPORTS = {
    "CayleyPuzzle": (".cayley", "CayleyPuzzle"),
    "CayleyPresets": (".cayley", "CayleyPresets"),
    "PDDL": (".pddl", "PDDL"),
    "PDDLPresets": (".pddl", "PDDLPresets"),
    "WorldModelPuzzle": (".world_model", "WorldModelPuzzle"),
    "WorldModelPresets": (".world_model", "WorldModelPresets"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
