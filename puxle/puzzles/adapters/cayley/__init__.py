"""Cayley/Schreier graph adapter (cayleypy) for PuXle."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "CayleyPuzzle",
    "CayleyPresets",
]

_EXPORTS = {
    "CayleyPuzzle": (".cayley", "CayleyPuzzle"),
    "CayleyPresets": (".presets", "CayleyPresets"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
