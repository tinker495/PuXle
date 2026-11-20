"""External-ecosystem puzzle adapters for PuXle."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "CayleyPuzzle",
    "CayleyPancake7",
    "CayleyPancake8",
    "CayleyLRX8",
    "CayleyTopSpin8K4",
    "CayleyCoxeter8",
    "PDDL",
]

_EXPORTS = {
    "CayleyPuzzle": (".cayley", "CayleyPuzzle"),
    "CayleyPancake7": (".cayley_subclasses", "CayleyPancake7"),
    "CayleyPancake8": (".cayley_subclasses", "CayleyPancake8"),
    "CayleyLRX8": (".cayley_subclasses", "CayleyLRX8"),
    "CayleyTopSpin8K4": (".cayley_subclasses", "CayleyTopSpin8K4"),
    "CayleyCoxeter8": (".cayley_subclasses", "CayleyCoxeter8"),
    "PDDL": (".pddl", "PDDL"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
