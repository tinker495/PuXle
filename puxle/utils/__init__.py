"""Utility functions and constants for PuXle puzzles."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "IMG_SIZE",
    "coloring_str",
]

_EXPORTS = {
    "IMG_SIZE": (".annotate", "IMG_SIZE"),
    "coloring_str": (".util", "coloring_str"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
