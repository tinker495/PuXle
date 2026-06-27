"""Puzzle Renderer Module."""

from __future__ import annotations

from puxle._lazy_imports import lazy_dir, load_lazy_export

__all__ = ["attach_state_renderer", "Cv2Backend"]

_EXPORTS = {
    "attach_state_renderer": (".state_renderer", "attach_state_renderer"),
    "Cv2Backend": (".backends.cv2_backend", "Cv2Backend"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
