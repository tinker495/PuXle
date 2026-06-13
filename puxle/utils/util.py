"""PuXle utility helpers.

For in-memory bitpacked puzzle states, use xtructure's
``FieldDescriptor.packed_tensor(...)``/``set_unpacked(...)`` and aggregate
bitpacking via ``bits=...``. The canonical low-level codec is
``xtructure.core.layout.bitpack.to_uint8`` / ``from_uint8`` — PuXle no longer
carries duplicate primitives.
"""

from typing import Callable, Type, TypeVar

T = TypeVar("T")


def add_img_parser(cls: Type[T], imgfunc: Callable) -> Type[T]:
    """Deprecated alias for :func:`puxle.render.attach_state_renderer`.

    Kept so external puzzle authors that depend on the historical
    ``from puxle.utils import add_img_parser`` import path still work. New
    code (including ``puxle.core.puzzle_base``) consumes
    ``puxle.render.attach_state_renderer`` directly. See CONTEXT.md
    "Puzzle Renderer".
    """
    from puxle.render import attach_state_renderer

    return attach_state_renderer(cls, imgfunc)


def coloring_str(string: str, color: tuple[int, int, int]) -> str:
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m{string}\x1b[0m"
