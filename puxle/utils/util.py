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


ANSI_COLOR_CODES = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "light_grey": "37",
    "dark_grey": "90",
    "light_red": "91",
    "light_green": "92",
    "light_yellow": "93",
    "light_blue": "94",
    "light_magenta": "95",
    "light_cyan": "96",
    "bright_white": "97",
}


def colored_str(string: str, color: str | None) -> str:
    code = ANSI_COLOR_CODES.get(color or "")
    if code is None:
        return string
    return f"\x1b[{code}m{string}\x1b[0m"
