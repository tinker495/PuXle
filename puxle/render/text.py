"""Monospace layout for host-rendered puzzle diagrams."""

import re
import unicodedata
from collections.abc import Iterable, Sequence

_SGR = re.compile(r"\x1b\[[0-9;]*m")


def visible_width(text: str) -> int:
    plain = _SGR.sub("", text)
    if plain.isascii():
        return len(plain)
    return sum(
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        for char in plain
        if not unicodedata.combining(char)
    )


def text_frame(rows: Sequence[str], *, title: str = "") -> str:
    """Use the native grid's square border and one space of inner padding."""
    widths = [visible_width(row) for row in rows]
    inner = max(max(widths, default=0) + 2, visible_width(title) + 4 if title else 0)
    top = "━" * inner
    if title:
        margin = inner - visible_width(title) - 2
        top = "━" * (margin // 2) + f" {title} " + "━" * (margin - margin // 2)
    lines = [
        f"┃ {row}{' ' * (inner - width - 1)}┃"
        for row, width in zip(rows, widths, strict=True)
    ]
    return "\n".join([f"┏{top}┓", *lines, f"┗{'━' * inner}┛"])


def wrap_items(items: Iterable[str], *, width: int, gap: str) -> list[str]:
    """Wrap between complete legend keys or argument tuples."""
    lines: list[str] = []
    line = ""
    for item in items:
        joined = f"{line}{gap}{item}" if line else item
        if line and visible_width(joined) > width:
            lines.append(line)
            line = item
        else:
            line = joined
    if line:
        lines.append(line)
    return lines
