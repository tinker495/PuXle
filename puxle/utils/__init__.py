"""
Utility functions and constants for PuXle puzzles.

This module provides helper functions, annotations, and common utilities used across puzzle implementations.
"""

from puxle.utils.annotate import IMG_SIZE
from puxle.utils.util import (
    add_img_parser,
    coloring_str,
    from_uint8,
    to_uint8,
    pack_variable_bits,
    unpack_variable_bits,
)

__all__ = [
    "IMG_SIZE",
    "add_img_parser",
    "coloring_str",
    "from_uint8", 
    "to_uint8",
    "pack_variable_bits",
    "unpack_variable_bits",
]