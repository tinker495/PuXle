"""Finite scalar quantization and its compact puzzle-state representation."""

from dataclasses import dataclass
from math import atanh

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FSQ:
    """Quantize each scalar to a uniform grid in [0, 1] with STE rounding.

    The shifted tanh bound supports both odd and even level counts. Integer
    codes are used only for storage; networks consume normalized grid values.
    """

    levels: int = 8

    def __post_init__(self):
        if type(self.levels) is not int or not 2 <= self.levels <= 256:
            raise ValueError("FSQ levels must be an integer in [2, 256].")

    @property
    def packed_bits(self) -> int:
        return (self.levels - 1).bit_length()

    def bound(self, inputs):
        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        half_width = (self.levels - 1) * 1.001 / 2
        offset = 0.5 if self.levels % 2 == 0 else 0.0
        shift = atanh(offset / half_width)
        bounded = jnp.tanh(inputs + shift) * half_width - offset
        return jnp.clip((bounded + self.levels // 2) / (self.levels - 1), 0.0, 1.0)

    def __call__(self, inputs):
        values = self.bound(inputs)
        rounded = jnp.round(values * (self.levels - 1)) / (self.levels - 1)
        return values + jax.lax.stop_gradient(rounded - values)

    def to_codes(self, values):
        return jnp.clip(
            jnp.round(values * (self.levels - 1)), 0, self.levels - 1
        ).astype(jnp.uint8)

    def from_codes(self, codes):
        return jnp.asarray(codes, dtype=jnp.float32) / (self.levels - 1)
