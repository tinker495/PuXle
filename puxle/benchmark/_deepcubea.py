from __future__ import annotations

import pickle
import warnings
from typing import Any, IO

import numpy as np


class DeepCubeAUnpickler(pickle.Unpickler):
    """Unpickler that recreates missing DeepCubeA environment classes on the fly."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("environments."):
            # Recreate the placeholder class once per (module, name) pair and reuse it.
            cache_key = f"{module}.{name}"
            return globals().setdefault(cache_key, type(name, (), {}))
        return super().find_class(module, name)


def load_deepcubea(handle: IO[bytes]) -> Any:
    """Helper that loads a DeepCubeA pickle with the compatible unpickler."""
    try:
        from numpy.exceptions import VisibleDeprecationWarning
    except ImportError:
        from numpy import VisibleDeprecationWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
        return DeepCubeAUnpickler(handle).load()


__all__ = ["DeepCubeAUnpickler", "load_deepcubea"]

