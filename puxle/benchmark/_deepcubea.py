from __future__ import annotations

import pickle
import warnings
from typing import Any, IO



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


from importlib.resources import files
from pathlib import Path

def load_deepcubea_dataset(
    dataset_path: Path | None,
    dataset_name: str,
    package_resource: str,
    fallback_dir: Path,
) -> dict[str, Any]:
    """Helper to load a DeepCubeA dataset from various possible locations."""
    if dataset_path is not None:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"DeepCubeA dataset not found at {dataset_path}")
        with dataset_path.open("rb") as handle:
            return load_deepcubea(handle)

    try:
        resource = files(package_resource) / dataset_name
        with resource.open("rb") as handle:
            return load_deepcubea(handle)
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    fallback = fallback_dir / dataset_name
    if not fallback.is_file():
        raise FileNotFoundError(
            f"Unable to locate {dataset_name} under package resources or at {fallback}"
        )
    with fallback.open("rb") as handle:
        return load_deepcubea(handle)

__all__ = ["DeepCubeAUnpickler", "load_deepcubea", "load_deepcubea_dataset"]

