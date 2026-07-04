"""No-arg CayleyPuzzle subclasses used by downstream registries."""

from __future__ import annotations

from puxle.puzzles.cayley_puzzle import CayleyPuzzle

__all__ = [
    "CayleyPancake7",
    "CayleyPancake8",
    "CayleyLRX8",
    "CayleyTopSpin8K4",
    "CayleyCoxeter8",
]


def _permutation_graph(factory_name: str, *args, **kwargs):
    from cayleypy.graphs_lib import PermutationGroups

    return getattr(PermutationGroups, factory_name)(*args, **kwargs)


class CayleyPancake7(CayleyPuzzle):
    _cayleypy_factory = "pancake"
    _cayleypy_args = (7,)
    _cayleypy_kwargs = {}

    def __init__(self, **kwargs):
        super().__init__(_permutation_graph("pancake", 7), **kwargs)


class CayleyPancake8(CayleyPuzzle):
    _cayleypy_factory = "pancake"
    _cayleypy_args = (8,)
    _cayleypy_kwargs = {}

    def __init__(self, **kwargs):
        super().__init__(_permutation_graph("pancake", 8), **kwargs)


class CayleyLRX8(CayleyPuzzle):
    _cayleypy_factory = "lrx"
    _cayleypy_args = (8,)
    _cayleypy_kwargs = {}

    def __init__(self, **kwargs):
        super().__init__(_permutation_graph("lrx", 8), **kwargs)


class CayleyTopSpin8K4(CayleyPuzzle):
    _cayleypy_factory = "top_spin"
    _cayleypy_args = (8,)
    _cayleypy_kwargs = {"k": 4}

    def __init__(self, **kwargs):
        super().__init__(_permutation_graph("top_spin", 8, k=4), **kwargs)


class CayleyCoxeter8(CayleyPuzzle):
    _cayleypy_factory = "coxeter"
    _cayleypy_args = (8,)
    _cayleypy_kwargs = {}

    def __init__(self, **kwargs):
        super().__init__(_permutation_graph("coxeter", 8), **kwargs)
