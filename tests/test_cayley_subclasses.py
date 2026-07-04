"""Tests for explicit no-arg CayleyPuzzle registry subclasses."""

from __future__ import annotations

import pytest

from puxle.puzzles.cayley_puzzle import CayleyPuzzle
from puxle.puzzles.cayley_subclasses import (
    CayleyCoxeter8,
    CayleyLRX8,
    CayleyPancake7,
    CayleyPancake8,
    CayleyTopSpin8K4,
)

CLASSES = (
    (CayleyPancake7, "pancake", (7,), {}),
    (CayleyPancake8, "pancake", (8,), {}),
    (CayleyLRX8, "lrx", (8,), {}),
    (CayleyTopSpin8K4, "top_spin", (8,), {"k": 4}),
    (CayleyCoxeter8, "coxeter", (8,), {}),
)


@pytest.mark.parametrize(("cls", "factory", "args", "kwargs"), CLASSES)
def test_explicit_subclass_metadata(cls, factory, args, kwargs):
    assert issubclass(cls, CayleyPuzzle)
    assert cls._cayleypy_factory == factory
    assert cls._cayleypy_args == args
    assert cls._cayleypy_kwargs == kwargs


def test_top_level_imports_export_explicit_classes():
    from puxle import (
        CayleyCoxeter8,
        CayleyLRX8,
        CayleyPancake7,
        CayleyPancake8,
        CayleyTopSpin8K4,
    )

    for cls in (
        CayleyPancake7,
        CayleyPancake8,
        CayleyLRX8,
        CayleyTopSpin8K4,
        CayleyCoxeter8,
    ):
        assert issubclass(cls, CayleyPuzzle)


def test_dynamic_factories_are_not_exported():
    import puxle.puzzles.cayley_subclasses as module

    assert not hasattr(module, "discover")
    assert not hasattr(module, "list_available_factories")
    with pytest.raises(AttributeError):
        getattr(module, "CayleyPancake6")


@pytest.mark.parametrize(("cls", "_factory", "_args", "_kwargs"), CLASSES)
def test_explicit_subclasses_instantiate_when_cayleypy_is_installed(
    cls, _factory, _args, _kwargs
):
    pytest.importorskip("cayleypy")

    puzzle = cls()
    assert puzzle.action_size > 0
