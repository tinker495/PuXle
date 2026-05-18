"""Tests for the cayley_subclasses meta-module (8 tests).

All tests that require cayleypy are gated via pytest.importorskip("cayleypy").
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Test 1: backward compat — 5 existing hard-coded names still importable
# ---------------------------------------------------------------------------


def test_backward_compat_5_existing_names():
    cayleypy = pytest.importorskip("cayleypy")  # noqa: F841

    import puxle.puzzles.cayley_subclasses as m
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    names = [
        "CayleyPancake7",
        "CayleyPancake8",
        "CayleyLRX8",
        "CayleyTopSpin8K4",
        "CayleyCoxeter8",
    ]
    for name in names:
        cls = getattr(m, name)
        assert issubclass(cls, CayleyPuzzle), f"{name} is not a CayleyPuzzle subclass"
        assert cls.__name__ == name, f"{name}.__name__ mismatch: {cls.__name__!r}"

    # Also verify the top-level puxle import chain works.
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


# ---------------------------------------------------------------------------
# Test 2: new dynamic class construction
# ---------------------------------------------------------------------------


def test_new_dynamic_class_construction():
    pytest.importorskip("cayleypy")

    from puxle.puzzles.cayley_puzzle import CayleyPuzzle
    from puxle.puzzles.cayley_subclasses import CayleyPancake6

    assert issubclass(CayleyPancake6, CayleyPuzzle)
    puzzle = CayleyPancake6()
    assert puzzle.action_size > 0


# ---------------------------------------------------------------------------
# Test 3: positional args parsing (underscore form)
# ---------------------------------------------------------------------------


def test_positional_args_parsing():
    pytest.importorskip("cayleypy")

    import puxle.puzzles.cayley_subclasses as m
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    cls = getattr(m, "CayleyConsecutiveKCycles8_3")
    assert issubclass(cls, CayleyPuzzle)
    assert cls._cayleypy_factory == "consecutive_k_cycles"
    assert cls._cayleypy_args == (8, 3)
    assert cls._cayleypy_kwargs == {}

    puzzle = cls()
    assert puzzle.action_size > 0


# ---------------------------------------------------------------------------
# Test 4: backward-compat K<n> kwarg form
# ---------------------------------------------------------------------------


def test_kwarg_parsing_backcompat():
    pytest.importorskip("cayleypy")

    import puxle.puzzles.cayley_subclasses as m
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    cls = getattr(m, "CayleyTopSpin8K4")
    assert issubclass(cls, CayleyPuzzle)
    assert cls._cayleypy_factory == "top_spin"
    assert cls._cayleypy_args == (8,)
    # The kwarg name should match the factory's actual parameter ('k').
    assert cls._cayleypy_kwargs == {"k": 4}

    puzzle = cls()
    assert puzzle.action_size > 0


# ---------------------------------------------------------------------------
# Test 5: unknown factory raises AttributeError (no cayleypy gate)
# ---------------------------------------------------------------------------


def test_unknown_factory_raises_attributeerror():
    import puxle.puzzles.cayley_subclasses as m

    with pytest.raises(AttributeError):
        _ = m.CayleyDoesNotExist99


# ---------------------------------------------------------------------------
# Test 6: caching returns same class object
# ---------------------------------------------------------------------------


def test_caching_returns_same_class():
    pytest.importorskip("cayleypy")

    import puxle.puzzles.cayley_subclasses as m

    cls1 = m.CayleyPancake6
    cls2 = m.CayleyPancake6
    assert cls1 is cls2


# ---------------------------------------------------------------------------
# Test 7: discover() helper
# ---------------------------------------------------------------------------


def test_discover_helper():
    pytest.importorskip("cayleypy")

    from puxle.puzzles.cayley_puzzle import CayleyPuzzle
    from puxle.puzzles.cayley_subclasses import CayleyPancake5, discover

    cls = discover("pancake", 5)
    assert issubclass(cls, CayleyPuzzle)

    puzzle_discover = cls()
    puzzle_named = CayleyPancake5()
    # Both should have the same action_size (same underlying graph).
    assert puzzle_discover.action_size == puzzle_named.action_size

    # discover() is cached: same call returns same object.
    cls2 = discover("pancake", 5)
    assert cls is cls2


# ---------------------------------------------------------------------------
# Test 8: list_available_factories()
# ---------------------------------------------------------------------------


def test_list_available_factories():
    pytest.importorskip("cayleypy")

    from puxle.puzzles.cayley_subclasses import list_available_factories

    factories = list_available_factories()
    assert isinstance(factories, list)
    assert len(factories) > 0
    assert "pancake" in factories
    assert "lrx" in factories
