"""Meta-module: auto-generates CayleyPuzzle subclasses for any cayleypy factory.

Usage via attribute access (module-level __getattr__):
    from puxle.puzzles.cayley_subclasses import CayleyPancake7
    from puxle.puzzles.cayley_subclasses import CayleyConsecutiveKCycles8_3

Name parsing rules:
    Cayley<FactoryPascalCase>[<NumericArg1>[K<KwargValue>|_<NumericArg2>...]]

    CayleyPancake7              -> pancake(7)
    CayleyPancake8              -> pancake(8)
    CayleyLRX8                  -> lrx(8)
    CayleyTopSpin8K4            -> top_spin(8, k=4)   [backward compat K<n> form]
    CayleyTopSpin8_4            -> top_spin(8, 4)      [positional _ form]
    CayleyCoxeter8              -> coxeter(8)
    CayleyAllCycles6            -> all_cycles(6)
    CayleyConsecutiveKCycles8_3 -> consecutive_k_cycles(8, 3)

Public helpers:
    discover(factory_name, *args, **kwargs) -> type[CayleyPuzzle]
    list_available_factories() -> list[str]

No top-level cayleypy import — imports lazily inside __getattr__ and inside
generated __init__ methods. Module loads cleanly without cayleypy installed.
"""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING

from puxle.puzzles.cayley_puzzle import CayleyPuzzle

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Cache: maps class name -> generated class object
# ---------------------------------------------------------------------------
_CLASS_CACHE: dict[str, type] = {}

# ---------------------------------------------------------------------------
# PascalCase -> snake_case converter
# ---------------------------------------------------------------------------
# Strategy: split on transitions from lowercase/digit to uppercase, or from a
# run of uppercase letters followed by an uppercase+lowercase (e.g. "LRXFoo"
# -> "LRX_Foo"). All-caps runs (like LRX, PDDL, LX) stay as a single token.
_RE_PASCAL_SPLIT = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"  # transition: lower/digit -> upper
    r"|(?<=[A-Z])(?=[A-Z][a-z])"  # transition: upper -> upper+lower (end of acronym)
)


def _pascal_to_snake(s: str) -> str:
    """Convert PascalCase (including all-caps acronyms) to snake_case.

    Examples:
        Pancake      -> pancake
        LRX          -> lrx
        TopSpin      -> top_spin
        AllCycles    -> all_cycles
        ConsecutiveKCycles -> consecutive_k_cycles
        CoxeterGroup -> coxeter_group
    """
    tokens = _RE_PASCAL_SPLIT.split(s)
    return "_".join(t.lower() for t in tokens)


# ---------------------------------------------------------------------------
# Name parser
# ---------------------------------------------------------------------------
# Pattern: Cayley<Pascal>[<digits>[K<digits>|(_<digits>)*]]
#
# The factory portion is everything from after "Cayley" up to the first digit
# (or end of string). The trailing numeric segments are the args.
#
# Backward-compat K<n> form:  CayleyTopSpin8K4 -> factory=top_spin, args=(8,), kwargs={k:4}
# Positional _ form:          CayleyTopSpin8_4  -> factory=top_spin, args=(8,4), kwargs={}
# Simple form:                CayleyPancake7    -> factory=pancake,   args=(7,), kwargs={}

_RE_NAME = re.compile(
    r"^Cayley"
    r"(?P<pascal>[A-Z][A-Za-z]*)"  # factory PascalCase name (must start uppercase)
    r"(?P<nums>[0-9].*)?$"  # optional numeric suffix
)

# Within the numeric suffix, detect K<digit> backward-compat kwarg form
_RE_K_KWARG = re.compile(r"^(?P<first>\d+)K(?P<kval>\d+)$")
# Underscore-separated positional args: "8_3" -> (8, 3)
_RE_POSITIONAL = re.compile(r"^\d+(?:_\d+)*$")


def _parse_name(name: str) -> tuple[str, tuple[int, ...], dict[str, int]]:
    """Parse a Cayley<...> class name into (factory_name, pos_args, kw_args).

    Raises AttributeError if the name does not match the expected pattern.
    """
    m = _RE_NAME.match(name)
    if not m:
        raise AttributeError(
            f"Cannot resolve {name!r}: does not match Cayley<FactoryName>[args] pattern. "
            "Call list_available_factories() for available factory names."
        )

    pascal = m.group("pascal")
    factory_name = _pascal_to_snake(pascal)
    nums_str = m.group("nums") or ""

    if not nums_str:
        return factory_name, (), {}

    # Try backward-compat K<n> form first: e.g. "8K4"
    k_match = _RE_K_KWARG.match(nums_str)
    if k_match:
        first_arg = int(k_match.group("first"))
        k_val = int(k_match.group("kval"))
        # We use the kwarg name 'k' here; _resolve_kwargs_from_sig may override
        # it to match the factory's actual last parameter name.
        return factory_name, (first_arg,), {"k": k_val}

    # Try positional _ form: e.g. "8", "8_3", "8_4_2"
    if _RE_POSITIONAL.match(nums_str):
        pos_args = tuple(int(x) for x in nums_str.split("_"))
        return factory_name, pos_args, {}

    raise AttributeError(
        f"Cannot resolve {name!r}: numeric suffix {nums_str!r} is not parseable. "
        "Call list_available_factories() for available factory names."
    )


# ---------------------------------------------------------------------------
# Signature-aware kwarg resolution
# ---------------------------------------------------------------------------


def _resolve_kwargs_from_sig(
    factory_name: str,
    pos_args: tuple[int, ...],
    kw_args: dict[str, int],
) -> tuple[tuple[int, ...], dict[str, int]]:
    """Validate pos_args+kw_args against the factory signature.

    If kw_args contains {'k': v} from the K<n> form, rename 'k' to the
    factory's actual last-parameter name (handles cases where the param is
    not literally named 'k').

    If pos_args has more elements than the factory accepts, raises AttributeError.
    """
    from cayleypy.graphs_lib import PermutationGroups

    factory = getattr(PermutationGroups, factory_name, None)
    if factory is None:
        raise AttributeError(
            f"cayleypy.graphs_lib.PermutationGroups has no factory {factory_name!r}. "
            "Call list_available_factories() for available factory names."
        )

    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        # If signature cannot be introspected, pass through as-is.
        return pos_args, kw_args

    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    n_params = len(params)

    if not kw_args:
        # Pure positional: check count fits
        if len(pos_args) > n_params:
            raise AttributeError(
                f"Factory {factory_name!r} accepts {n_params} parameters but "
                f"{len(pos_args)} positional args were parsed from the class name. "
                "Call list_available_factories() for available factory names."
            )
        return pos_args, {}

    # We have kw_args (from K<n> form). The key is nominally 'k' but may need
    # renaming to match the factory's actual last parameter.
    if "k" in kw_args and n_params > 0:
        last_param_name = params[-1].name
        if last_param_name != "k":
            kw_args = {last_param_name: kw_args["k"]}
    return pos_args, kw_args


# ---------------------------------------------------------------------------
# Class factory
# ---------------------------------------------------------------------------


def _make_class(
    class_name: str,
    factory_name: str,
    pos_args: tuple[int, ...],
    kw_args: dict[str, int],
) -> type:
    """Dynamically construct a CayleyPuzzle subclass for the given factory call."""
    _pos = tuple(pos_args)
    _kw = dict(kw_args)
    _fname = factory_name

    def __init__(self, **kwargs):
        from cayleypy.graphs_lib import PermutationGroups

        factory = getattr(PermutationGroups, _fname)
        graph_def = factory(*_pos, **_kw)
        super(cls, self).__init__(graph_def, **kwargs)

    cls = type(
        class_name,
        (CayleyPuzzle,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "__qualname__": class_name,
            "__doc__": (
                f"Auto-generated CayleyPuzzle for "
                f"PermutationGroups.{factory_name}({_pos!r}, **{_kw!r})."
            ),
            "_cayleypy_factory": factory_name,
            "_cayleypy_args": _pos,
            "_cayleypy_kwargs": _kw,
        },
    )
    return cls


# ---------------------------------------------------------------------------
# Module-level __getattr__
# ---------------------------------------------------------------------------


def __getattr__(name: str):
    if not name.startswith("Cayley"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Return cached class if available.
    if name in _CLASS_CACHE:
        return _CLASS_CACHE[name]

    # Parse the name.
    try:
        factory_name, pos_args, kw_args = _parse_name(name)
    except AttributeError:
        raise

    # Resolve kwargs against factory signature (lazy import of cayleypy).
    try:
        pos_args, kw_args = _resolve_kwargs_from_sig(factory_name, pos_args, kw_args)
    except AttributeError:
        raise

    cls = _make_class(name, factory_name, pos_args, kw_args)
    _CLASS_CACHE[name] = cls
    # Also install into module globals so repeated attribute access is O(1)
    # and isinstance checks work correctly.
    globals()[name] = cls
    return cls


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def discover(factory_name: str, *args, **kwargs) -> type:
    """Build a CayleyPuzzle subclass for a named factory without name parsing.

    Args:
        factory_name: A snake_case name matching a PermutationGroups method,
            e.g. "pancake", "top_spin", "lrx".
        *args: Positional arguments forwarded to the factory.
        **kwargs: Keyword arguments forwarded to the factory.

    Returns:
        A new CayleyPuzzle subclass (or a cached one if the call was seen before).

    Example:
        PancakePuzzle5 = discover("pancake", 5)
        puzzle = PancakePuzzle5()
    """
    # Build a canonical cache key that identifies this exact factory call.
    kw_items = tuple(sorted(kwargs.items()))
    cache_key = (factory_name, args, kw_items)

    if cache_key in _DISCOVER_CACHE:
        return _DISCOVER_CACHE[cache_key]

    # Validate the factory exists.
    from cayleypy.graphs_lib import PermutationGroups

    factory = getattr(PermutationGroups, factory_name, None)
    if factory is None:
        raise AttributeError(
            f"cayleypy.graphs_lib.PermutationGroups has no factory {factory_name!r}. "
            "Call list_available_factories() for available factory names."
        )

    # Build a class name that is somewhat human-readable.
    parts = [factory_name.replace("_", " ").title().replace(" ", "")]
    parts += [str(a) for a in args]
    parts += [f"{k}{v}" for k, v in sorted(kwargs.items())]
    class_name = "Cayley" + "".join(parts)

    cls = _make_class(class_name, factory_name, tuple(args), dict(kwargs))
    _DISCOVER_CACHE[cache_key] = cls
    return cls


_DISCOVER_CACHE: dict = {}


def list_available_factories() -> list[str]:
    """Return the list of PermutationGroups factory names from cayleypy.

    Lazily imports cayleypy. Raises ImportError if cayleypy is not installed.
    """
    from cayleypy.graphs_lib import PermutationGroups

    return [
        name
        for name in dir(PermutationGroups)
        if not name.startswith("_") and callable(getattr(PermutationGroups, name, None))
    ]
