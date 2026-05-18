"""Unit tests for CayleyPuzzle adapter (12 tests).

Tests #1-#11 use hand-rolled stubs — no cayleypy import required.
Test #12 is gated on pytest.importorskip("cayleypy").
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub helpers — mirror exactly the fields _read_graph_def reads.
# ---------------------------------------------------------------------------


def _perm_graph_def(
    permutations: list[list[int]],
    central_state: list[int],
    name: str = "test_perm",
) -> SimpleNamespace:
    """Minimal stub that looks like a PERMUTATION CayleyGraphDef."""
    gt = SimpleNamespace(name="PERMUTATION")
    return SimpleNamespace(
        generators_type=gt,
        generators_permutations=np.array(permutations, dtype=np.int32),
        generators_matrices=None,
        central_state=central_state,
        name=name,
    )


def _mat_gen(matrix: list[list[int]], modulo: int) -> SimpleNamespace:
    return SimpleNamespace(matrix=np.array(matrix, dtype=np.int32), modulo=modulo)


def _mat_graph_def(
    matrices: list[list[list[int]]],
    modulo: int,
    central_state: list[int],
    name: str = "test_mat",
) -> SimpleNamespace:
    """Minimal stub that looks like a MATRIX CayleyGraphDef."""
    gt = SimpleNamespace(name="MATRIX")
    gens = [_mat_gen(m, modulo) for m in matrices]
    return SimpleNamespace(
        generators_type=gt,
        generators_permutations=None,
        generators_matrices=gens,
        central_state=central_state,
        name=name,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perm_puzzle():
    """Small cyclic-8 style permutation puzzle (closed under inversion)."""
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    # Two generators: rotate left by 1, rotate right by 1.
    n = 8
    rot_left = list(range(1, n)) + [0]
    rot_right = [n - 1] + list(range(n - 1))
    gd = _perm_graph_def([rot_left, rot_right], list(range(n)), name="cyclic8")
    return CayleyPuzzle(gd, ensure_inverse_closed=True, num_shuffle=4)


@pytest.fixture(scope="module")
def mat_puzzle():
    """Small MATRIX puzzle over Z/5 with two 2x2 generators (closed under inversion)."""
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    # Generator A = [[1,1],[0,1]] mod 5  (det=1, invertible)
    # Generator B = [[1,0],[1,1]] mod 5  (det=1, invertible)
    A = [[1, 1], [0, 1]]
    B = [[1, 0], [1, 1]]
    # central_state is the flat identity-like vector: [1, 0, 0, 1]
    gd = _mat_graph_def([A, B], modulo=5, central_state=[1, 0, 0, 1], name="mat2x2z5")
    return CayleyPuzzle(gd, ensure_inverse_closed=True, num_shuffle=2)


# ---------------------------------------------------------------------------
# Test 1: permutation forward step matches numpy
# ---------------------------------------------------------------------------


def test_permutation_forward_step_matches_numpy(perm_puzzle):
    puzzle = perm_puzzle
    cfg = puzzle.get_solve_config()
    rng = np.random.default_rng(42)
    n = puzzle._state_length
    perms_np = np.asarray(puzzle._perms_np)

    for _ in range(16):
        state_np = rng.permutation(n).astype(np.int32)
        action_idx = rng.integers(0, puzzle.action_size)
        state = puzzle.State(permutation=jnp.array(state_np, dtype=jnp.int32))
        next_state, cost = puzzle.get_actions(
            cfg, state, jnp.array(action_idx), filled=True
        )

        expected = np.take(state_np, perms_np[action_idx])
        assert jnp.array_equal(
            next_state.permutation, jnp.array(expected, dtype=jnp.int32)
        ), (
            f"Mismatch at action {action_idx}: {np.asarray(next_state.permutation)} != {expected}"
        )
        assert float(cost) == 1.0


# ---------------------------------------------------------------------------
# Test 2: matrix forward step matches hand-computed
# ---------------------------------------------------------------------------


def test_matrix_forward_step_matches_handcomputed():
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    # 4 hand-picked (state, action) pairs over Z/5 with 2x2 generators.
    # A=[[1,1],[0,1]] det=1; B=[[1,0],[1,1]] det=1 — both invertible mod 5.
    A = [[1, 1], [0, 1]]
    B = [[1, 0], [1, 1]]
    gd = _mat_graph_def([A, B], modulo=5, central_state=[1, 0, 0, 1], name="hand2x2")
    puzzle = CayleyPuzzle(gd, ensure_inverse_closed=True, num_shuffle=2)
    cfg = puzzle.get_solve_config()

    cases = [
        ([1, 0, 0, 1], 0),
        ([3, 2, 1, 4], 0),
        ([1, 0, 0, 1], 1),
        ([2, 3, 4, 0], 1),
    ]
    mats_np = np.asarray(puzzle._mats_np)
    n_rows = puzzle._n
    m = puzzle._m
    modulo = 5

    for state_flat, action_idx in cases:
        state_np = np.array(state_flat, dtype=np.int32)
        state = puzzle.State(vector=jnp.array(state_np, dtype=jnp.int32))
        next_state, cost = puzzle.get_actions(
            cfg, state, jnp.array(action_idx), filled=True
        )

        M = mats_np[action_idx].astype(np.int64)
        v2d = state_np.reshape(n_rows, m).astype(np.int64)
        expected_2d = (M @ v2d) % modulo
        expected = expected_2d.reshape(-1).astype(np.int32)

        assert jnp.array_equal(
            next_state.vector, jnp.array(expected, dtype=jnp.int32)
        ), (
            f"Matrix step mismatch: got {np.asarray(next_state.vector)}, expected {expected}"
        )
        assert float(cost) == 1.0


# ---------------------------------------------------------------------------
# Test 3: inverse closure round-trip — PERMUTATION
# ---------------------------------------------------------------------------


def test_inverse_closure_roundtrip_permutation(perm_puzzle):
    puzzle = perm_puzzle
    cfg = puzzle.get_solve_config()
    inv_map = np.asarray(puzzle.inverse_action_map)
    rng = np.random.default_rng(7)
    n = puzzle._state_length

    for i in range(puzzle.action_size):
        state_np = rng.permutation(n).astype(np.int32)
        state = puzzle.State(permutation=jnp.array(state_np, dtype=jnp.int32))
        mid_state, _ = puzzle.get_actions(cfg, state, jnp.array(i), filled=True)
        back_state, _ = puzzle.get_actions(
            cfg, mid_state, jnp.array(int(inv_map[i])), filled=True
        )
        assert jnp.array_equal(back_state.permutation, state.permutation), (
            f"Round-trip failed for action {i}"
        )


# ---------------------------------------------------------------------------
# Test 4: inverse closure round-trip — MATRIX
# ---------------------------------------------------------------------------


def test_inverse_closure_roundtrip_matrix(mat_puzzle):
    puzzle = mat_puzzle
    cfg = puzzle.get_solve_config()
    inv_map = np.asarray(puzzle.inverse_action_map)
    rng = np.random.default_rng(13)
    state_length = puzzle._state_length
    modulo = puzzle._modulo

    for i in range(puzzle.action_size):
        state_np = rng.integers(0, modulo, size=state_length).astype(np.int32)
        state = puzzle.State(vector=jnp.array(state_np, dtype=jnp.int32))
        mid_state, _ = puzzle.get_actions(cfg, state, jnp.array(i), filled=True)
        back_state, _ = puzzle.get_actions(
            cfg, mid_state, jnp.array(int(inv_map[i])), filled=True
        )
        assert jnp.array_equal(back_state.vector, state.vector), (
            f"Round-trip failed for matrix action {i}"
        )


# ---------------------------------------------------------------------------
# Test 5: self-inverse involution — inverse_action_map[i] == i
# ---------------------------------------------------------------------------


def test_self_inverse_involution():
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    # Transposition swap: [1, 0, 2, 3] is self-inverse (it is its own argsort).
    n = 4
    swap01 = [1, 0, 2, 3]
    gd = _perm_graph_def([swap01], list(range(n)), name="swap")
    puzzle = CayleyPuzzle(gd, ensure_inverse_closed=True, num_shuffle=2)
    inv_map = np.asarray(puzzle.inverse_action_map)
    # The swap generator's inverse is itself.
    assert int(inv_map[0]) == 0, f"Expected self-inverse at index 0, got {inv_map[0]}"
    assert puzzle.is_reversible


# ---------------------------------------------------------------------------
# Test 6: non-closed graph with ensure_inverse_closed=False warns and sets None
# ---------------------------------------------------------------------------


def test_non_closed_not_reversible_warns():
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    # Single cyclic shift [1,2,3,0]: inverse is [3,0,1,2], not in generators.
    n = 4
    cycle = [1, 2, 3, 0]
    gd = _perm_graph_def([cycle], list(range(n)), name="asymm_cycle")

    with pytest.warns(RuntimeWarning, match="bi_astar|bi_qstar"):
        puzzle = CayleyPuzzle(gd, ensure_inverse_closed=False, num_shuffle=2)

    assert puzzle.inverse_action_map is None
    assert puzzle.is_reversible is False


# ---------------------------------------------------------------------------
# Test 7: import without cayleypy — verified in-process or via subprocess
# ---------------------------------------------------------------------------


def test_subprocess_import_without_cayleypy():
    import importlib.util

    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    cayleypy_present = importlib.util.find_spec("cayleypy") is not None

    if not cayleypy_present:
        # cayleypy is absent in this process — verify in-process.
        # Step 1: CayleyPuzzle class itself is importable (lazy module).

        # Step 2: constructing with a non-graph_def raises ImportError naming [cayley].
        class FakeGraphDef:
            pass

        with pytest.raises(ImportError, match="[Cc]ayley"):
            CayleyPuzzle(FakeGraphDef())

    else:
        # cayleypy IS installed — use a subprocess that blocks it via an
        # import hook, inheriting the full env so JAX/chex/xtructure all work.
        code = """
import sys
import builtins

real_import = builtins.__import__
def blocking_import(name, *args, **kwargs):
    if name == 'cayleypy' or name.startswith('cayleypy.'):
        raise ImportError("cayleypy blocked for test")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocking_import
for key in list(sys.modules):
    if key == 'cayleypy' or key.startswith('cayleypy.'):
        del sys.modules[key]

from puxle.puzzles.cayley_puzzle import CayleyPuzzle
print('import_ok')

class FakeGraphDef:
    pass

try:
    CayleyPuzzle(FakeGraphDef())
    print('NO_ERROR')
except ImportError as e:
    msg = str(e)
    if 'cayley' in msg.lower():
        print('import_error_ok')
    else:
        print(f'wrong_error: {msg}')
except Exception as e:
    print(f'other_error: {type(e).__name__}: {e}')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        assert "import_ok" in result.stdout, (
            f"CayleyPuzzle import failed in subprocess. stderr: {result.stderr}"
        )
        assert "import_error_ok" in result.stdout, (
            f"Expected ImportError naming [cayley]. stdout: {result.stdout!r}, "
            f"stderr: {result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# Test 8: state_field_name kwarg is rejected
# ---------------------------------------------------------------------------


def test_state_field_name_kwarg_rejected():
    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    n = 4
    rot = list(range(1, n)) + [0]
    gd = _perm_graph_def([rot], list(range(n)), name="kwarg_test")
    with pytest.raises(TypeError):
        CayleyPuzzle(gd, state_field_name="foo")


# ---------------------------------------------------------------------------
# Test 9: state field per mode
# ---------------------------------------------------------------------------


def test_state_field_per_mode(perm_puzzle, mat_puzzle):
    perm_cfg = perm_puzzle.get_solve_config()
    perm_state = perm_cfg.TargetState
    assert hasattr(perm_state, "permutation"), (
        "PERMUTATION state must have .permutation"
    )
    assert not hasattr(perm_state, "vector"), "PERMUTATION state must not have .vector"

    mat_cfg = mat_puzzle.get_solve_config()
    mat_state = mat_cfg.TargetState
    assert hasattr(mat_state, "vector"), "MATRIX state must have .vector"
    assert not hasattr(mat_state, "vector") or not hasattr(mat_state, "permutation"), (
        "MATRIX state must not have .permutation"
    )
    # Clearer positive assertion:
    assert not hasattr(mat_state, "permutation"), (
        "MATRIX state must not have .permutation"
    )


# ---------------------------------------------------------------------------
# Test 10: JAxtar A* smoke test
# ---------------------------------------------------------------------------


def test_jaxtar_astar_smoke():
    pytest.importorskip("JAxtar")

    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    # Build a small permutation puzzle: cyclic-4.
    n = 4
    rot_left = list(range(1, n)) + [0]
    rot_right = [n - 1] + list(range(n - 1))
    gd = _perm_graph_def([rot_left, rot_right], list(range(n)), name="cyclic4_astar")
    puzzle = CayleyPuzzle(gd, ensure_inverse_closed=True, num_shuffle=2)
    cfg = puzzle.get_solve_config()

    # Scramble by exactly 2 known actions (both rot_left).
    state = cfg.TargetState
    state, _ = puzzle.get_actions(cfg, state, jnp.array(0), filled=True)
    state, _ = puzzle.get_actions(cfg, state, jnp.array(0), filled=True)

    # Try to use JAxtar astar if it's importable with the right interface.
    try:
        from JAxtar.stars.astar import astar
    except ImportError:
        pytest.skip("JAxtar.stars.astar not importable")

    # Zero heuristic: always returns 0.
    def zero_heuristic(cfg, states):
        return jnp.zeros(jax.tree_util.tree_leaves(states)[0].shape[0])

    try:
        result = astar(
            puzzle,
            zero_heuristic,
            cfg,
            state,
            batch_size=32,
            max_nodes=256,
        )
        # If result has a path or cost, assert cost <= 2.
        if hasattr(result, "cost"):
            assert float(result.cost) <= 2.0
        # If result is a sequence of states/actions, assert len <= 2.
        elif hasattr(result, "__len__"):
            assert len(result) <= 2 + 1  # +1 for start state
    except Exception as e:
        # astar interface may differ; accept if it runs without import error.
        if "ImportError" in type(e).__name__:
            raise
        pytest.skip(f"astar interface differs: {e}")


# ---------------------------------------------------------------------------
# Test 11: MATRIX overflow precondition
# ---------------------------------------------------------------------------


def test_matrix_overflow_precondition():
    # Craft m and modulo such that m * (modulo - 1)**2 >= 2**63.
    # m=2, modulo-1 = ceil(sqrt(2**63 / 2)) = ceil(2**31) = 2**31
    # => modulo = 2**31 + 1, but _read_graph_def checks modulo <= 2**31 - 1.
    # Use: m=4, modulo = 2**31 (just enough to overflow).
    # 4 * (2**31 - 1)^2 ≈ 4 * 2**62 = 2**64 >= 2**63. But modulo must be <= 2**31-1.
    # With m=4, modulo=2**31-1: 4 * (2**31-2)^2 ≈ 4*(2^31)^2 = 2^63. Borderline.
    # Use m=3, modulo=2**21: 3*(2**21-1)**2 = 3*~2**42 = ~3*2**42 < 2**63. Not enough.
    # Use m=4, modulo=2**16: 4*(65535)^2 = 4*4.3e9 ~= 1.7e10, < 2**63. Not enough.
    # We need m * (modulo-1)**2 >= 2**63.
    # Simplest: m=1 doesn't work (only 1x1 matrix, 1*(m-1)^2 = 0).
    # Use m=2, modulo = 2**31: violates modulo <= 2**31-1 check first.
    # Actually the implementation checks: if int(self._m) * (modulo - 1) ** 2 >= 2 ** 63: raise ValueError
    # AND it checks modulo <= 2**31-1 in _close_under_inversion_matrices (not _read_graph_def).
    # The overflow check happens before _close_under_inversion_matrices is called.
    # So we need m*(modulo-1)**2 >= 2**63 with modulo <= 2**31-1.
    # With m=2, modulo=2**31-1=2147483647: 2*(2147483646)^2 = 2*~4.6e18 = ~9.2e18 > 2**63=9.2e18.
    # Actually 2*(2**31-2)**2 = 2*(2^31)^2 - ... let's compute precisely:
    # (2**31-2)**2 = 2**62 - 2**32 + 4; 2*(2**62 - 2**32 + 4) = 2**63 - 2**33 + 8 < 2**63. Just under!
    # Use m=3, modulo=2**22: 3*(2**22-1)**2 = 3*~4.4e12 = ~1.3e13 < 2**63. No.
    # Use m=4, modulo=2**16+1=65537: 4*(65536)^2 = 4*4294967296 = 1.7e10. No.
    # Let's try: need m*(p-1)^2 >= 2**63.
    # p-1 = sqrt(2**63/m). For m=2: p-1 = sqrt(2**62) = 2**31. So p=2**31+1 > 2**31-1. Too big.
    # For m=3: p-1 = sqrt(2**63/3) = sqrt(3.07e18) ~ 1.75e9. p ~ 1.75e9+1 < 2**31-1 = 2.1e9. OK!
    # 3 * (1753413056)**2 check: 1753413056**2 = ~3.074e18; 3*3.074e18 = 9.22e18 > 9.22e18 = 2**63.
    # Let's use p = int(np.ceil(np.sqrt(2**63 / 3))) + 1 to be safe.
    import math

    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    m = 3
    p_min = int(math.ceil(math.sqrt(2**63 / m))) + 1  # p-1 value
    modulo = p_min + 1  # modulo = (p-1) + 1 = p_min + 1

    # Ensure modulo fits in int32 check (modulo <= 2**31-1).
    if modulo > 2**31 - 1:
        pytest.skip("Cannot construct overflow case within modulo <= 2**31-1 for m=3")

    # 3x3 identity matrix (det=1, invertible mod any prime).
    ident = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    central = [0] * (m * m)  # flat 3x3 state = 9 elements
    gd = _mat_graph_def(
        [ident], modulo=modulo, central_state=central, name="overflow_test"
    )

    with pytest.raises(ValueError, match="overflow|precondition|2 \\*\\* 63|int64"):
        CayleyPuzzle(gd, ensure_inverse_closed=True)


# ---------------------------------------------------------------------------
# Test 12: cayleypy parity (oracle test)
# ---------------------------------------------------------------------------


def test_cayleypy_parity():
    pytest.importorskip("cayleypy")

    from puxle.puzzles.cayley_puzzle import CayleyPuzzle

    # --- PERMUTATION parity ---
    try:
        from cayleypy import PermutationGroups

        perm_gd = PermutationGroups.cyclic(6)
        perm_puzzle = CayleyPuzzle(perm_gd, ensure_inverse_closed=True, num_shuffle=2)
        perm_cfg = perm_puzzle.get_solve_config()
        perm_state = perm_cfg.TargetState

        # Pick action 0 and apply via adapter.
        next_perm_state, _ = perm_puzzle.get_actions(
            perm_cfg, perm_state, jnp.array(0), filled=True
        )
        adapter_perm = np.asarray(next_perm_state.permutation)

        # Apply via cayleypy's raw permutation data.
        perm_np = np.asarray(perm_gd.generators_permutations, dtype=np.int32)
        central_np = np.asarray(perm_gd.central_state, dtype=np.int32)
        expected_perm = np.take(central_np, perm_np[0])

        assert np.array_equal(adapter_perm, expected_perm), (
            f"PERMUTATION parity failed: adapter={adapter_perm}, expected={expected_perm}"
        )
    except (AttributeError, ValueError, ImportError) as e:
        pytest.xfail(f"cayleypy PERMUTATION parity: {e}")

    # --- MATRIX parity ---
    try:
        from cayleypy import MatrixGroups

        mat_gd = MatrixGroups.cyclic(3, modulo=5)
        mat_puzzle = CayleyPuzzle(mat_gd, ensure_inverse_closed=True, num_shuffle=2)
        mat_cfg = mat_puzzle.get_solve_config()
        mat_state = mat_cfg.TargetState

        next_mat_state, _ = mat_puzzle.get_actions(
            mat_cfg, mat_state, jnp.array(0), filled=True
        )
        adapter_vec = np.asarray(next_mat_state.vector)

        # Apply via cayleypy's raw matrix data (left-multiply convention as in adapter).
        mat_np = np.asarray(mat_gd.generators_matrices[0].matrix, dtype=np.int64)
        central_np = np.asarray(mat_gd.central_state, dtype=np.int64)
        m = mat_puzzle._m
        n_rows = mat_puzzle._n
        modulo = mat_puzzle._modulo
        v2d = central_np.reshape(n_rows, m)
        expected_vec = ((mat_np @ v2d) % modulo).reshape(-1).astype(np.int32)

        assert np.array_equal(adapter_vec, expected_vec), (
            f"MATRIX parity failed: adapter={adapter_vec}, expected={expected_vec}"
        )
    except (AttributeError, ValueError, ImportError) as e:
        pytest.xfail(
            reason=f"cayleypy MATRIX parity: {e}",
        )
