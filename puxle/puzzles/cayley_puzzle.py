"""CayleyPuzzle: PuXle adapter for cayleypy's CayleyGraphDef.

This module is **JAX-only**. It imports cayleypy lazily inside ``__init__``
and the convenience factory; nothing at module top level touches torch.
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass

if TYPE_CHECKING:
    # Type-only import — never executed at runtime.
    from cayleypy.cayley_graph_def import CayleyGraphDef


_ELT_DTYPE = jnp.int32

_GENERATORS_TYPE_PERMUTATION = "PERMUTATION"
_GENERATORS_TYPE_MATRIX = "MATRIX"


def _coerce_generators_type(raw) -> str:
    """Reduce cayleypy's ``GeneratorType`` enum (or a string) to one of the
    two supported tags."""
    name = getattr(raw, "name", None)
    if isinstance(name, str):
        token = name.upper()
    else:
        token = str(raw).upper()
    if "PERMUTATION" in token:
        return _GENERATORS_TYPE_PERMUTATION
    if "MATRIX" in token:
        return _GENERATORS_TYPE_MATRIX
    raise ValueError(
        f"Unsupported cayleypy generators_type: {raw!r}. "
        "Only PERMUTATION and MATRIX are supported."
    )


def _modular_matrix_inverse(mat: np.ndarray, modulo: int) -> np.ndarray:
    """Compute the inverse of an integer matrix modulo ``modulo`` using the
    adjugate and the modular inverse of the determinant.

    Raises ValueError if the matrix is not invertible mod ``modulo``.
    """
    m = mat.shape[0]
    if mat.shape != (m, m):
        raise ValueError(f"expected square matrix, got shape {mat.shape}")

    # Work in Python ints to avoid overflow during cofactor expansion.
    A = [[int(mat[i, j]) % int(modulo) for j in range(m)] for i in range(m)]

    det = _det_int(A) % int(modulo)
    det_inv = pow(det, -1, int(modulo))

    # Adjugate via cofactor expansion.
    adj = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            minor = _minor(A, i, j)
            cof = ((-1) ** (i + j)) * _det_int(minor)
            # adjugate is transpose of cofactor matrix.
            adj[j][i] = cof % int(modulo)

    inv = np.zeros((m, m), dtype=np.int64)
    for i in range(m):
        for j in range(m):
            inv[i, j] = (det_inv * adj[i][j]) % int(modulo)
    return inv


def _minor(A: list[list[int]], drop_row: int, drop_col: int) -> list[list[int]]:
    return [
        [A[i][j] for j in range(len(A)) if j != drop_col]
        for i in range(len(A))
        if i != drop_row
    ]


def _det_int(A: list[list[int]]) -> int:
    n = len(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    total = 0
    for j in range(n):
        sign = 1 if (j % 2 == 0) else -1
        total += sign * A[0][j] * _det_int(_minor(A, 0, j))
    return total


def _row_key_perm(row: np.ndarray) -> bytes:
    return row.astype(np.int64, copy=False).tobytes()


def _mat_key(mat: np.ndarray) -> bytes:
    return mat.astype(np.int64, copy=False).tobytes()


class CayleyPuzzle(Puzzle):
    """Adapter exposing a ``cayleypy.CayleyGraphDef`` as a PuXle ``Puzzle``.

    Supports two generator types from cayleypy:

    * ``PERMUTATION``: state is an int permutation vector of length
      ``n = len(graph_def.central_state)``. A move is
      ``new_state = jnp.take(state, perms[action])``.
    * ``MATRIX``: state is a flattened int vector of length ``n*m``. A move
      is ``new_vec = (M[action] @ state.reshape(n, m)) % modulo`` followed
      by a flatten back to ``(n*m,)``.

    All moves cost ``1.0``. Inverse closure is performed at construction
    time (numpy, not JIT) to populate ``inverse_action_map``.

    Args:
        graph_def: A ``cayleypy.CayleyGraphDef`` instance.
        ensure_inverse_closed: If True (default), augment the generator
            set with inverses not already present so that the puzzle is
            reversible. If False and the generators are not naturally
            closed, ``inverse_action_map`` returns ``None`` and a
            ``RuntimeWarning`` is emitted naming ``bi_astar`` /
            ``bi_qstar`` as unusable.
        num_shuffle: Default number of random moves applied by
            :meth:`get_initial_state` when scrambling from the central
            state.

    The single tensor field on the State dataclass is mode-resolved (not
    user-overridable): PERMUTATION → ``permutation``, MATRIX → ``vector``.
    """

    def __init__(
        self,
        graph_def: "CayleyGraphDef",
        *,
        ensure_inverse_closed: bool = True,
        num_shuffle: int = 100,
        **kwargs,
    ) -> None:
        if "state_field_name" in kwargs:
            raise TypeError(
                "CayleyPuzzle does not accept `state_field_name`; the field "
                "name is mode-resolved (PERMUTATION→'permutation', "
                "MATRIX→'vector')."
            )

        # 1. Read plain data fields from graph_def (no cayleypy methods).
        self._read_graph_def(graph_def)

        # 2. Build raw generator arrays and (optionally) close under
        #    inversion. All numpy work — runs before super().__init__.
        if self._generators_type == _GENERATORS_TYPE_PERMUTATION:
            raw_perms = self._build_permutation_generators(graph_def)
            augmented, inverse_map_np, naturally_closed = (
                self._close_under_inversion_permutations(raw_perms)
            )
            if ensure_inverse_closed:
                self._perms_np = augmented
                self._inverse_action_map_jax = jnp.asarray(
                    inverse_map_np, dtype=jnp.int32
                )
            else:
                self._perms_np = raw_perms
                if naturally_closed:
                    self._inverse_action_map_jax = jnp.asarray(
                        inverse_map_np[: raw_perms.shape[0]], dtype=jnp.int32
                    )
                else:
                    self._inverse_action_map_jax = None
                    warnings.warn(
                        "CayleyPuzzle: graph not closed under inversion; "
                        "bi_astar/bi_qstar will be unusable",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self._perms_jax = jnp.asarray(self._perms_np, dtype=jnp.int32)
            self.action_size = int(self._perms_np.shape[0])

        elif self._generators_type == _GENERATORS_TYPE_MATRIX:
            # Construction-time matmul overflow precondition. JAX silently
            # truncates int64 → int32 unless x64 is enabled, so the active
            # ceiling depends on the runtime config.
            modulo = int(self._modulo)
            self._x64_enabled = bool(jax.config.read("jax_enable_x64"))
            self._matmul_dtype = jnp.int64 if self._x64_enabled else jnp.int32
            overflow_ceiling = 2**63 if self._x64_enabled else 2**31
            if int(self._m) * (modulo - 1) ** 2 >= overflow_ceiling:
                mode = "x64 enabled" if self._x64_enabled else "x64 disabled"
                raise ValueError(
                    "MATRIX kernel matmul overflow precondition violated "
                    f"(JAX {mode}): m={self._m}, modulo={modulo} "
                    f"(require m * (modulo - 1) ** 2 < {overflow_ceiling}). "
                    "Enable JAX x64 via "
                    "`jax.config.update('jax_enable_x64', True)` to raise "
                    "the ceiling to 2 ** 63."
                )

            raw_mats = self._build_matrix_generators(graph_def)
            augmented, inverse_map_np, naturally_closed = (
                self._close_under_inversion_matrices(raw_mats)
            )
            if ensure_inverse_closed:
                self._mats_np = augmented
                self._inverse_action_map_jax = jnp.asarray(
                    inverse_map_np, dtype=jnp.int32
                )
            else:
                self._mats_np = raw_mats
                if naturally_closed:
                    self._inverse_action_map_jax = jnp.asarray(
                        inverse_map_np[: raw_mats.shape[0]], dtype=jnp.int32
                    )
                else:
                    self._inverse_action_map_jax = None
                    warnings.warn(
                        "CayleyPuzzle: graph not closed under inversion; "
                        "bi_astar/bi_qstar will be unusable",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self._mats_jax = jnp.asarray(self._mats_np, dtype=jnp.int32)
            self.action_size = int(self._mats_np.shape[0])

        else:  # pragma: no cover - validated in _read_graph_def
            raise ValueError(f"Unsupported generators_type: {self._generators_type!r}")

        # 3. Bind the per-action kernel once at construction (no
        #    traced-time switching on generator type).
        if self._generators_type == _GENERATORS_TYPE_PERMUTATION:
            self._step_kernel = self._make_permutation_kernel()
        else:
            self._step_kernel = self._make_matrix_kernel()

        # 4. Stash construction-only options.
        self._num_shuffle = int(num_shuffle)
        self._ensure_inverse_closed = bool(ensure_inverse_closed)

        # 5. Pre-resolve the central state for fast SolveConfig builds.
        self._central_state_jax = jnp.asarray(self._central_state_np, dtype=jnp.int32)

        # 6. Let the base class build State/SolveConfig and JIT methods.
        super().__init__(**kwargs)

    @classmethod
    def from_cayleypy_factory(
        cls,
        name: str,
        *factory_args,
        ensure_inverse_closed: bool = True,
        num_shuffle: int = 100,
        **kwargs,
    ) -> "CayleyPuzzle":
        """Build a ``CayleyPuzzle`` from a named cayleypy registry entry.

        Lookup order: ``PermutationGroups`` → ``MatrixGroups`` → ``Puzzles``.
        First match wins; on name collision, construct the
        ``CayleyGraphDef`` directly and pass it to ``cls(graph_def, ...)``
        to disambiguate.
        """
        try:
            import cayleypy  # noqa: F401  (lazy)
        except ImportError as exc:
            raise ImportError(
                "cayleypy is required for CayleyPuzzle. Install the "
                "[cayley] extra: pip install puxle[cayley]"
            ) from exc

        from cayleypy import MatrixGroups, PermutationGroups, Puzzles

        for registry in (PermutationGroups, MatrixGroups, Puzzles):
            entry = getattr(registry, name, None)
            if entry is None:
                continue
            if not callable(entry):
                continue
            graph_def = entry(*factory_args, **kwargs.pop("factory_kwargs", {}))
            return cls(
                graph_def,
                ensure_inverse_closed=ensure_inverse_closed,
                num_shuffle=num_shuffle,
                **kwargs,
            )
        raise ValueError(
            f"cayleypy registry has no entry named {name!r} in "
            "PermutationGroups, MatrixGroups, or Puzzles."
        )

    def define_state_class(self) -> PuzzleState:
        str_parser = self.get_string_parser()
        state_length = int(self._state_length)

        if self._generators_type == _GENERATORS_TYPE_PERMUTATION:

            @state_dataclass
            class State:
                permutation: FieldDescriptor.tensor(
                    dtype=jnp.int32, shape=(state_length,)
                )

                def __str__(self, **kwargs):
                    return str_parser(self, **kwargs)

            return State

        @state_dataclass
        class State:
            vector: FieldDescriptor.tensor(dtype=jnp.int32, shape=(state_length,))

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

        return State

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        if self._generators_type == _GENERATORS_TYPE_PERMUTATION:
            target = self.State(permutation=self._central_state_jax)
        else:
            target = self.State(vector=self._central_state_jax)
        return self.SolveConfig(TargetState=target)

    def get_initial_state(
        self,
        solve_config: Puzzle.SolveConfig,
        key=None,
        data=None,
    ) -> "CayleyPuzzle.State":
        return self._get_shuffled_state(
            solve_config, solve_config.TargetState, key, self._num_shuffle
        )

    def _apply(
        self,
        solve_config: Puzzle.SolveConfig,
        state: "CayleyPuzzle.State",
        action: chex.Array,
    ) -> tuple["CayleyPuzzle.State", chex.Array]:
        """Pure transition: every generator is valid and costs ``1.0``."""
        return self._step_kernel(state, action)

    def is_solved(
        self, solve_config: Puzzle.SolveConfig, state: "CayleyPuzzle.State"
    ) -> bool:
        return state == solve_config.TargetState

    @property
    def inverse_action_map(self) -> Optional[jnp.ndarray]:
        return self._inverse_action_map_jax

    def get_string_parser(self) -> Callable:
        graph_name = self._graph_name

        if self._generators_type == _GENERATORS_TYPE_PERMUTATION:

            def parser(state: "CayleyPuzzle.State", **kwargs) -> str:
                values = " ".join(str(int(x)) for x in state.permutation)
                return f"{graph_name}: {values}"

            return parser

        def parser(state: "CayleyPuzzle.State", **kwargs) -> str:
            values = " ".join(str(int(x)) for x in state.vector)
            return f"{graph_name}: {values}"

        return parser

    def get_img_parser(self) -> Callable:
        state_length = int(self._state_length)
        is_perm = self._generators_type == _GENERATORS_TYPE_PERMUTATION

        def img_func(state: "CayleyPuzzle.State", **kwargs) -> np.ndarray:
            vec = np.asarray(state.permutation if is_perm else state.vector)
            if vec.size == 0:
                return np.zeros((1, 1, 3), dtype=np.uint8)
            max_val = int(vec.max()) if vec.size else 1
            denom = max(1, max_val)
            scaled = (vec.astype(np.int64) * 255 // denom).clip(0, 255)
            scaled = scaled.astype(np.uint8)
            strip = np.stack(
                [scaled, scaled, scaled], axis=-1
            )  # shape (state_length, 3)
            img = strip.reshape(1, state_length, 3)
            return img

        return img_func

    def action_to_string(self, action: int) -> str:
        return f"{self._graph_name}.gen[{int(action)}]"

    # ----- Internal helpers -----

    def _read_graph_def(self, graph_def: "CayleyGraphDef") -> None:
        """Validate and copy plain-data fields from ``graph_def``.

        Lazy-imports ``cayleypy`` only to surface a friendly ``ImportError``
        when the optional extra is missing AND the user has handed in a
        non-cayleypy object.
        """
        try:
            generators_type_raw = graph_def.generators_type
        except AttributeError as exc:
            # The user passed something that is not a CayleyGraphDef; the
            # most common cause is that cayleypy is not installed and they
            # tried to construct with a placeholder.
            try:
                import cayleypy  # noqa: F401
            except ImportError as ie:
                raise ImportError(
                    "cayleypy is required for CayleyPuzzle. Install the "
                    "[cayley] extra: pip install puxle[cayley]"
                ) from ie
            raise TypeError(
                "graph_def must be a cayleypy.CayleyGraphDef; got "
                f"{type(graph_def).__name__}"
            ) from exc

        self._generators_type = _coerce_generators_type(generators_type_raw)
        self._graph_name = str(getattr(graph_def, "name", "cayley"))

        central_state = np.asarray(graph_def.central_state, dtype=np.int64)
        if central_state.ndim != 1:
            raise ValueError(
                f"central_state must be 1-D; got shape {central_state.shape}"
            )
        self._central_state_np = central_state.astype(np.int32, copy=False)
        self._state_length = int(self._central_state_np.shape[0])

        if self._generators_type == _GENERATORS_TYPE_MATRIX:
            generators_matrices = getattr(graph_def, "generators_matrices", None)
            if not generators_matrices:
                raise ValueError("MATRIX graph_def has no generators_matrices.")
            first_mat = np.asarray(generators_matrices[0].matrix)
            if first_mat.ndim != 2 or first_mat.shape[0] != first_mat.shape[1]:
                raise ValueError(
                    f"MATRIX generators must be square; got shape {first_mat.shape}"
                )
            self._m = int(first_mat.shape[0])

            modulos = {int(g.modulo) for g in generators_matrices}
            if len(modulos) != 1:
                raise ValueError(
                    f"MATRIX generators have non-uniform moduli: {modulos}"
                )
            self._modulo = int(modulos.pop())
            if self._state_length % self._m != 0:
                raise ValueError(
                    "central_state length "
                    f"{self._state_length} is not divisible by m={self._m}."
                )
            self._n = self._state_length // self._m

    def _build_permutation_generators(self, graph_def: "CayleyGraphDef") -> np.ndarray:
        perms = np.asarray(graph_def.generators_permutations, dtype=np.int64)
        if perms.ndim != 2 or perms.shape[1] != self._state_length:
            raise ValueError(
                "generators_permutations must have shape (K, n) with "
                f"n={self._state_length}; got {perms.shape}"
            )
        return perms.astype(np.int32, copy=False)

    def _build_matrix_generators(self, graph_def: "CayleyGraphDef") -> np.ndarray:
        mats = np.stack(
            [
                np.asarray(g.matrix, dtype=np.int64)
                for g in graph_def.generators_matrices
            ],
            axis=0,
        )
        if mats.ndim != 3 or mats.shape[1] != self._m or mats.shape[2] != self._m:
            raise ValueError(
                "generators_matrices must yield shape (K, m, m) with "
                f"m={self._m}; got {mats.shape}"
            )
        modulo = int(self._modulo)
        mats = mats % modulo
        return mats.astype(np.int32, copy=False)

    def _close_under_inversion_permutations(
        self, perms_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Augment with ``argsort`` inverses; dedupe; return mapping.

        Returns:
            (augmented_perms, inverse_action_map_np, naturally_closed)
        """
        # Dedupe inputs first while preserving order.
        seen: dict[bytes, int] = {}
        deduped: list[np.ndarray] = []
        for row in perms_np:
            key = _row_key_perm(row)
            if key not in seen:
                seen[key] = len(deduped)
                deduped.append(row.astype(np.int32, copy=True))

        naturally_closed = True
        for row in list(deduped):
            inv_row = np.argsort(row).astype(np.int32)
            key = _row_key_perm(inv_row)
            if key not in seen:
                naturally_closed = False
                seen[key] = len(deduped)
                deduped.append(inv_row)

        augmented = np.stack(deduped, axis=0)
        n_aug = augmented.shape[0]
        inverse_map = np.empty((n_aug,), dtype=np.int32)
        for i in range(n_aug):
            inv_row = np.argsort(augmented[i]).astype(np.int32)
            inverse_map[i] = seen[_row_key_perm(inv_row)]
        return augmented, inverse_map, naturally_closed

    def _close_under_inversion_matrices(
        self, mats_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if self._m > 16:
            raise NotImplementedError(
                "Modular matrix inverse only supported for m <= 16; "
                f"got m={self._m}. Pass ensure_inverse_closed=False to "
                "skip inverse closure."
            )
        if self._modulo > 2**31 - 1:
            raise NotImplementedError(
                "Modular matrix inverse only supported for modulo "
                f"<= 2**31 - 1; got modulo={self._modulo}. Pass "
                "ensure_inverse_closed=False to skip inverse closure."
            )

        modulo = int(self._modulo)
        seen: dict[bytes, int] = {}
        deduped: list[np.ndarray] = []
        for mat in mats_np:
            key = _mat_key(mat)
            if key not in seen:
                seen[key] = len(deduped)
                deduped.append(mat.astype(np.int32, copy=True))

        naturally_closed = True
        for mat in list(deduped):
            inv_mat_int64 = _modular_matrix_inverse(mat, modulo)
            inv_mat = inv_mat_int64.astype(np.int32, copy=False)
            key = _mat_key(inv_mat)
            if key not in seen:
                naturally_closed = False
                seen[key] = len(deduped)
                deduped.append(inv_mat)

        augmented = np.stack(deduped, axis=0)
        n_aug = augmented.shape[0]
        inverse_map = np.empty((n_aug,), dtype=np.int32)
        for i in range(n_aug):
            inv_mat_int64 = _modular_matrix_inverse(augmented[i], modulo)
            inv_mat = inv_mat_int64.astype(np.int32, copy=False)
            inverse_map[i] = seen[_mat_key(inv_mat)]
        return augmented, inverse_map, naturally_closed

    def _make_permutation_kernel(self) -> Callable:
        perms_jax = self._perms_jax  # shape (A, n) int32

        def kernel(state, action):
            new_perm = jnp.take(state.permutation, perms_jax[action])
            return self.State(permutation=new_perm), 1.0

        return kernel

    def _make_matrix_kernel(self) -> Callable:
        mats_jax = self._mats_jax  # shape (A, m, m) int32
        matmul_dtype = self._matmul_dtype
        modulo = matmul_dtype(self._modulo)
        n_rows = int(self._n)
        m = int(self._m)
        state_length = int(self._state_length)

        def kernel(state, action):
            vec2d = state.vector.reshape(n_rows, m).astype(matmul_dtype)
            M = mats_jax[action].astype(matmul_dtype)
            new2d = (M @ vec2d) % modulo
            new_vec = new2d.reshape(state_length).astype(_ELT_DTYPE)
            return self.State(vector=new_vec), 1.0

        return kernel
