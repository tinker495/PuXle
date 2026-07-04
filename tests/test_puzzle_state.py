import jax.numpy as jnp

from puxle.core.puzzle_state import FieldDescriptor, state_dataclass


def test_state_dataclass_standard_packing():
    @state_dataclass
    class StandardState:
        board: FieldDescriptor.packed_tensor(shape=(4,), packed_bits=2)

    state = StandardState.from_unpacked(board=jnp.array([1, 2, 3, 0], dtype=jnp.uint8))
    assert state.board_unpacked.tolist() == [1, 2, 3, 0]
    updated = state.set_unpacked(board=jnp.array([0, 1, 2, 3], dtype=jnp.uint8))
    assert updated.board_unpacked.tolist() == [0, 1, 2, 3]


def test_state_dataclass_bitpack_off_does_not_add_identity_packing():
    @state_dataclass(bitpack="off")
    class PlainState:
        board: FieldDescriptor.tensor(shape=(4,), dtype=jnp.uint8)

    state = PlainState(board=jnp.array([1, 2, 3, 0], dtype=jnp.uint8))
    assert not hasattr(state, "packed")
    assert not hasattr(state, "unpacked")
