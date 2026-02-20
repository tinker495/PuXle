import jax.numpy as jnp
import pytest
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass

def test_state_dataclass_standard_packing():
    @state_dataclass
    class StandardState:
        board: FieldDescriptor.packed_tensor(shape=(4,), packed_bits=2)

    # Should have packed and unpacked properties
    state = StandardState(board=jnp.array([1, 2, 3, 0], dtype=jnp.uint8))
    assert hasattr(state, "packed")
    assert hasattr(state, "unpacked")

def test_state_dataclass_fallback_properties():
    # When bitpack="auto" is removed or not used properly,
    # it should provide backwards compatible identity properties
    @state_dataclass(bitpack="off")
    class FallbackState:
        board: FieldDescriptor.tensor(shape=(4,), dtype=jnp.uint8)

    state = FallbackState(board=jnp.array([1, 2, 3, 0], dtype=jnp.uint8))
    assert state.packed is state
    assert state.unpacked is state

def test_state_dataclass_partial_packing_error():
    # Test that providing only one of packed/unpacked raises an error
    with pytest.raises(ValueError, match="State class must implement both packing and unpacking"):
        @state_dataclass(bitpack="off")
        class PartialState:
            board: FieldDescriptor.tensor(shape=(4,), dtype=jnp.uint8)

            @property
            def packed(self):
                return self
