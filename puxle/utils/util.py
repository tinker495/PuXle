from typing import Type, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from xtructure import StructuredType

T = TypeVar("T")


def to_uint8(input: chex.Array, active_bits: int = 1) -> chex.Array:
    """
    Efficiently pack arrays into uint8 format with support for 1-8 bits per value.
    Now supports all bit widths (1-8) efficiently, including 3,5,6,7 bits, using only uint32 accumulators.
    """
    assert 1 <= active_bits <= 8, f"active_bits must be 1-8, got {active_bits}"

    if active_bits == 1:
        # Handle boolean arrays efficiently
        if input.dtype == jnp.bool_:
            flatten_input = input.reshape((-1,))
            return jnp.packbits(flatten_input, axis=-1, bitorder="little")
        else:
            # Convert integer input to boolean for 1-bit packing
            flatten_input = (input != 0).reshape((-1,))
            return jnp.packbits(flatten_input, axis=-1, bitorder="little")
    elif active_bits in (2, 4, 8):
        # Handle multi-bit integer arrays for 2, 4, 8 bits efficiently
        assert jnp.issubdtype(input.dtype, jnp.integer), (
            f"Input must be integer array for active_bits={active_bits} > 1, got dtype={input.dtype}"
        )
        values_flat = input.flatten()
        if active_bits == 8:
            return values_flat.astype(jnp.uint8)
        values_per_byte = 8 // active_bits
        padding_needed = (values_per_byte - (len(values_flat) % values_per_byte)) % values_per_byte
        if padding_needed > 0:
            values_flat = jnp.concatenate([values_flat, jnp.zeros(padding_needed, dtype=values_flat.dtype)])
        grouped_values = values_flat.reshape(-1, values_per_byte)
        def pack_group(group):
            result = jnp.uint8(0)
            for i, val in enumerate(group):
                result = result | (val.astype(jnp.uint8) << (i * active_bits))
            return result
        return jax.vmap(pack_group)(grouped_values)
    else:
        # Efficient block-based packing for 3,5,6,7 bits using only uint32
        assert jnp.issubdtype(input.dtype, jnp.integer), (
            f"Input must be integer array for active_bits={active_bits} > 1, got dtype={input.dtype}"
        )
        values_flat = input.flatten()
        L = np.lcm(active_bits, 8)  # total bits per block
        num_values_per_block = L // active_bits
        num_bytes_per_block = L // 8
        padding_needed = (num_values_per_block - (len(values_flat) % num_values_per_block)) % num_values_per_block
        if padding_needed > 0:
            values_flat = jnp.concatenate([values_flat, jnp.zeros(padding_needed, dtype=values_flat.dtype)])
        grouped_values = values_flat.reshape(-1, num_values_per_block)
        if L <= 32:
            # Can use a single uint32 accumulator
            def pack_block(group):
                acc = jnp.uint32(0)
                for i in range(num_values_per_block):
                    acc = acc | (group[i].astype(jnp.uint32) << (i * active_bits))
                return jnp.array([(acc >> (8 * j)) & 0xFF for j in range(num_bytes_per_block)], dtype=jnp.uint8)
            packed_blocks = jax.vmap(pack_block)(grouped_values)
            return packed_blocks.reshape(-1)
        else:
            # Use uint32 accumulator, but handle >32 bits (for 5,7 bits)
            def pack_block(group):
                packed_bytes = jnp.zeros((num_bytes_per_block,), dtype=jnp.uint8)
                acc = jnp.uint32(0)
                bits_in_acc = 0
                byte_idx = 0
                for i in range(num_values_per_block):
                    acc = acc | (group[i].astype(jnp.uint32) << bits_in_acc)
                    bits_in_acc += active_bits
                    while bits_in_acc >= 8:
                        packed_bytes = packed_bytes.at[byte_idx].set(acc & 0xFF)
                        acc = acc >> 8
                        bits_in_acc -= 8
                        byte_idx += 1
                if byte_idx < num_bytes_per_block:
                    packed_bytes = packed_bytes.at[byte_idx].set(acc & 0xFF)
                return packed_bytes
            packed_blocks = jax.vmap(pack_block)(grouped_values)
            return packed_blocks.reshape(-1)


def from_uint8(
    packed_bytes: chex.Array, target_shape: tuple[int, ...], active_bits: int = 1
) -> chex.Array:
    """
    Efficiently unpack uint8 array back to original format.
    Now supports all bit widths (1-8) efficiently, including 3,5,6,7 bits, using only uint32 accumulators.
    """
    assert packed_bytes.dtype == jnp.uint8, f"Input must be uint8, got {packed_bytes.dtype}"
    assert 1 <= active_bits <= 8, f"active_bits must be 1-8, got {active_bits}"

    num_target_elements = np.prod(target_shape)
    assert num_target_elements > 0, f"num_target_elements={num_target_elements} must be positive"

    if active_bits == 1:
        # Unpack to boolean array
        all_unpacked_bits = jnp.unpackbits(
            packed_bytes, count=num_target_elements, bitorder="little"
        )
        return all_unpacked_bits.reshape(target_shape).astype(jnp.bool_)
    elif active_bits in (2, 4, 8):
        # Direct copy for 8-bit values, or efficient unpack for 2,4 bits
        if active_bits == 8:
            assert len(packed_bytes) >= num_target_elements, "Not enough packed data"
            return packed_bytes[:num_target_elements].reshape(target_shape)
        values_per_byte = 8 // active_bits
        mask = (1 << active_bits) - 1
        def unpack_byte(byte_val):
            values = []
            for i in range(values_per_byte):
                val = (byte_val >> (i * active_bits)) & mask
                values.append(val)
            return jnp.array(values)
        unpacked_groups = jax.vmap(unpack_byte)(packed_bytes)
        all_values = unpacked_groups.flatten()
        assert len(all_values) >= num_target_elements, "Not enough unpacked values"
        return all_values[:num_target_elements].reshape(target_shape).astype(jnp.uint8)
    else:
        # Efficient block-based unpacking for 3,5,6,7 bits using only uint32
        L = np.lcm(active_bits, 8)
        num_values_per_block = L // active_bits
        num_bytes_per_block = L // 8
        total_blocks = (len(packed_bytes) + num_bytes_per_block - 1) // num_bytes_per_block
        padding_needed = total_blocks * num_bytes_per_block - len(packed_bytes)
        if padding_needed > 0:
            packed_bytes = jnp.concatenate([packed_bytes, jnp.zeros(padding_needed, dtype=packed_bytes.dtype)])
        grouped_bytes = packed_bytes.reshape(-1, num_bytes_per_block)
        mask = (1 << active_bits) - 1
        if L <= 32:
            def unpack_block(byte_group):
                acc = jnp.uint32(0)
                for j in range(num_bytes_per_block):
                    acc = acc | (byte_group[j].astype(jnp.uint32) << (8 * j))
                values = [(acc >> (i * active_bits)) & mask for i in range(num_values_per_block)]
                return jnp.array(values, dtype=jnp.uint8)
            unpacked_blocks = jax.vmap(unpack_block)(grouped_bytes)
            all_values = unpacked_blocks.flatten()
            assert len(all_values) >= num_target_elements, "Not enough unpacked values"
            return all_values[:num_target_elements].reshape(target_shape).astype(jnp.uint8)
        else:
            def unpack_block(byte_group):
                values = jnp.zeros((num_values_per_block,), dtype=jnp.uint8)
                acc = jnp.uint32(0)
                bits_in_acc = 0
                byte_idx = 0
                for i in range(num_values_per_block):
                    while bits_in_acc < active_bits:
                        if byte_idx < num_bytes_per_block:
                            acc = acc | (byte_group[byte_idx].astype(jnp.uint32) << bits_in_acc)
                            bits_in_acc += 8
                            byte_idx += 1
                    values = values.at[i].set(acc & mask)
                    acc = acc >> active_bits
                    bits_in_acc -= active_bits
                return values
            unpacked_blocks = jax.vmap(unpack_block)(grouped_bytes)
            all_values = unpacked_blocks.flatten()
            assert len(all_values) >= num_target_elements, "Not enough unpacked values"
            return all_values[:num_target_elements].reshape(target_shape).astype(jnp.uint8)



def pack_variable_bits(values_and_bits: list[tuple[chex.Array, int]]) -> chex.Array:
    """
    Pack multiple arrays with different bit requirements into a single uint8 array.
    
    Args:
        values_and_bits: List of (values_array, bits_per_value) tuples
        
    Returns:
        Packed uint8 array with metadata for unpacking
        
    Example:
        # Pack different data types together efficiently
        bool_array = jnp.array([True, False, True])  # 1 bit each
        nibble_array = jnp.array([3, 7, 1])         # 4 bits each  
        byte_array = jnp.array([255, 128])          # 8 bits each
        
        packed = pack_variable_bits([
            (bool_array, 1),
            (nibble_array, 4), 
            (byte_array, 8)
        ])
    """
    if not values_and_bits:
        return jnp.array([], dtype=jnp.uint8)
    
    # Pack metadata: number of arrays, then for each array: (length, bits_per_value)
    metadata = [len(values_and_bits)]
    packed_arrays = []
    
    for values, bits in values_and_bits:
        values_flat = values.flatten()
        metadata.extend([len(values_flat), bits])
        packed_arrays.append(to_uint8(values_flat, bits))
    
    # Pack metadata as uint8 (assume metadata values fit in uint8)
    metadata_packed = jnp.array(metadata, dtype=jnp.uint8)
    
    # Concatenate metadata and all packed arrays
    return jnp.concatenate([metadata_packed] + packed_arrays)


def unpack_variable_bits(packed_data: chex.Array, target_shapes: list[tuple[int, ...]]) -> list[chex.Array]:
    """
    Unpack variable bit data back to original arrays.
    
    Args:
        packed_data: Packed uint8 array from pack_variable_bits
        target_shapes: List of target shapes for each array
        
    Returns:
        List of unpacked arrays
    """
    if len(packed_data) == 0:
        return []
    
    # Read metadata
    num_arrays = int(packed_data[0])
    metadata_size = 1 + num_arrays * 2
    
    assert len(target_shapes) == num_arrays, f"Expected {num_arrays} shapes, got {len(target_shapes)}"
    
    # Parse metadata for each array
    arrays_info = []
    for i in range(num_arrays):
        length = int(packed_data[1 + i * 2])
        bits = int(packed_data[1 + i * 2 + 1])
        arrays_info.append((length, bits))
    
    # Unpack each array
    current_pos = metadata_size
    results = []
    
    for i, (target_shape, (length, bits)) in enumerate(zip(target_shapes, arrays_info)):
        # Calculate how many bytes this array needs
        if bits == 1:
            bytes_needed = (length + 7) // 8  # Round up for bit packing
        elif bits == 8:
            bytes_needed = length
        else:
            values_per_byte = 8 // bits
            bytes_needed = (length + values_per_byte - 1) // values_per_byte
        
        # Extract data for this array
        array_data = packed_data[current_pos:current_pos + bytes_needed]
        
        # Unpack and reshape
        unpacked = from_uint8(array_data, target_shape, bits)
        results.append(unpacked)
        
        current_pos += bytes_needed
    
    return results


def add_img_parser(cls: Type[T], imgfunc: callable) -> Type[T]:
    """
    This function is a decorator that adds a __str__ method to
    the class that returns a string representation of the class.
    """

    def get_img(self, **kwargs) -> np.ndarray:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            return imgfunc(self, **kwargs)
        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.batch_shape
            batch_len = (
                jnp.prod(jnp.array(batch_shape)) if len(batch_shape) != 1 else batch_shape[0]
            )
            results = []
            for i in trange(batch_len):
                index = jnp.unravel_index(i, batch_shape)
                current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                results.append(imgfunc(current_state, **kwargs))
            results = np.stack(results, axis=0)
            return results
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    setattr(cls, "img", get_img)
    return cls


def coloring_str(string: str, color: tuple[int, int, int]) -> str:
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m{string}\x1b[0m"
