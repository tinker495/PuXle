"""Xtructure storage for states reachable from one compiled PuzzleScript level."""

from typing import TYPE_CHECKING, Annotated

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from xtructure import FieldDescriptor, Xtructurable
from xtructure.core.layout.bitpack import packed_num_bytes


def _rules(blocks):
    for rule in blocks:
        if hasattr(rule, "rules"):
            yield from _rules(rule.rules)
        else:
            yield rule


def _mutable_layers(env, rules):
    """Keep collision layers that can receive a player force or a rule write."""
    pending = {
        token
        for rule in rules
        for cell in jax.tree.leaves((rule.left_kernels, rule.right_kernels))
        for token in str(cell).lower().split()
    }
    # A replacement can remove a matched object or displace another occupant.
    # Resolve names on both sides through the compiled and source legends.
    touched = np.zeros(env.n_objs, dtype=bool)
    touched[env.player_idxs] = True
    modifiers = {
        ">",
        "<",
        "^",
        "v",
        "...",
        "no",
        "up",
        "down",
        "left",
        "right",
        "horizontal",
        "vertical",
        "parallel",
        "perpendicular",
        "moving",
        "stationary",
        "orthogonal",
        "orthoganal",
        "random",
        "randomdir",
        "action",
    }
    seen = set()
    while pending:
        name = pending.pop().lower()
        if name in seen:
            continue
        seen.add(name)
        known = name in modifiers
        for ids in (env.objs_to_idxs, env.chars_to_idxs):
            if name in ids:
                touched |= env.obj_vecs[ids[name]]
                known = True
        if name in env.tree.legend:
            pending.update(env.tree.legend[name].obj_names)
            known = True
        if name in env.char_to_obj:
            pending.add(env.char_to_obj[name])
            known = True
        if name in env.meta_objs_lower:
            pending.update(env.meta_objs_lower[name])
            known = True
        if not known:
            return np.ones(env.n_objs, dtype=bool)
    layers = env.layer_masks
    return layers[layers[:, touched].any(axis=1)].any(axis=0)


def _motion_only(env, rules):
    """Recognize rules that only change forces on matched, individual objects."""
    if env.has_randomness():
        return None
    directions = {"<", ">", "^", "v", "up", "down", "left", "right"}
    prefixes = {"left", "right", "up", "down", "horizontal", "vertical", "+"}
    atomic = {
        name: i
        for i, name in enumerate(env.atomic_obj_names)
        if name not in env.joint_tiles
        and name not in env.meta_objs_lower
        and env.objs_to_idxs.get(name) == i
        and env.chars_to_idxs.get(name, i) == i
        and env.char_to_obj.get(name, name) == name
    }
    moving = np.zeros(env.n_objs, dtype=bool)
    moving[env.player_idxs] = True
    action_is_noop = True

    def parse_cell(cell):
        if len(cell) == 1 and isinstance(cell[0], str):
            words = cell[0].lower().split()
            if len(words) == 1 or (len(words) == 2 and words[0] in directions):
                return atomic.get(words[-1]), len(words) == 2
        return None, False

    for rule in rules:
        if (
            rule.command
            or len(rule.left_kernels or ()) != 1
            or len(rule.right_kernels or ()) != 1
            or not set(rule.prefixes) <= prefixes
        ):
            return None
        left, right = rule.left_kernels[0], rule.right_kernels[0]
        if len(left) != len(right):
            return None
        requires_direction = False
        for before, after in zip(left, right):
            obj, directed = parse_cell(before)
            if obj is None or obj != parse_cell(after)[0]:
                return None
            moving[obj] = True
            requires_direction |= directed
        action_is_noop &= requires_direction
    return moving, action_is_noop


def _mutable_cells(env, initial):
    rules = list(_rules(env.tree.rules))
    motion = _motion_only(env, rules)
    # Forces are shared by an entire layer. Overlapping initial occupants
    # prevent identifying the moved object from the matched rule alone.
    if motion is None or any(
        (initial[layer].sum(axis=0) > 1).any() for layer in env.layer_masks
    ):
        planes = _mutable_layers(env, rules)
        return np.broadcast_to(planes[:, None, None], initial.shape), 0, False

    moving, action_is_noop = motion
    cells = np.zeros_like(initial)
    for layer in env.layer_masks:
        active = np.flatnonzero(layer & moving)
        if active.size:
            # A moving object can only visit components containing one of its
            # initial positions. Other movable objects may vacate their cells.
            free = ~initial[layer & ~moving].any(axis=0)
            _, regions = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
            for obj in active:
                cells[obj] = np.isin(regions, np.unique(regions[initial[obj]]))
    return cells, int(np.count_nonzero(initial[cells])), action_is_noop


def build_compact_state(env, template, default_key, render):
    level = template.multihot_level
    _, height, width = level.shape
    initial = np.asarray(level)
    mutable, piece_count, action_is_noop = _mutable_cells(env, initial)
    cells = np.flatnonzero(mutable)
    size = cells.size
    selection = (
        slice(int(cells[0]), int(cells[-1]) + 1)
        if size and cells[-1] - cells[0] + 1 == size
        else cells
    )
    flat_level = level.ravel()
    random_rules = env.has_randomness()
    view_h, view_w = env._get_view_shape((height, width))
    scrolling = (view_h, view_w) != (height, width)
    coord_dtype = np.min_scalar_type(max(height, width) - 1)

    # Number each mutable (object, row, column) slot. A conserved set of
    # pieces can store occupied slot indices and use their high bit for win.
    position_flag = 1 << max(size - 1, 0).bit_length()
    position_dtype = np.min_scalar_type(2 * position_flag - 1)
    store_positions = (
        0 < piece_count
        and piece_count * position_dtype.itemsize < packed_num_bytes(size + 1, 1)
    )
    if store_positions:
        storage = FieldDescriptor.tensor(dtype=position_dtype, shape=(piece_count,))
        flat_level = jnp.asarray((initial & ~mutable).ravel())
        win_mask = position_flag
    else:
        storage = FieldDescriptor.packed_tensor(shape=(size + 1,), packed_bits=1)
        win_mask = 1 << (size % 8)

    class State(Xtructurable):
        payload: Annotated[jax.Array, storage]
        if random_rules:
            rng_key: Annotated[
                jax.Array,
                FieldDescriptor.tensor(dtype=jnp.uint32, shape=default_key.shape),
            ]
        if scrolling:
            origin: Annotated[
                jax.Array, FieldDescriptor.tensor(dtype=coord_dtype, shape=(2,))
            ]
        if TYPE_CHECKING:

            @property
            def payload_unpacked(self) -> jax.Array: ...

        @classmethod
        def from_engine(cls, state, won):
            occupied = state.multihot_level.ravel()[selection]
            if store_positions:
                # top_k breaks ties by ascending index, so equal boards
                # always produce the same key for the hash table.
                _, positions = jax.lax.top_k(occupied.astype(jnp.int32), piece_count)
                payload = positions.astype(position_dtype)
                payload = payload.at[-1].set(
                    payload[-1] | (won.astype(position_dtype) * win_mask)
                )
            else:
                payload = jnp.concatenate((occupied, won[None]))
            fields = {}
            if random_rules:
                fields["rng_key"] = state.rng
            if scrolling:
                fields["origin"] = state.view_bounds[::2].astype(coord_dtype)
            return cls.from_unpacked(payload=payload, **fields)

        @property
        def rng(self):
            if random_rules:
                return self.rng_key
            return jnp.broadcast_to(default_key, self.shape + default_key.shape)

        @property
        def won(self):
            return (self.payload[..., -1] & win_mask) != 0

        @property
        def board(self):
            if store_positions:

                def restore(positions):
                    indices = jnp.asarray(cells)[
                        positions.astype(jnp.int32) & (win_mask - 1)
                    ]
                    return flat_level.at[indices].set(True)

                positions = self.payload.reshape((-1, piece_count))
                return jax.vmap(restore)(positions).reshape(self.shape + level.shape)
            values = self.payload_unpacked[..., :size]
            if size == level.size:
                return values.reshape(self.shape + level.shape)
            base = jnp.broadcast_to(flat_level, self.shape + flat_level.shape)
            board = base.at[..., selection].set(
                values, indices_are_sorted=True, unique_indices=True
            )
            return board.reshape(self.shape + level.shape)

        @property
        def view_bounds(self):
            if not scrolling:
                return jnp.broadcast_to(template.view_bounds, self.shape + (4,))
            row, col = jnp.moveaxis(self.origin.astype(jnp.int32), -1, 0)
            return jnp.stack(
                (
                    row,
                    jnp.minimum(row + view_h, height),
                    col,
                    jnp.minimum(col + view_w, width),
                ),
                axis=-1,
            )

        def __str__(self, **kwargs):
            return render(self, **kwargs)

    return State, action_is_noop
