"""Batch rendering for world-model datasets using native geometry and sprites."""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import cv2
import jax
import numpy as np
from tqdm.auto import tqdm
from xtructure import Xtructurable

from puxle.core.puzzle_base import Puzzle
from puxle.puzzles.rubikscube import RubiksCube, rgb_map
from puxle.puzzles.sokoban import Sokoban
from puxle.utils.util import IMG_SIZE


def save_transition_samples(
    path: Path,
    images: np.ndarray,
    next_images: np.ndarray,
    actions: np.ndarray,
    seed: int,
) -> None:
    """Save up to three distinct, seeded transition samples without copying the dataset."""
    if (
        actions.ndim != 1
        or len(actions) == 0
        or images.ndim != 4
        or images.shape[-1] != 3
        or min(images.shape[1:3]) <= 0
        or images.shape != next_images.shape
        or len(images) != len(actions)
    ):
        raise ValueError("Preview requires paired RGB images and one action per image.")
    indices = np.random.default_rng(seed % 2**32).choice(
        len(actions), size=min(3, len(actions)), replace=False
    )
    scale = min(320 / images.shape[2], 240 / images.shape[1])
    width = max(1, round(images.shape[2] * scale))
    height = max(1, round(images.shape[1] * scale))
    row_height = height + 64
    plot = np.full((40 + len(indices) * row_height, 2 * width + 208, 3), 255, np.uint8)
    for title, x in (
        ("State", 16),
        ("Action", width + 32),
        ("Next state", width + 176),
    ):
        cv2.putText(
            plot,
            title,
            (x, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    for row, index in enumerate(
        tqdm(
            indices,
            desc="Plotting transition samples",
            unit="sample",
            dynamic_ncols=True,
        )
    ):
        y = 40 + row * row_height
        cv2.putText(
            plot,
            f"Sample {row + 1} (index {index})",
            (16, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        for batch, x in ((images, 16), (next_images, width + 176)):
            plot[y + 32 : y + 32 + height, x : x + width] = cv2.resize(
                batch[index], (width, height), interpolation=cv2.INTER_NEAREST
            )
        cv2.putText(
            plot,
            f"action {int(actions[index])}",
            (width + 32, y + 24 + height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.arrowedLine(
            plot,
            (width + 40, y + 48 + height // 2),
            (width + 144, y + 48 + height // 2),
            (0, 0, 0),
            2,
            tipLength=0.15,
        )
    if not cv2.imwrite(str(path), cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save transition samples to {path}")
    tqdm.write(f"Transition samples: {path}")


@lru_cache(maxsize=8)
def _rubik_renderer(
    puzzle: RubiksCube, image_size: tuple[int, int]
) -> Callable[[np.ndarray], np.ndarray]:
    native = puzzle.get_img_parser()
    shape = (6, puzzle.size * puzzle.size)
    baseline = native(puzzle.State.from_unpacked(faces=np.zeros(shape, np.uint8)))
    masks = []
    for tile in range(6 * puzzle.size * puzzle.size):
        faces = np.zeros(shape, np.uint8)
        faces.flat[tile] = 1
        masks.append(
            np.any(native(puzzle.State.from_unpacked(faces=faces)) != baseline, axis=-1)
        )
    tile_masks = np.stack(masks)
    if np.any(tile_masks.sum(axis=0) > 1):
        raise ValueError("Native Rubik renderer produced overlapping sticker masks.")
    width, height = image_size
    factor_y, factor_x = baseline.shape[0] // height, baseline.shape[1] // width
    counts = (
        tile_masks.reshape(-1, height, factor_y, width, factor_x)
        .sum(axis=(2, 4), dtype=np.int32)
        .reshape(len(tile_masks), -1)
        .T
    )
    support = int(np.max(np.count_nonzero(counts, axis=-1)))
    indices = np.argsort(counts, axis=-1)[:, -support:]
    weights = np.take_along_axis(counts, indices, axis=-1)
    baseline[np.any(tile_masks, axis=0)] = 0
    fixed = (
        baseline.reshape(height, factor_y, width, factor_x, 3)
        .sum(axis=(1, 3), dtype=np.int32)
        .reshape(-1, 3)
    )
    palette = np.asarray([rgb_map[code] for code in range(6)], dtype=np.int32)
    scale = np.float32(1.0 / (factor_y * factor_x))

    def render(rows: np.ndarray) -> np.ndarray:
        colors = palette[rows]
        sums = np.broadcast_to(fixed, (len(rows), *fixed.shape)).copy()
        for slot in range(support):
            sums += colors[:, indices[:, slot], :] * weights[:, slot][None, :, None]
        # OpenCV integer-ratio INTER_AREA rounds the scaled integer pixel sum.
        return (
            np.rint(sums.astype(np.float32) * scale)
            .astype(np.uint8)
            .reshape(len(rows), height, width, 3)
        )

    return render


@lru_cache(maxsize=8)
def _sokoban_renderer(
    puzzle: Sokoban, image_size: tuple[int, int]
) -> Callable[[np.ndarray], np.ndarray]:
    native = puzzle.get_img_parser()
    width, height = image_size
    cell_h, cell_w = height // puzzle.size, width // puzzle.size
    tiles = np.stack(
        [
            cv2.resize(
                native(
                    puzzle.State.from_unpacked(
                        board=np.full(puzzle.size**2, code, np.uint8)
                    )
                ),
                image_size,
                interpolation=cv2.INTER_AREA,
            )[:cell_h, :cell_w]
            for code in range(4)
        ]
    )

    def render(rows: np.ndarray) -> np.ndarray:
        return (
            tiles[rows]
            .reshape(-1, puzzle.size, puzzle.size, cell_h, cell_w, 3)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(-1, height, width, 3)
        )

    return render


def render_states(
    puzzle: Puzzle,
    states: Xtructurable,
    image_size: tuple[int, int],
    desc: str,
) -> np.ndarray:
    width, height = image_size
    if min(width, height) <= 0 or len(states) == 0:
        raise ValueError(
            "Rendering requires positive image dimensions and nonempty states."
        )
    if (
        isinstance(puzzle, RubiksCube)
        and puzzle.color_embedding
        and (2 * IMG_SIZE[0]) % width == 0
        and IMG_SIZE[0] % height == 0
    ):
        with tqdm(
            total=1,
            desc="Preparing Rubik batch renderer",
            unit="renderer",
            dynamic_ncols=True,
        ) as progress:
            render = _rubik_renderer(puzzle, image_size)
            values = np.asarray(
                jax.jit(lambda batch: batch.faces_unpacked)(states)
            ).reshape(len(states), -1)
            progress.update()
    elif (
        isinstance(puzzle, Sokoban)
        and width % puzzle.size == 0
        and height % puzzle.size == 0
        and IMG_SIZE[0] % width == 0
        and IMG_SIZE[1] % height == 0
    ):
        with tqdm(
            total=1,
            desc="Preparing Sokoban batch renderer",
            unit="renderer",
            dynamic_ncols=True,
        ) as progress:
            render = _sokoban_renderer(puzzle, image_size)
            values = np.asarray(
                jax.jit(lambda batch: batch.board_unpacked)(states)
            ).reshape(len(states), -1)
            progress.update()
    else:
        native = puzzle.get_img_parser()
        return np.stack(
            [
                cv2.resize(
                    np.asarray(native(state)), image_size, interpolation=cv2.INTER_AREA
                )
                for state in tqdm(
                    states,
                    total=len(states),
                    desc=desc,
                    unit="image",
                    dynamic_ncols=True,
                )
            ]
        )

    images = np.empty((len(states), height, width, 3), dtype=np.uint8)
    batch_size = max(1, min(1024, 2_097_152 // (width * height)))
    with tqdm(
        total=len(states), desc=desc, unit="image", dynamic_ncols=True
    ) as progress:
        for start in range(0, len(states), batch_size):
            images[start : start + batch_size] = render(
                values[start : start + batch_size]
            )
            progress.update(min(batch_size, len(states) - start))
    return images
