from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from tqdm.auto import tqdm

from puxle.world_model import world_model_data_path

from ._puzzles import PuzzleName, _normalized, create_puzzle, data_name_for_puzzle
from ._world_model_render import render_states, save_transition_samples


@dataclass
class DatasetOptions:
    """Options shared by world-model dataset generators."""

    puzzle: PuzzleName
    puzzle_args: str = "{}"
    data_name: str | None = None
    dataset_size: int = 300_000
    dataset_minibatch_size: int = 30_000
    shuffle_length: int = 30
    image_size: tuple[int, int] = (32, 32)
    seed: int = 0
    output_dir: Path | None = None


DatasetOptionsArg = Annotated[DatasetOptions, tyro.conf.OmitArgPrefixes]


@dataclass
class TrainOptions:
    """Options for world-model training."""

    model: str
    data_name: str | None = None
    data_dir: Path | None = None
    epochs: int = 200
    mini_batch_size: int = 1_000
    optimizer: Literal["adam", "sgd"] = "adam"
    learning_rate: float = 3e-3
    lr_schedule: Literal["constant", "cosine"] = "cosine"
    """`cosine` decays the learning rate to zero over all training steps."""
    save_every: int = 100
    seed: int = 0
    """Seeds minibatch shuffling and, with --reset, parameter initialization."""
    reset: bool = False
    wm_loss_ratio: float = 0.5
    """Maximum world-model loss weight; the AE loss gets the remainder."""
    wm_onset_epoch: int = 20
    """Epoch at which the world-model loss weight starts ramping up."""
    wm_ramp_epochs: int = 1_000
    """Epochs from onset until the weight reaches --wm-loss-ratio."""


TrainOptionsArg = Annotated[TrainOptions, tyro.conf.OmitArgPrefixes]


def _dataset_context(options: DatasetOptions):
    with tqdm(
        total=1,
        desc=f"Initializing {options.puzzle}",
        unit="puzzle",
        dynamic_ncols=True,
    ) as progress:
        puzzle = create_puzzle(options.puzzle, options.puzzle_args)
        progress.update()
    data_name = options.data_name or data_name_for_puzzle(puzzle)
    root = options.output_dir or world_model_data_path(data_name)
    root.mkdir(parents=True, exist_ok=True)
    return puzzle, root


def _save_array(path: Path, array: np.ndarray) -> None:
    with tqdm(
        total=array.nbytes,
        desc=f"Saving {path.name}",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        np.save(path, array)
        progress.update(array.nbytes)


def make_transition_dataset(options: DatasetOptionsArg) -> int:
    """Generate transition images and actions."""
    from puxle.world_model import get_world_model_dataset_builder

    puzzle, root = _dataset_context(options)
    output = root / "transition"
    output.mkdir(parents=True, exist_ok=True)
    shuffle_parallel = math.ceil(
        options.dataset_minibatch_size / options.shuffle_length
    )
    with tqdm(
        total=options.dataset_size,
        desc="Generating transitions",
        unit="transition",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(
            shuffle=options.shuffle_length,
            batch=options.dataset_minibatch_size,
            phase="compile/execute",
        )
        build_dataset = get_world_model_dataset_builder(
            puzzle,
            options.dataset_size,
            shuffle_parallel,
            options.shuffle_length,
            options.dataset_minibatch_size,
            progress=progress.update,
        )
        states, actions, next_states = jax.block_until_ready(
            build_dataset(jax.random.PRNGKey(options.seed))
        )
    _save_array(output / "actions.npy", np.asarray(actions))
    _save_array(
        output / "images.npy",
        render_states(puzzle, states, options.image_size, "Rendering current states"),
    )
    _save_array(
        output / "next_images.npy",
        render_states(puzzle, next_states, options.image_size, "Rendering next states"),
    )
    save_transition_samples(
        output / "transition_samples.png",
        np.load(output / "images.npy", mmap_mode="r"),
        np.load(output / "next_images.npy", mmap_mode="r"),
        np.load(output / "actions.npy", mmap_mode="r"),
        options.seed,
    )
    print(output)
    return 0


def make_sample_data(options: DatasetOptionsArg) -> int:
    """Generate initial and target state images."""
    from puxle.world_model import get_sample_data_builder

    puzzle, output = _dataset_context(options)
    build_dataset = get_sample_data_builder(
        puzzle, options.dataset_size, options.dataset_minibatch_size
    )
    targets, initials = build_dataset(jax.random.PRNGKey(options.seed))
    _save_array(
        output / "targets.npy",
        render_states(puzzle, targets, options.image_size, "Rendering targets"),
    )
    _save_array(
        output / "inits.npy",
        render_states(puzzle, initials, options.image_size, "Rendering initial states"),
    )
    print(output)
    return 0


def make_eval_trajectory(options: DatasetOptionsArg) -> int:
    """Generate a trajectory used during evaluation."""
    from puxle.world_model import create_eval_trajectory

    puzzle, root = _dataset_context(options)
    output = root / "transition"
    output.mkdir(parents=True, exist_ok=True)
    with tqdm(
        total=options.dataset_size,
        desc="Generating evaluation trajectory",
        unit="move",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix_str("compiling/executing trajectory")
        states, actions = jax.block_until_ready(
            create_eval_trajectory(
                puzzle, options.dataset_size, jax.random.PRNGKey(options.seed)
            )
        )
        progress.update(len(actions))
    _save_array(output / "eval_actions.npy", np.asarray(actions))
    _save_array(
        output / "eval_traj_images.npy",
        render_states(
            puzzle, states, options.image_size, "Rendering evaluation states"
        ),
    )
    images = np.load(output / "eval_traj_images.npy", mmap_mode="r")
    save_transition_samples(
        output / "eval_transition_samples.png",
        images[:-1],
        images[1:],
        np.load(output / "eval_actions.npy", mmap_mode="r"),
        options.seed,
    )
    print(output)
    return 0


def _resolve_world_model_preset(name: str):
    from puxle import WorldModelPresets

    presets = WorldModelPresets.all()
    normalized = _normalized(name)
    for key, preset in presets.items():
        if _normalized(key) == normalized:
            return preset
    available = ", ".join(sorted(presets))
    raise ValueError(f"Unknown model '{name}'. Available: {available}")


def train(options: TrainOptionsArg) -> int:
    """Train a registered world model."""
    import optax

    from puxle import WorldModelPuzzle
    from puxle.world_model import (
        apply_with_conditional_batch_stats,
        world_model_eval_builder,
        world_model_train_builder,
    )

    preset = _resolve_world_model_preset(options.model)
    data_name = options.data_name or str(preset["data_name"])
    data_dir = options.data_dir or world_model_data_path(data_name) / "transition"
    with tqdm(
        (
            "images.npy",
            "next_images.npy",
            "actions.npy",
            "eval_traj_images.npy",
            "eval_actions.npy",
        ),
        desc="Loading dataset",
        unit="file",
        dynamic_ncols=True,
    ) as progress:
        arrays = []
        for filename in progress:
            progress.set_postfix_str(filename)
            arrays.append(np.load(data_dir / filename, mmap_mode="r"))
    datas, next_datas, actions, eval_states, eval_actions = arrays
    del arrays

    with tqdm(
        total=1, desc="Initializing model", unit="model", dynamic_ncols=True
    ) as progress:
        progress.set_postfix(model=options.model, seed=options.seed)
        model_overrides: dict[str, object] = {
            "data_name": data_name,
            "init_params": options.reset,
            "aqt_cfg": None,
        }
        if (
            options.data_dir is not None
            and (options.data_dir.parent / "inits.npy").exists()
            and (options.data_dir.parent / "targets.npy").exists()
        ):
            model_overrides["data_path"] = str(options.data_dir.parent)
        world_model = WorldModelPuzzle(**(dict(preset) | model_overrides))
        if options.reset:
            world_model.params = world_model.get_new_params(options.seed)
        jax.block_until_ready(world_model.params)
        progress.update()
    datas, next_datas, actions = [
        jax.block_until_ready(jnp.asarray(array))
        for array in tqdm(
            (datas, next_datas, actions),
            desc=f"Transferring training data to {jax.default_backend()}",
            unit="array",
            dynamic_ncols=True,
        )
    ]
    model = world_model.model

    def train_info_fn(params, data, next_data, action, training):
        if training:
            return apply_with_conditional_batch_stats(
                model.apply,
                params,
                data,
                next_data,
                action,
                training=True,
                method=model.train_info,
            )
        return model.apply(
            params,
            data,
            next_data,
            action,
            training=False,
            method=model.train_info,
        )

    learning_rate = (
        optax.cosine_decay_schedule(
            options.learning_rate,
            options.epochs * math.ceil(len(actions) / options.mini_batch_size),
        )
        if options.lr_schedule == "cosine"
        else options.learning_rate
    )
    optimizer = (
        optax.adam(learning_rate)
        if options.optimizer == "adam"
        else optax.sgd(learning_rate)
    )
    eval_fn = world_model_eval_builder(train_info_fn, options.mini_batch_size)
    params = world_model.params
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(options.seed)

    tqdm.write(
        f"dataset={len(actions):,} transitions on {jax.default_backend()} "
        f"({(datas.nbytes + next_datas.nbytes + actions.nbytes) / 2**30:.2f} GiB), "
        f"mini_batch_size={options.mini_batch_size:,}; shuffling indices only"
    )

    with (
        tqdm(
            total=options.epochs,
            desc="Training epochs",
            unit="epoch",
            dynamic_ncols=True,
        ) as epoch_progress,
        tqdm(
            total=math.ceil(len(actions) / options.mini_batch_size),
            desc="Preparing training",
            unit="batch",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ) as batch_progress,
    ):

        def report_training(completed: int, values: np.ndarray) -> None:
            if epoch == 0 and batch_progress.n == 0:
                epoch_progress.set_postfix(phase="training")
                batch_progress.set_description("Epoch 1 batches")
            batch_progress.set_postfix(
                loss=f"{float(values[0]):.5f}",
                ae=f"{float(values[1]):.5f}",
                wm=f"{float(values[2]):.5f}",
                exact=f"{float(values[3]):.1%}",
                refresh=False,
            )
            batch_progress.update(completed - batch_progress.n)

        train_fn = world_model_train_builder(
            options.mini_batch_size,
            train_info_fn,
            optimizer,
            loss_ratio=options.wm_loss_ratio,
            progress=report_training,
            wm_onset_epoch=options.wm_onset_epoch,
            wm_ramp_epochs=options.wm_ramp_epochs,
        )
        for epoch in range(options.epochs):
            batch_progress.reset()
            batch_progress.set_description(
                "Compiling first batch" if epoch == 0 else f"Epoch {epoch + 1} batches"
            )
            epoch_progress.set_postfix(
                phase="compiling first batch" if epoch == 0 else "training"
            )
            key, subkey = jax.random.split(key)
            params, opt_state, loss, ae_loss, wm_loss, accuracy, wm_weight = train_fn(
                subkey, (datas, next_datas, actions), params, opt_state, epoch
            )
            batch_progress.refresh()
            losses = dict(
                loss=f"{float(loss):.5f}",
                ae=f"{float(ae_loss):.5f}",
                wm=f"{float(wm_loss):.5f}",
                exact=f"{float(accuracy):.1%}",
            )
            if epoch % 10 == 0 or epoch + 1 == options.epochs:
                epoch_progress.set_postfix(phase="evaluation", **losses)
                eval_accuracy = eval_fn(params, (eval_states, eval_actions))
                losses["eval"] = f"{float(eval_accuracy):.1%}"
                tqdm.write(
                    f"epoch={epoch} loss={float(loss):.6f} ae={float(ae_loss):.6f} "
                    f"wm={float(wm_loss):.6f} wm_weight={float(wm_weight):.6f} "
                    f"accuracy={float(accuracy):.6f} eval={float(eval_accuracy):.6f}"
                )
            if options.save_every > 0 and (epoch + 1) % options.save_every == 0:
                epoch_progress.set_postfix(phase="saving checkpoint", **losses)
                world_model.params = params
                world_model.save_model()
            epoch_progress.set_postfix(phase="complete", **losses)
            epoch_progress.update()

        world_model.params = params
        epoch_progress.set_postfix(phase="saving checkpoint", **losses)
        world_model.save_model()

    print(world_model.path)
    return 0
