import subprocess
import sys
from pathlib import Path

import pytest
import tyro


def test_cli_exposes_human_play_and_world_model_training_commands():
    from puxle.cli import main

    for args in (
        ["human-play", "--help"],
        ["world-model-train", "make-transition-dataset", "--help"],
        ["world-model-train", "make-sample-data", "--help"],
        ["world-model-train", "make-eval-trajectory", "--help"],
        ["world-model-train", "train", "--help"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        assert exc_info.value.code == 0


def test_world_model_dataset_cli_preserves_options_and_defaults():
    from puxle.cli.world_model import DatasetOptions

    args = tyro.cli(
        DatasetOptions,
        args=[
            "--puzzle",
            "SlidePuzzle",
            "--puzzle-args",
            '{"size": 4}',
            "--dataset-size",
            "12",
            "--image-size",
            "64",
            "48",
            "--output-dir",
            "/tmp/puxle-data",
        ],
        console_outputs=False,
    )

    assert args.puzzle == "SlidePuzzle"
    assert args.puzzle_args == '{"size": 4}'
    assert args.data_name is None
    assert args.dataset_size == 12
    assert args.dataset_minibatch_size == 30_000
    assert args.shuffle_length == 30
    assert args.image_size == (64, 48)
    assert args.seed == 0
    assert args.output_dir == Path("/tmp/puxle-data")


def test_world_model_train_cli_preserves_options_and_defaults():
    from puxle.cli.world_model import TrainOptions

    args = tyro.cli(
        TrainOptions,
        args=[
            "--model",
            "RubiksCubeWorldModel",
            "--optimizer",
            "sgd",
            "--reset",
        ],
        console_outputs=False,
    )

    assert args.data_name is None
    assert args.data_dir is None
    assert args.epochs == 2_000
    assert args.mini_batch_size == 1_000
    assert args.optimizer == "sgd"
    assert args.learning_rate == 1e-4
    assert args.save_every == 100
    assert args.seed == 0
    assert args.reset is True


def test_visualize_cli_preserves_repeated_kwargs_and_boolean_pair():
    from scripts.visualize_puzzle import VisualizeOptions

    args = tyro.cli(
        VisualizeOptions,
        args=[
            "--puzzle",
            "SlidePuzzle",
            "--img",
            "--kwarg",
            "size=4",
            "--kwarg",
            "foo=bar",
        ],
        console_outputs=False,
    )

    assert args.seed == 42
    assert args.img is True
    assert args.output_dir == Path("images/visualizations")
    assert args.puzzle_kwargs == ["size=4", "foo=bar"]


def test_cli_help_is_runnable():
    from puxle.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0


def test_cli_help_lists_registered_puzzles(capsys):
    from puxle.cli import main

    with pytest.raises(SystemExit):
        main(["human-play", "--help"])

    output = capsys.readouterr().out
    assert "--puzzle {" in output
    assert "SlidePuzzle" in output


@pytest.mark.parametrize(
    "args",
    [
        ["human-play", "--puzzle", "slide-puzzle"],
        [
            "world-model-train",
            "make-transition-dataset",
            "--puzzle",
            "slide-puzzle",
        ],
    ],
)
def test_cli_rejects_unregistered_puzzle_names(args):
    from puxle.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(args)

    assert exc_info.value.code == 2


def test_cli_reports_invalid_puzzle_json_as_usage_error(capsys):
    from puxle.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["human-play", "--puzzle", "SlidePuzzle", "--puzzle-args", "{"])

    assert exc_info.value.code == 2
    assert "puxle: error:" in capsys.readouterr().err


def test_source_main_connects_to_package_cli():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "main.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "human-play" in result.stdout
    assert "world-model-train" in result.stdout


def test_visualize_puzzle_rejects_malformed_kwarg():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.visualize_puzzle",
            "--puzzle",
            "SlidePuzzle",
            "--kwarg",
            "missing-separator",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--kwarg must be provided as key=value" in result.stderr


def test_cli_creates_public_puzzle_by_class_name():
    from puxle.cli._puzzles import create_puzzle
    from puxle.puzzles.slidepuzzle import SlidePuzzle

    puzzle = create_puzzle("SlidePuzzle", '{"size": 2}')

    assert isinstance(puzzle, SlidePuzzle)
    assert puzzle.size == 2
