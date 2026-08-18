import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_exposes_human_play_and_world_model_training_commands():
    from puxle.cli import build_parser

    parser = build_parser()

    human_play = parser.parse_args(["human-play", "--puzzle", "SlidePuzzle"])
    dataset = parser.parse_args(
        [
            "world-model-train",
            "make-transition-dataset",
            "--puzzle",
            "RubiksCube",
        ]
    )
    training = parser.parse_args(
        ["world-model-train", "train", "--model", "RubiksCubeWorldModel"]
    )

    assert human_play.command == "human-play"
    assert dataset.world_model_command == "make-transition-dataset"
    assert training.world_model_command == "train"


def test_cli_help_is_runnable():
    from puxle.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0


def test_source_main_connects_to_package_cli():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "main.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "human-play" in result.stdout
    assert "world-model-train" in result.stdout


def test_cli_creates_public_puzzle_by_class_name():
    from puxle.cli._puzzles import create_puzzle
    from puxle.puzzles.slidepuzzle import SlidePuzzle

    puzzle = create_puzzle("SlidePuzzle", '{"size": 2}')

    assert isinstance(puzzle, SlidePuzzle)
    assert puzzle.size == 2
