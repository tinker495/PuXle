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


def test_visualize_puzzle_uses_bound_img_and_sanitizes_filenames(tmp_path):
    """Author: tinker495.

    Why: Regression test for ``scripts.visualize_puzzle`` verifying that
    ``solve_config``-dependent puzzles (``Maze``) render via bound
    ``state.img(solve_config=...)`` and ANSI-colored actions (``PDDL``) produce
    sanitized output filenames.
    """
    from scripts.visualize_puzzle import visualize_puzzle

    visualize_puzzle(
        puzzle_name="Maze",
        seed=0,
        img=True,
        output_dir=tmp_path,
        puzzle_kwargs=["size=7"],
    )
    maze_files = sorted(p.name for p in (tmp_path / "Maze").iterdir())
    assert "initial.png" in maze_files
    assert len(maze_files) == 5

    visualize_puzzle(
        puzzle_name="PDDL",
        seed=0,
        img=True,
        output_dir=tmp_path,
        puzzle_kwargs=["domain='blocksworld'", "problem='bw-S-01'"],
    )
    pddl_files = sorted(p.name for p in (tmp_path / "PDDL").iterdir())
    assert "initial.png" in pddl_files
    assert all("\x1b" not in name for name in pddl_files)
