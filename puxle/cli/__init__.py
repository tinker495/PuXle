from __future__ import annotations

import json
import sys
from collections.abc import Sequence

import tyro

from . import human_play, world_model


def build_app() -> tyro.extras.SubcommandApp:
    app = tyro.extras.SubcommandApp()
    app.command(name="human-play")(human_play.run)

    world_model_app = tyro.extras.SubcommandApp()
    world_model_app.command(name="make-transition-dataset")(
        world_model.make_transition_dataset
    )
    world_model_app.command(name="make-sample-data")(world_model.make_sample_data)
    world_model_app.command(name="make-eval-trajectory")(
        world_model.make_eval_trajectory
    )
    world_model_app.command(name="train")(world_model.train)
    app.command(
        world_model_app,
        name="world-model-train",
        help="Generate datasets and train world models.",
    )
    return app


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return build_app().cli(
            prog="puxle",
            description="PuXle puzzle utilities.",
            args=argv,
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"puxle: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


__all__ = ["build_app", "main"]
