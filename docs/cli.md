# Command-line interface

PuXle owns the puzzle-only command-line workflows. Run them through the installed
`puxle` command or from a checkout with `python main.py`.

## Human play

```bash
puxle human-play --puzzle SlidePuzzle --puzzle-args '{"size": 4}' --seed 0
```

Choose actions with their displayed number or `WASD`; enter `q` to exit.

## World-model datasets

```bash
puxle world-model-train make-transition-dataset --puzzle RubiksCube
puxle world-model-train make-sample-data --puzzle RubiksCube
puxle world-model-train make-eval-trajectory --puzzle RubiksCube
```

By default, generated data and downloaded checkpoints are stored below
`$PUXLE_HOME/world_model`. If `PUXLE_HOME` is unset, PuXle uses
`$XDG_CACHE_HOME/puxle` or `~/.cache/puxle`.

Use `--output-dir` to override a dataset command's destination.

## World-model training

```bash
puxle world-model-train train \
  --model RubiksCubeWorldModel \
  --reset \
  --epochs 2000 \
  --mini-batch-size 1000
```

Available model names are the attributes exposed by
`puxle.world_model.trained_world_model_registry`.
