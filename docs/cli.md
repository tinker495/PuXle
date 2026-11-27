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
  --model RUBIKSCUBE \
  --reset
```

Available model names are the preset keys exposed by
`puxle.WorldModelPresets.all()`.
