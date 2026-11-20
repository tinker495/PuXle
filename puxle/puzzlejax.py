"""Convert PuzzleScript games into PuXle puzzles using the PuzzleJAX engine."""

from importlib import import_module
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
from xtructure import FieldDescriptor, Xtructurable

from puxle.core.puzzle_base import Puzzle


def _load_game(game: str | Path):
    from lark import Lark
    from lark.exceptions import LarkError

    paths = import_module("puzzlescript_jax.globals")
    preprocessing = import_module("puzzlescript_jax.preprocessing")
    gen_tree = import_module("puzzlescript_jax.gen_tree")
    path = Path(game)
    if isinstance(game, str) and path.name == game and not path.is_file():
        for directory in (paths.CUSTOM_GAMES_DIR, paths.GAMES_DIR):
            candidate = Path(directory, game + ".txt")
            if candidate.is_file():
                path = candidate
                break
    source = path.read_text(encoding="utf-8-sig")
    parser = Lark.open(
        paths.LARK_SYNTAX_PATH, start="ps_game", maybe_placeholders=False
    )
    try:
        parsed = parser.parse(preprocessing.preprocess_ps(source))
        # The corpus preprocessor normally drops malformed rules. A search puzzle
        # must keep the supplied rules or report the line it cannot compile.
        for broken in parsed.find_data("rule_data_broken"):
            raise ValueError(f"Unsupported PuzzleScript rule: {broken.children[0]}")
        stripped = preprocessing.StripPuzzleScript().transform(parsed)
        return gen_tree.GenPSTree().transform(stripped)
    except (LarkError, ValueError) as exc:
        raise ValueError(f"Cannot convert PuzzleScript file {path}: {exc}") from exc


class PuzzleJAX(Puzzle):
    """One PuzzleScript level compiled with PuzzleJAX for forward search.

    ``game`` accepts a PuzzleJAX catalog name, a local PuzzleScript path, a
    parsed PSGameTree, or a PuzzleJaxEnv. Levels are zero-based. An existing
    single-level environment defaults to its selected level.
    """

    def __init__(self, game="sokoban_basic", level: int | None = None):
        try:
            # PuzzleJAX uses PIL.Image through the parent package at startup.
            import_module("PIL.Image")
            backend = import_module("puzzlescript_jax.env")
            if isinstance(game, (str, Path)):
                game = _load_game(game)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"PuzzleJAX requires {exc.name!r}. Install puxle[puzzlejax] and put "
                "the PuzzleJAX (smearle/script-doctor) checkout on PYTHONPATH."
            ) from exc

        if isinstance(game, backend.PuzzleJaxEnv):
            if level is None and game.level_i >= 0:
                level = game.level_i
            game = game.tree
        if not isinstance(game, backend.PSGameTree):
            raise TypeError(
                "Expected a game name, PuzzleScript path, PSGameTree or PuzzleJaxEnv"
            )
        level = 0 if level is None else level
        if type(level) is not int or not 0 <= level < len(game.levels):
            raise ValueError(
                f"Invalid level {level!r} for a game with {len(game.levels)} levels"
            )
        self.env = backend.PuzzleJaxEnv(
            game, level_i=level, jit=True, print_score=False
        )
        self.action_size = self.env.action_space.n
        self.params = backend.PJParams(level=self.env.get_level(level), level_i=level)
        self._has_random_rules = self.env.has_randomness()
        self._default_key = jax.random.PRNGKey(0)
        _, self._template = self.env.reset(self._default_key, self.params)
        super().__init__()

    def define_state_class(self):
        template = self._template
        render = self.get_string_parser()

        class State(Xtructurable):
            board: Annotated[
                jax.Array,
                FieldDescriptor.tensor(
                    dtype=jnp.bool_, shape=template.multihot_level.shape
                ),
            ]
            won: Annotated[jax.Array, FieldDescriptor.scalar(dtype=jnp.bool_)]
            rng: Annotated[
                jax.Array,
                FieldDescriptor.tensor(dtype=jnp.uint32, shape=template.rng.shape),
            ]
            view_bounds: Annotated[
                jax.Array,
                FieldDescriptor.tensor(
                    dtype=jnp.int32, shape=template.view_bounds.shape
                ),
            ]

            def __str__(self, **kwargs):
                return render(self, **kwargs)

        return State

    def define_goal_spec_class(self):
        class GoalSpec(Xtructurable):
            pass

        return GoalSpec

    def get_solve_config(self, key=None, data=None):
        return self.SolveConfig.default()

    def _search_state(self, state, won):
        # Reward, heuristic and elapsed step counters do not affect the rules.
        # Deterministic boards reached by different paths must compare equal.
        return self.State(
            board=state.multihot_level,
            won=won,
            rng=state.rng if self._has_random_rules else self._default_key,
            view_bounds=state.view_bounds,
        )

    def _engine_state(self, state):
        return self._template.replace(
            multihot_level=state.board,
            win=state.won,
            rng=state.rng,
            view_bounds=state.view_bounds,
        )

    def get_initial_state(self, solve_config, key=None, data=None):
        state = self._template
        if self._has_random_rules and key is not None:
            _, state = self.env.reset(jax.random.key_data(key), self.params)
        won, _, _ = self.env.check_win(state.multihot_level)
        return self._search_state(state, won | state.win)

    def _apply(self, solve_config, state, action):
        # step_env preserves terminal boards. Gym-style step auto-resets on win.
        _, result, _, _, _ = self.env.step_env(
            state.rng, self._engine_state(state), action, self.params
        )
        successor = self._search_state(result, result.win)
        return successor, jnp.where((successor != state) & ~state.won, 1.0, jnp.inf)

    def is_solved(self, solve_config, state):
        return state.won

    def action_to_string(self, action: int) -> str:
        if not 0 <= action < self.action_size:
            raise ValueError(f"Invalid PuzzleJAX action: {action}")
        return ("←", "↓", "→", "↑", "Action")[action]

    def get_string_parser(self):
        describe = import_module("puzzlescript_jax.env_utils").multihot_to_desc
        visible = jax.jit(self.env.get_visible_multihot_level)
        names = {
            name: self.env.objs_to_idxs[name] for name in self.env.atomic_obj_names
        }

        def render(state, **kwargs):
            board = visible(level=state.board, view_bounds=state.view_bounds)
            return describe(
                jax.device_get(board),
                names,
                self.env.n_objs,
                self.env.obj_idxs_to_force_idxs,
                show_forces=False,
            )

        return render

    def get_img_parser(self):
        def render(state, **kwargs):
            rgba = self.env.render(self._engine_state(state), False)
            return np.asarray(rgba)[..., :3]

        return render

    def get_solve_config_string_parser(self):
        return lambda config, **kwargs: f"{self.env.title}, level {self.env.level_i}"

    def get_solve_config_img_parser(self):
        def render(config, **kwargs):
            raise NotImplementedError(
                "A PuzzleScript win condition has no single goal image"
            )

        return render
