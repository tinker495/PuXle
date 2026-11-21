from pathlib import Path

import jax
import pytest

from puxle.puzzles.adapters.pddl import PDDL

# Simple-move defaults for non-param tests
DATA_DIR = Path(__file__).resolve().parents[1] / "pddl_data" / "simple_move"
DOMAIN = DATA_DIR / "domain.pddl"
PROBLEM = DATA_DIR / "problem.pddl"


@pytest.fixture
def puzzle():
    return PDDL(str(DOMAIN), str(PROBLEM))


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)
