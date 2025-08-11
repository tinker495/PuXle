from dataclasses import dataclass
from pathlib import Path

@dataclass
class PDDLSpec:
    """Specification for a PDDL test fixture."""
    name: str
    domain: str
    problem: str
    expected_atoms: int
    expected_actions: int
    solvable: bool
    max_solution_steps: int = 10


# Comprehensive test data specifications
BASE = Path(__file__).resolve().parents[1] / "pddl_data"

DATA_SPECS = [
    PDDLSpec(
        name="simple-move-3",
        domain=str(BASE / "simple_move" / "domain.pddl"),
        problem=str(BASE / "simple_move" / "problem.pddl"),
        expected_atoms=12, expected_actions=9, solvable=True, max_solution_steps=3,
    ),
    PDDLSpec(
        name="toggle",
        domain=str(BASE / "toggle" / "domain.pddl"),
        problem=str(BASE / "toggle" / "problem.pddl"),
        expected_atoms=2, expected_actions=2, solvable=True, max_solution_steps=1,
    ),
    PDDLSpec(
        name="door-move",
        domain=str(BASE / "door_move" / "domain.pddl"),
        problem=str(BASE / "door_move" / "problem.pddl"),
        expected_atoms=22, expected_actions=18, solvable=True, max_solution_steps=2,
    ),
    PDDLSpec(
        name="typed-move",
        domain=str(BASE / "typed_move" / "domain.pddl"),
        problem=str(BASE / "typed_move" / "problem.pddl"),
        expected_atoms=15, expected_actions=4, solvable=True, max_solution_steps=2,
    ),
    PDDLSpec(
        name="unreachable-move",
        domain=str(BASE / "simple_move" / "domain.pddl"),
        problem=str(BASE / "unreachable_move" / "problem.pddl"),
        expected_atoms=12, expected_actions=9, solvable=False, max_solution_steps=10,
    ),
]
