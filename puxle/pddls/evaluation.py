"""Utility routines for validating and evaluating generated PDDL tasks."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import jax
import numpy as np

from puxle.pddls.pddl import PDDL


def _state_key(state) -> bytes:
    return np.asarray(state.atoms).tobytes()


@dataclass
class ValidationResult:
    passed: bool
    explored_states: int
    depth_limit: int


def bfs_validate(domain_path: str | Path, problem_path: str | Path, depth_limit: int = 6) -> ValidationResult:
    """Run a bounded BFS to check reachability of the goal state."""
    env = PDDL(str(domain_path), str(problem_path))
    rng = jax.random.PRNGKey(0)
    solve_config, initial_state = env.get_inits(rng)

    if bool(env.is_solved(solve_config, initial_state)):
        return ValidationResult(True, 1, depth_limit)

    from collections import deque

    queue = deque([(initial_state, 0)])
    visited = {_state_key(initial_state)}
    explored = 0

    while queue:
        state, depth = queue.popleft()
        explored += 1
        if depth >= depth_limit:
            continue
        neighbours, costs = env.get_neighbours(solve_config, state, filled=True)
        costs_np = np.array(costs)
        applicable = np.where(np.isfinite(costs_np))[0]
        if applicable.size == 0:
            continue
        for idx in applicable:
            next_state = jax.tree_util.tree_map(lambda x: x[idx], neighbours)
            key = _state_key(next_state)
            if key in visited:
                continue
            if bool(env.is_solved(solve_config, next_state)):
                return ValidationResult(True, explored + 1, depth_limit)
            visited.add(key)
            queue.append((next_state, depth + 1))

    return ValidationResult(False, explored, depth_limit)


@dataclass
class PlannerRunResult:
    planner: str
    command: List[str]
    returncode: Optional[int]
    stdout: str
    stderr: str
    runtime_seconds: Optional[float]
    solved: Optional[bool]
    error: Optional[str]
    available: bool


def _build_result(
    planner: str,
    command: Sequence[str],
    *,
    returncode: Optional[int] = None,
    stdout: str = "",
    stderr: str = "",
    runtime: Optional[float] = None,
    solved: Optional[bool] = None,
    error: Optional[str] = None,
    available: bool = True,
) -> PlannerRunResult:
    return PlannerRunResult(
        planner=planner,
        command=list(command),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        runtime_seconds=runtime,
        solved=solved,
        error=error,
        available=available,
    )


def run_fast_downward(
    domain_path: str | Path,
    problem_path: str | Path,
    *,
    fd_binary: str = "fast-downward",
    search: str = "astar(lmcut())",
    plan_filename: Optional[str] = None,
    extra_args: Optional[Iterable[str]] = None,
    timeout_seconds: Optional[float] = None,
) -> PlannerRunResult:
    """Execute Fast Downward if available and capture the result."""

    binary_path = shutil.which(fd_binary)
    command_prefix: List[str] = [fd_binary]
    if binary_path is None:
        return _build_result(
            "fast-downward",
            command_prefix,
            error="fast-downward binary not found on PATH",
            available=False,
        )

    domain_path = str(domain_path)
    problem_path = str(problem_path)

    plan_file_context = None
    plan_target: Path

    if plan_filename is None:
        plan_file_context = tempfile.NamedTemporaryFile(prefix="fd-plan-", suffix=".txt", delete=False)
        plan_target = Path(plan_file_context.name)
        plan_file_context.close()
    else:
        plan_target = Path(plan_filename)

    command: List[str] = [fd_binary, "--plan-file", str(plan_target)]
    if extra_args:
        command.extend(extra_args)
    command.extend([domain_path, problem_path, "--search", search])

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        runtime = time.perf_counter() - start
        solved = completed.returncode == 0
        return _build_result(
            "fast-downward",
            command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime=runtime,
            solved=solved,
        )
    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - start
        return _build_result(
            "fast-downward",
            command,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            runtime=runtime,
            solved=None,
            error=f"timeout after {timeout_seconds} seconds",
        )
    except FileNotFoundError:
        return _build_result(
            "fast-downward",
            command,
            error="fast-downward executable vanished during execution",
            available=False,
        )
    finally:
        if plan_filename is None and "plan_target" in locals():
            try:
                plan_target.unlink(missing_ok=True)
            except Exception:
                pass


def run_lpg(
    domain_path: str | Path,
    problem_path: str | Path,
    *,
    lpg_binary: str = "lpg",
    search_time_limit: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> PlannerRunResult:
    """Execute LPG if available and capture the result."""

    binary_path = shutil.which(lpg_binary)
    command_prefix: List[str] = [lpg_binary]
    if binary_path is None:
        return _build_result(
            "lpg",
            command_prefix,
            error="lpg binary not found on PATH",
            available=False,
        )

    domain_path = str(domain_path)
    problem_path = str(problem_path)

    command: List[str] = [lpg_binary, "-o", domain_path, "-f", problem_path]
    if search_time_limit is not None:
        command.extend(["-n", str(search_time_limit)])
    if extra_args:
        command.extend(extra_args)

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        runtime = time.perf_counter() - start
        solved = completed.returncode == 0
        return _build_result(
            "lpg",
            command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime=runtime,
            solved=solved,
        )
    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - start
        return _build_result(
            "lpg",
            command,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            runtime=runtime,
            solved=None,
            error=f"timeout after {timeout_seconds} seconds",
        )
    except FileNotFoundError:
        return _build_result(
            "lpg",
            command,
            error="lpg executable vanished during execution",
            available=False,
        )


__all__ = [
    "ValidationResult",
    "PlannerRunResult",
    "bfs_validate",
    "run_fast_downward",
    "run_lpg",
]

