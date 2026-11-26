#!/usr/bin/env python3
"""AST-based seam-leakage guard for standalone PuXle CI."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SANCTIONED_XTRUCTURE_SEAMS = frozenset(
    {
        "xtructure",
        "xtructure.numpy",
        "xtructure.xtructure_numpy",
        "xtructure.io",
        "xtructure.core.layout",
        "xtructure.core.display",
        "xtructure.core.packing",
    }
)

UNAPPROVED_XTRUCTURE_CHILDREN = frozenset(
    {"core", "hashtable", "bgpq", "queue", "stack"}
)

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    ".git",
    "build",
    "dist",
    "runs",
    "docs",
    "tests",
    "scripts",
}


@dataclass(frozen=True)
class SeamViolation:
    path: str
    lineno: int
    kind: str
    raw_import: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "lineno": self.lineno,
            "kind": self.kind,
            "raw_import": self.raw_import,
            "reason": self.reason,
        }


def iter_source_files(root: Path) -> list[Path]:
    base = root / "puxle"
    candidates: Iterable[Path] = base.rglob("*.py") if base.is_dir() else ()
    files: list[Path] = []
    for path in candidates:
        rel = path.relative_to(root)
        if any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in rel.parts
        ):
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _is_private_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _has_private_segment(dotted_name: str) -> bool:
    return any(_is_private_name(part) for part in dotted_name.split(".") if part)


def check_file_for_leaks(root: Path, path: Path) -> list[SeamViolation]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    rel_path = path.relative_to(root).as_posix()
    violations: list[SeamViolation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raw = (ast.get_source_segment(source, node) or "").strip()
            for alias in node.names:
                if alias.name.split(".")[0] != "xtructure":
                    continue
                if _has_private_segment(alias.name):
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="private-cross-repo-import",
                            raw_import=raw,
                            reason=(
                                f"Private cross-repo import `{alias.name}`; "
                                "deepen or expose the public seam in `Xtructure`."
                            ),
                        )
                    )
                elif alias.name not in SANCTIONED_XTRUCTURE_SEAMS:
                    violations.append(
                        SeamViolation(
                            path=rel_path,
                            lineno=node.lineno,
                            kind="internal-cross-repo-seam",
                            raw_import=raw,
                            reason=(
                                f"Internal cross-repo import `{alias.name}` from `Xtructure`; "
                                "use a sanctioned seam from CONTEXT.md."
                            ),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            mod = node.module
            if mod.split(".")[0] != "xtructure":
                continue
            raw = (ast.get_source_segment(source, node) or "").strip()
            if _has_private_segment(mod) or any(
                _is_private_name(alias.name)
                for alias in node.names
                if alias.name != "*"
            ):
                violations.append(
                    SeamViolation(
                        path=rel_path,
                        lineno=node.lineno,
                        kind="private-cross-repo-import",
                        raw_import=raw,
                        reason="Private symbol or module imported from `Xtructure`.",
                    )
                )
                continue
            imports_internal_child = mod == "xtructure" and any(
                alias.name in UNAPPROVED_XTRUCTURE_CHILDREN for alias in node.names
            )
            if mod not in SANCTIONED_XTRUCTURE_SEAMS or imports_internal_child:
                violations.append(
                    SeamViolation(
                        path=rel_path,
                        lineno=node.lineno,
                        kind="internal-cross-repo-seam",
                        raw_import=raw,
                        reason=(
                            f"Unsanctioned internal seam `{mod}` imported from `Xtructure`; "
                            "use a sanctioned seam or document the adapter in CONTEXT.md."
                        ),
                    )
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=".", help="Repository root (default: current directory)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = iter_source_files(root)
    violations: list[SeamViolation] = []
    for path in files:
        violations.extend(check_file_for_leaks(root, path))

    if args.json:
        print(
            json.dumps(
                {
                    "failed": bool(violations),
                    "files_checked": len(files),
                    "violation_count": len(violations),
                    "violations": [v.to_dict() for v in violations],
                },
                indent=2,
            )
        )
    else:
        print(f"Files checked: {len(files)}, Seam violations: {len(violations)}")
        for v in violations:
            print(f"  {v.path}:{v.lineno} [{v.kind}] {v.raw_import}\n    -> {v.reason}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
