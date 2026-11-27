#!/usr/bin/env python3
"""AST-based seam-leakage guard for standalone PuXle CI."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from importlib.util import resolve_name
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
    relative_path = path.relative_to(root)
    package = ".".join(relative_path.parts[:-1])
    owner = relative_path.parts[1] if len(relative_path.parts) > 2 else None
    violations: list[SeamViolation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mod = ""
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                try:
                    mod = resolve_name("." * node.level + mod, package)
                except ImportError:
                    continue
        else:
            continue

        for alias in node.names:
            target = f"{mod}.{alias.name}" if mod else alias.name
            parts = target.split(".")
            if parts[0] == "puxle" and len(parts) > 1:
                target_owner = parts[1]
                if target_owner == owner or not _has_private_segment(target):
                    continue
                # Modules directly under puxle provide shared package infrastructure.
                if (root / "puxle" / f"{target_owner}.py").is_file():
                    continue
                kind = "private-cross-package-import"
                reason = (
                    f"Private cross-package import `{target}`; "
                    "expose a public interface in the owning PuXle package."
                )
            elif parts[0] == "xtructure":
                if _has_private_segment(target):
                    kind = "private-cross-repo-import"
                    reason = (
                        f"Private symbol or module imported from Xtructure: `{target}`."
                    )
                elif target in SANCTIONED_XTRUCTURE_SEAMS or (
                    mod in SANCTIONED_XTRUCTURE_SEAMS
                    and (
                        mod != "xtructure"
                        or alias.name not in UNAPPROVED_XTRUCTURE_CHILDREN
                    )
                ):
                    continue
                else:
                    kind = "internal-cross-repo-seam"
                    reason = (
                        f"Unsanctioned internal seam `{target}` imported from Xtructure; "
                        "use a sanctioned seam or document the adapter in CONTEXT.md."
                    )
            else:
                continue

            violations.append(
                SeamViolation(
                    path=relative_path.as_posix(),
                    lineno=node.lineno,
                    kind=kind,
                    raw_import=(ast.get_source_segment(source, node) or "").strip(),
                    reason=reason,
                )
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=".", help="Repository root (default: current directory)"
    )
    parser.add_argument(
        "--check-file",
        type=Path,
        default=None,
        help="Check a single file path inside the PuXle repository",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    violations: list[SeamViolation] = []
    if args.check_file is not None:
        files_checked = 1
        violations.extend(check_file_for_leaks(root, args.check_file.resolve()))
    else:
        files = iter_source_files(root)
        files_checked = len(files)
        for path in files:
            violations.extend(check_file_for_leaks(root, path))

    if args.json:
        print(
            json.dumps(
                {
                    "failed": bool(violations),
                    "files_checked": files_checked,
                    "violation_count": len(violations),
                    "violations": [v.to_dict() for v in violations],
                },
                indent=2,
            )
        )
    else:
        print(f"Files checked: {files_checked}, Seam violations: {len(violations)}")
        for v in violations:
            print(f"  {v.path}:{v.lineno} [{v.kind}] {v.raw_import}\n    -> {v.reason}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
