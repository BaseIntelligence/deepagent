"""Feature-deletion mutators for synthetic benchmark tasks."""

from __future__ import annotations

import ast
import difflib
from pathlib import Path

from swe_forge.synthetic.models import PythonFeatureDeletion


class FeatureDeletionError(ValueError):
    """Raised when a feature cannot be safely deleted."""


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise FeatureDeletionError(f"{path} is not inside {root}") from exc


def _diff(old: str, new: str, rel_path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )


def _find_symbol(tree: ast.AST, symbol: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
    ]
    if not matches:
        raise FeatureDeletionError(f"Function or method not found: {symbol}")
    if len(matches) > 1:
        raise FeatureDeletionError(
            f"Symbol {symbol!r} is ambiguous; found {len(matches)} matches"
        )
    return matches[0]


def build_python_function_deletion(
    repo_root: Path | str,
    source_file: Path | str,
    symbol: str,
) -> PythonFeatureDeletion:
    """Create deletion/oracle patches by replacing a Python function body.

    The generated deletion patch keeps the public signature intact and replaces
    the body with ``NotImplementedError``. The oracle patch is the inverse patch.
    """
    root = Path(repo_root).resolve()
    source_path = Path(source_file)
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()

    if source_path.suffix != ".py":
        raise FeatureDeletionError("Only Python source files are supported")
    if not source_path.exists():
        raise FeatureDeletionError(f"Source file does not exist: {source_path}")

    original = source_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(original)
    except SyntaxError as exc:
        raise FeatureDeletionError(f"Cannot parse Python source: {exc}") from exc

    node = _find_symbol(tree, symbol)
    if not node.body:
        raise FeatureDeletionError(f"Symbol {symbol!r} has no body")

    first = node.body[0]
    last = node.body[-1]
    if first.lineno is None or last.end_lineno is None:
        raise FeatureDeletionError("Python AST is missing line positions")

    lines = original.splitlines(keepends=True)
    body_start = first.lineno - 1
    body_end = last.end_lineno
    indent = " " * first.col_offset
    replacement = (
        f'{indent}raise NotImplementedError("Synthetic feature deletion: {symbol}")\n'
    )

    mutated_lines = [*lines[:body_start], replacement, *lines[body_end:]]
    mutated = "".join(mutated_lines)
    rel_path = _relative_posix(source_path, root)

    deletion_patch = _diff(original, mutated, rel_path)
    oracle_patch = _diff(mutated, original, rel_path)

    if not deletion_patch or not oracle_patch:
        raise FeatureDeletionError("Generated empty patch")

    return PythonFeatureDeletion(
        source_file=Path(rel_path),
        symbol=symbol,
        deletion_patch=deletion_patch,
        oracle_patch=oracle_patch,
        original_source=original,
        mutated_source=mutated,
    )
