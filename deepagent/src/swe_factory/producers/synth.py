"""Synthetic_grounded producer: multi-file multi-fault + function-removal.

On a *green* real-repo (or fixture) base, inject multi-file inversible mutations
and emit a labeled TaskRecord whose gold patch is the exact inverse of the
mutation. Gold is intended to validate under the certified oracle before export.

Track label is always ``source_track=synthetic_grounded``. Offline path needs
no LLM/network; optional Docker pass-through is available via certified oracle.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from swe_factory.oracle.docker_run import OracleRunnerBackend
from swe_factory.oracle.gates import (
    GateResult,
    append_gate_audit,
    count_files_in_patch,
    run_certified_gates_for_task,
    run_stub_gates,
)
from swe_factory.schema import EnvironmentMeta, SourceTrack, TaskRecord
from swe_factory.sources.allowlist import TINY_GREEN, SeedRepo, get_seed

MutationKind = Literal["multi_fault", "function_removal"]

MUTATION_MULTI_FAULT: MutationKind = "multi_fault"
MUTATION_FUNCTION_REMOVAL: MutationKind = "function_removal"

_DEFAULT_IMAGE_DIGEST = "sha256:synth_grounded_pending"


class SynthError(RuntimeError):
    """Raised when synthetic candidate construction fails."""


@dataclass(frozen=True, slots=True)
class MutationTarget:
    """One source file + optional symbol chosen for a fault."""

    relative_path: str
    kind: MutationKind
    symbol: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SynthCandidate:
    """Produced labeled synthetic_grounded candidate with inverse gold."""

    task: TaskRecord
    broken_workspace: Path
    green_workspace: Path | None
    mutation_kind: MutationKind
    targets: tuple[MutationTarget, ...]
    gold_files: tuple[str, ...]
    inverse_meta: dict[str, Any]
    gates: GateResult | None = None
    provider_calls: int = 0

    @property
    def source_track(self) -> str:
        track = self.task.source_track
        return track.value if hasattr(track, "value") else str(track)


def _copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            "node_modules",
            ".pytest_cache",
            "*.egg-info",
        ),
    )


def _list_python_source_files(workspace: Path) -> list[Path]:
    """Prefer non-test modular package modules under the workspace.

    Skips scripts, docs helpers, easter eggs, and other non-product trees that
    lack a corresponding tests/test_*.py motor for F2P mapping.
    """
    skip_parts = {
        "tests",
        "test",
        "docs",
        "doc",
        "examples",
        "example",
        "scripts",
        "script",
        "misc",
        "benchmark",
        "benchmarks",
        "vendor",
        "third_party",
        ".tox",
        ".venv",
    }
    skip_names = {
        "conftest.py",
        "setup.py",
        "setup.cfg",
        "easterutils.py",
        "version.py",
        "_version.py",
    }
    files: list[Path] = []
    for path in sorted(workspace.rglob("*.py")):
        rel = path.relative_to(workspace).as_posix()
        parts = set(path.relative_to(workspace).parts)
        if any(part.startswith(".") for part in path.parts):
            continue
        if parts & skip_parts:
            continue
        if "/tests/" in f"/{rel}/" or rel.startswith("tests/") or rel.endswith("_test.py"):
            continue
        if path.name in skip_names:
            continue
        if path.name == "__init__.py":
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not re.search(r"^\s*def\s+\w+", text, re.MULTILINE):
                continue
        # Prefer modules that have a colocated tests/test_<stem>.py (real repos).
        stem = path.stem
        test_hit = any(
            (workspace / cand).is_file()
            for cand in (
                Path("tests") / f"test_{stem}.py",
                Path("test") / f"test_{stem}.py",
            )
        )
        # Local fixtures may have no naming convention — keep them via package path.
        if not test_hit and (workspace / "tests").is_dir() and "boltons" in parts:
            # skip boltons modules without dedicated unit tests when tests/ exists
            continue
        files.append(path)
    return files


def _list_js_source_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(workspace.rglob("*.js")):
        rel = path.relative_to(workspace).as_posix()
        if any(part.startswith(".") for part in path.parts):
            continue
        if "/test/" in f"/{rel}/" or "/tests/" in f"/{rel}/" or rel.endswith(".test.js"):
            continue
        files.append(path)
    return files


def _list_go_source_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(workspace.rglob("*.go")):
        if path.name.endswith("_test.go"):
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        files.append(path)
    return files


# Prefer *module-level* defs (indent empty) for multi_fault: class methods have
# brittle spans and dilute F2P motors. Function-removal may still use indented.
_PY_FUNC_RE = re.compile(
    r"(?P<indent>^[ \t]*)def\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^)]*)\)\s*"
    r"(?:->\s*[^:]+)?:\s*\n(?P<body>(?:(?:\1[ \t]+|\s*#).*\n|\s*\n)*)",
    re.MULTILINE,
)
_PY_MODULE_FUNC_RE = re.compile(
    r"(?P<indent>^)def\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^)]*)\)\s*"
    r"(?:->\s*[^:]+)?:\s*\n(?P<body>(?:(?:[ \t]+|\s*#).*\n|\s*\n)*)",
    re.MULTILINE,
)


def _break_python_expr(expr: str) -> str | None:
    expr = expr.rstrip()
    if expr in {"None", "True", "False", "...", "NotImplemented"}:
        return None
    if re.fullmatch(r"[A-Za-z_]\w*\s*\+\s*[A-Za-z_]\w*", expr):
        flipped = re.sub(r"\+", "-", expr, count=1)
        return str(flipped)
    if re.fullmatch(r"[A-Za-z_]\w*\s*-\s*[A-Za-z_]\w*", expr):
        flipped = re.sub(r"-", "+", expr, count=1)
        return str(flipped)
    if "reversed(" in expr or '" ".join(reversed(' in expr:
        return "text" if "text" in expr else "s" if "s" in expr else "''"
    if expr.startswith('"') or expr.startswith("'"):
        return "''"
    if re.fullmatch(r"-?\d+(\.\d+)?", expr):
        return "0" if expr != "0" else "1"
    names = list(re.findall(r"[A-Za-z_]\w*", expr))
    if names:
        first = names[0]
        return str(first)
    return None


def _simple_wrong_return_python(body: str) -> str | None:
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)return\s+(.+)$", ln)
        if not m:
            continue
        ret_indent, expr = m.group(1), m.group(2).rstrip()
        # Drop trailing inline comments for expression rewrite.
        expr_code = expr.split("  #", 1)[0].rstrip()
        broken = _break_python_expr(expr_code)
        if broken is None:
            # Generic safe flip: invert Truthy return or swap to NotImplemented-ish None.
            if expr_code not in {"None", "True", "False"}:
                broken = "None"
            elif expr_code == "True":
                broken = "False"
            elif expr_code == "False":
                broken = "True"
            else:
                broken = "0"
        new_lines = list(lines)
        new_lines[i] = f"{ret_indent}return {broken}"
        return "\n".join(new_lines) + ("\n" if body.endswith("\n") else "")
    return None


def _replace_python_body(text: str, name: str, new_body: str) -> str | None:
    pat = re.compile(
        rf"(^[ \t]*def\s+{re.escape(name)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*\n)"
        rf"((?:(?:[ \t]+|\s*#).*\n|\s*\n)*)",
        re.MULTILINE,
    )
    m2 = pat.search(text)
    if not m2:
        return None
    out = text[: m2.start(2)] + new_body + text[m2.end(2) :]
    return out if out != text else None


def mutate_python_multi_fault(
    text: str,
    *,
    max_funcs: int = 2,
    max_body_lines: int = 40,
    module_level_only: bool = False,
    rank_offset: int = 0,
) -> tuple[str, list[str]]:
    """Break up to max_funcs return bodies; return (mutated, symbols).

    Prefer compact helpers. Optional ``module_level_only`` for real-repo
    hardness (skips class methods that dig huge regex spans).
    ``rank_offset`` rotates candidate function order for harvest diversity.
    """
    matches = list(_PY_FUNC_RE.finditer(text))
    if not matches:
        raise SynthError("no Python functions found for multi_fault")
    ranked = sorted(
        matches,
        key=lambda m: (
            m.group("name").startswith("_"),
            0 if m.group("indent") == "" else 1,
            len(m.group("body").splitlines()),
            len(m.group("body")),
            m.start(),
        ),
    )
    if rank_offset and ranked:
        rot = int(rank_offset) % len(ranked)
        ranked = ranked[rot:] + ranked[:rot]
    symbols: list[str] = []
    out = text
    for match in ranked:
        if len(symbols) >= max_funcs:
            break
        name = match.group("name")
        if name.startswith("__") or name.startswith("test"):
            continue
        if module_level_only and match.group("indent") != "":
            continue
        body = match.group("body")
        if len(body.splitlines()) > max_body_lines:
            continue
        new_body = _simple_wrong_return_python(body)
        if new_body is None:
            continue
        candidate = _replace_python_body(out, name, new_body)
        if candidate is None or candidate == out:
            continue
        # Guard against catastrophic rewrite of large modules only.
        # Small pure-helper fixtures can shrink a lot when one return flips.
        if len(out) > 2000 and len(candidate) < 0.7 * len(out):
            continue
        out = candidate
        symbols.append(name)
    if not symbols:
        raise SynthError("could not mutate any small Python function bodies")
    return out, symbols


def mutate_python_function_removal(
    text: str,
    *,
    module_level_only: bool = False,
    max_body_lines: int = 80,
    rank_offset: int = 0,
) -> tuple[str, list[str]]:
    """Replace function body with ``raise NotImplementedError`` (inversible).

    Prefer mid-size module-level pure helpers over giant methods so inverse gold
    is restorative and F2P is motorized without half-file deletion spans.
    ``rank_offset`` skips the first N ranked candidates for diversity.
    """
    matches = list(_PY_FUNC_RE.finditer(text))
    if not matches:
        raise SynthError("no Python functions found for function_removal")
    # Prefer non-private module-level mid-size bodies.
    ranked = sorted(
        matches,
        key=lambda m: (
            m.group("name").startswith("_"),
            0 if m.group("indent") == "" else 1,
            abs(len(m.group("body").splitlines()) - 12),
            m.start(),
        ),
    )
    if rank_offset and ranked:
        skip = int(rank_offset) % len(ranked)
        ranked = ranked[skip:] + ranked[:skip]
    for match in ranked:
        name = match.group("name")
        if name.startswith("__") or name.startswith("test"):
            continue
        indent = match.group("indent")
        if module_level_only and indent != "":
            continue
        nlines = len(match.group("body").splitlines())
        if nlines > max_body_lines or nlines < 1:
            continue
        body_indent = (indent or "") + "    "
        stub = f"{body_indent}raise NotImplementedError({name!r})\n"
        out = _replace_python_body(text, name, stub)
        if out is None or out == text:
            continue
        # Only reject catastrophic shrink on large files.
        if len(text) > 2000 and len(out) < 0.5 * len(text):
            continue
        return out, [name]
    raise SynthError("could not remove any Python function body")


def mutate_js_multi_fault(text: str, *, max_funcs: int = 2) -> tuple[str, list[str]]:
    matches = list(
        re.finditer(
            r"(?P<head>(?:export\s+)?function\s+(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)\s*\{)"
            r"(?P<body>.*?)"
            r"(?P<tail>\n\})",
            text,
            re.DOTALL,
        )
    )
    if not matches:
        raise SynthError("no JS functions found for multi_fault")
    out = text
    symbols: list[str] = []
    for match in matches:
        if len(symbols) >= max_funcs:
            break
        name = match.group("name")
        body = match.group("body")
        new_body, n = re.subn(r"return\s+([^;]+);", r"return null;", body, count=1)
        if n == 0:
            new_body = "\n  throw new Error('synth fault');\n" + body
        old = match.group("head") + body + match.group("tail")
        new = match.group("head") + new_body + match.group("tail")
        if old not in out:
            continue
        out = out.replace(old, new, 1)
        symbols.append(name)
    if not symbols:
        raise SynthError("could not mutate any JS functions")
    return out, symbols


def mutate_js_function_removal(text: str) -> tuple[str, list[str]]:
    matches = list(
        re.finditer(
            r"(?P<head>(?:export\s+)?function\s+(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)\s*\{)"
            r"(?P<body>.*?)"
            r"(?P<tail>\n\})",
            text,
            re.DOTALL,
        )
    )
    if not matches:
        raise SynthError("no JS functions found for function_removal")
    match = matches[0]
    name = match.group("name")
    old = match.group("head") + match.group("body") + match.group("tail")
    stub = f"\n  throw new Error('function removed: {name}');\n"
    new = match.group("head") + stub + match.group("tail")
    if old not in text:
        raise SynthError("JS function removal target vanished")
    return text.replace(old, new, 1), [name]


def mutate_go_multi_fault(text: str, *, max_funcs: int = 2) -> tuple[str, list[str]]:
    matches = list(
        re.finditer(
            r"(?P<head>func\s+(?:\([^)]+\)\s+)?(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)[^{]*\{)"
            r"(?P<body>.*?)"
            r"(?P<tail>\n\})",
            text,
            re.DOTALL,
        )
    )
    if not matches:
        raise SynthError("no Go functions found for multi_fault")
    out = text
    symbols: list[str] = []
    for match in matches:
        if len(symbols) >= max_funcs:
            break
        name = match.group("name")
        if name == "init" or name.startswith("Test"):
            continue
        body = match.group("body")
        if "return" not in body and "panic" not in body:
            continue
        old = match.group("head") + body + match.group("tail")
        new = match.group("head") + '\n\tpanic("synth fault")\n' + match.group("tail")
        if old not in out:
            continue
        out = out.replace(old, new, 1)
        symbols.append(name)
    if not symbols:
        raise SynthError("could not mutate any Go functions")
    return out, symbols


def mutate_go_function_removal(text: str) -> tuple[str, list[str]]:
    matches = list(
        re.finditer(
            r"(?P<head>func\s+(?:\([^)]+\)\s+)?(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)[^{]*\{)"
            r"(?P<body>.*?)"
            r"(?P<tail>\n\})",
            text,
            re.DOTALL,
        )
    )
    if not matches:
        raise SynthError("no Go functions found for function_removal")
    for match in matches:
        name = match.group("name")
        if name == "init" or name.startswith("Test"):
            continue
        old = match.group("head") + match.group("body") + match.group("tail")
        new = match.group("head") + '\n\tpanic("function removed")\n' + match.group("tail")
        if old not in text:
            continue
        return text.replace(old, new, 1), [name]
    raise SynthError("could not remove any Go function")


def _language_mutators(
    language: str,
) -> tuple[
    Callable[[Path], list[Path]],
    Callable[[str], tuple[str, list[str]]],
    Callable[[str], tuple[str, list[str]]],
]:
    lang = language.lower()
    if lang in {"js", "javascript", "typescript", "ts"}:
        return _list_js_source_files, mutate_js_multi_fault, mutate_js_function_removal
    if lang == "go":
        return _list_go_source_files, mutate_go_multi_fault, mutate_go_function_removal
    return _list_python_source_files, mutate_python_multi_fault, mutate_python_function_removal


def build_problem_statement(
    *,
    mutation_kind: MutationKind,
    targets: Sequence[MutationTarget],
    repo: str,
    language: str,
    broken_workspace: Path | None = None,
    max_snippet_chars: int = 1200,
) -> str:
    """Non-empty agent-facing problem statement (VAL-PROD-003).

    When ``broken_workspace`` is provided, attach short source snippets so
    panel/agents can emit a correct path-relative unified diff without browsing.
    """
    files = ", ".join(sorted({t.relative_path for t in targets}))
    symbols = ", ".join(sorted({t.symbol for t in targets if t.symbol})) or "(module helpers)"
    kind_label = (
        "multi-file multi-fault regression"
        if mutation_kind == MUTATION_MULTI_FAULT
        else "function-body removal regression"
    )
    base = (
        f"The {language} package in `{repo}` has a {kind_label}. "
        f"Affected sources include {files}. "
        f"Relevant symbols: {symbols}. "
        "Restore the original multi-module behaviour so the fail_to_pass tests pass "
        "without weakening pass_to_pass regressions. "
        "Do not delete tests; implement the missing/correct logic. "
        "Return a unified diff with paths relative to the repository root "
        f"(e.g. `--- a/{next(iter(sorted({t.relative_path for t in targets})), 'path.py')}`), "
        "never under fixtures/ or absolute prefixes."
    )
    if broken_workspace is None:
        return base
    root = Path(broken_workspace)
    # Signatures-only context: enough path guidance without full buggy bodies
    # (full bodies make trivial multi-faults solve-all for frontier models).
    sig_lines: list[str] = []
    by_file: dict[str, list[str | None]] = {}
    for t in targets:
        by_file.setdefault(t.relative_path, []).append(t.symbol)
    for rel in sorted(by_file):
        path = root / rel
        if not path.is_file():
            continue
        full = path.read_text(encoding="utf-8", errors="replace")
        file_symbols = [s for s in by_file[rel] if s]
        for sym in file_symbols:
            m = re.search(
                rf"^[ \t]*def\s+{re.escape(sym)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
                full,
                re.MULTILINE,
            )
            if m:
                sig_lines.append(f"- `{rel}` :: `{m.group(0).strip()}`")
            else:
                sig_lines.append(f"- `{rel}` :: `{sym}`")
        if not file_symbols:
            sig_lines.append(f"- `{rel}`")
    if not sig_lines:
        return base
    return (
        base
        + "\n\nAffected signatures (bodies intentionally omitted from the prompt):\n"
        + "\n".join(sig_lines)
        + "\n\nInspect the repository for incorrect implementations and emit a "
        "minimal multi-file unified diff."
    )


def unified_diff_trees(broken: Path, green: Path) -> str:
    """Build a multi-file unified diff that transforms *broken* → *green* (gold)."""
    if not broken.is_dir() or not green.is_dir():
        raise SynthError("unified_diff_trees requires two directories")

    with tempfile.TemporaryDirectory(prefix="sdf-synth-diff-") as tmp:
        repo = Path(tmp) / "repo"
        _copy_tree(broken, repo)
        subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=False)
        for key, value in (("user.email", "synth@localhost"), ("user.name", "synth")):
            subprocess.run(
                ["git", "config", key, value],
                cwd=repo,
                capture_output=True,
                check=False,
            )
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, check=False)
        commit = subprocess.run(
            ["git", "commit", "-m", "broken baseline"],
            cwd=repo,
            capture_output=True,
            check=False,
            text=True,
        )
        if commit.returncode != 0:
            (repo / ".sdf_keep").write_text("keep\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, check=False)
            subprocess.run(
                ["git", "commit", "-m", "broken baseline"],
                cwd=repo,
                capture_output=True,
                check=False,
            )

        for path in list(repo.rglob("*")):
            if path.is_file() and ".git" not in path.parts:
                path.unlink(missing_ok=True)
        for path in sorted((p for p in repo.rglob("*") if p.is_dir()), reverse=True):
            if path.name == ".git" or ".git" in path.parts:
                continue
            with contextlib.suppress(OSError):
                path.rmdir()

        for src in green.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(green)
            if rel.parts and rel.parts[0] == ".git":
                continue
            dest = repo / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, check=False)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--no-color", "--find-renames"],
            cwd=repo,
            capture_output=True,
            check=False,
            text=True,
        )
        patch = diff.stdout or ""
        if not patch.strip():
            raise SynthError(
                "empty gold diff: green tree equals broken tree (mutation had no textual effect)"
            )
        return patch


def _apply_mutations(
    broken_ws: Path,
    *,
    language: str,
    mutation_kind: MutationKind,
    min_files: int = 2,
    diversification_index: int = 0,
    prefer_stems: Sequence[str] | None = None,
    exclude_stems: Sequence[str] | None = None,
) -> list[MutationTarget]:
    """Inject multi-file mutations, optionally rotated for harvest diversity.

    ``diversification_index`` rotates preferred target stems so repeated
    harvest attempts on the same seed produce distinct candidate sets.
    """
    import hashlib

    list_files, multi_mut, remove_mut = _language_mutators(language)
    candidates = list_files(broken_ws)
    if not candidates:
        raise SynthError(f"no source files found for language={language!r} in {broken_ws}")
    if len(candidates) < min_files:
        raise SynthError(
            f"need ≥{min_files} source files for {mutation_kind}; found {len(candidates)}"
        )

    targets: list[MutationTarget] = []
    primary = multi_mut if mutation_kind == MUTATION_MULTI_FAULT else remove_mut
    fallback = remove_mut if mutation_kind == MUTATION_MULTI_FAULT else multi_mut

    # Prefer pure-logic helpers first (math/str/iter) for multi_fault hardness band.
    preferred_stems = list(
        prefer_stems
        or (
            "mathutils",
            "strutils",
            "iterutils",
            "dictutils",
            "listutils",
            "setutils",
            "formatutils",
            "funcutils",
            "timeutils",
            "urlutils",
            "typeutils",
            "statsutils",
            "cacheutils",
            "tableutils",
            "tbutils",
            "namedutils",
            "fileutils",
            "ioutils",
            "jsonutils",
            "socketutils",
            "gcutils",
            "ecoutils",
            "debugutils",
            # cachetools / other modular libs
            "lru",
            "lfu",
            "ttl",
            "rr",
            "fifo",
            "mru",
            "keys",
            "func",
            # js / go common stems
            "parse",
            "stringify",
            "formats",
            "caste",
            "time",
            "version",
            "hash",
            "node",
            "null",
            "dce",
            "util",
            "md5",
            "sha1",
        )
    )
    exclude = {s.strip().lower() for s in (exclude_stems or []) if s and str(s).strip()}
    div = max(0, int(diversification_index))
    if preferred_stems and div:
        rot = div % len(preferred_stems)
        preferred_stems = preferred_stems[rot:] + preferred_stems[:rot]

    def _rank(path: Path) -> tuple[int, int, int, str]:
        stem = path.stem.lower()
        if stem in exclude:
            return (10_000, 0, path.stat().st_size, path.as_posix())
        try:
            pref = preferred_stems.index(stem)
        except ValueError:
            # stabilize tertiary order via diversification-stable hash
            dig = hashlib.sha1(f"{div}:{path.as_posix()}".encode()).hexdigest()
            pref = 100 + int(dig[:4], 16) % 500
        # secondary rotate by size bucket so larger modules enter when div high
        size_bias = (path.stat().st_size // 400) % 7
        if div % 2 == 1:
            size_bias = -size_bias
        return (pref, size_bias, path.stat().st_size, path.as_posix())

    ordered = sorted(candidates, key=_rank)
    # Skip some mutatable files when diversifying so first hits differ.
    if div and len(ordered) > min_files + 1:
        skip_n = div % min(4, max(1, len(ordered) - min_files))
        ordered = ordered[skip_n:] + ordered[:skip_n]

    attempt_skip_func = div  # rotate which function match rank is preferred

    for path in ordered:
        if len({t.relative_path for t in targets}) >= min_files:
            break
        if path.stem.lower() in exclude:
            continue
        text = path.read_text(encoding="utf-8")
        new_text: str | None = None
        symbols: list[str] = []
        try:
            if mutation_kind == MUTATION_MULTI_FAULT:
                # Detect large real-repo modules (>8kb) and constrain mutations.
                large = len(text) > 8000
                try:
                    new_text, symbols = multi_mut(
                        text,
                        max_funcs=1,
                        max_body_lines=12 if large else 40,
                        module_level_only=large,
                        rank_offset=attempt_skip_func,
                    )  # type: ignore[call-arg]
                except TypeError:
                    try:
                        new_text, symbols = multi_mut(
                            text,
                            max_funcs=1,
                            max_body_lines=12 if large else 40,
                            module_level_only=large,
                        )  # type: ignore[call-arg]
                    except TypeError:
                        new_text, symbols = multi_mut(text, max_funcs=1)  # type: ignore[call-arg]
            else:
                # function_removal — constrain only truly large real-repo modules.
                large = len(text) > 8000
                try:
                    new_text, symbols = remove_mut(
                        text,
                        module_level_only=large,
                        max_body_lines=40 if large else 80,
                        rank_offset=attempt_skip_func,
                    )  # type: ignore[call-arg]
                except TypeError:
                    try:
                        new_text, symbols = remove_mut(
                            text,
                            module_level_only=large,
                            max_body_lines=40 if large else 80,
                        )  # type: ignore[call-arg]
                    except TypeError:
                        new_text, symbols = primary(text)
                # Tiny modules may need unrestricted fallback if constrained call fails later.
                if new_text is None:
                    new_text, symbols = primary(text)
        except SynthError:
            try:
                new_text, symbols = fallback(text)
            except SynthError:
                continue
        if new_text is None or new_text == text:
            continue
        # Reject catastrophic rewrites on large modules only.
        if len(text) > 2000 and len(new_text) < 0.6 * len(text):
            continue
        path.write_text(new_text, encoding="utf-8")
        rel = path.relative_to(broken_ws).as_posix()
        for sym in symbols:
            targets.append(
                MutationTarget(
                    relative_path=rel,
                    kind=mutation_kind,
                    symbol=sym,
                    detail=(
                        "body return/fault injection"
                        if mutation_kind == MUTATION_MULTI_FAULT
                        else "function body removed / stubbed"
                    ),
                )
            )

    if len({t.relative_path for t in targets}) < min_files:
        raise SynthError(
            f"{mutation_kind} could not touch ≥{min_files} files "
            f"(touched={sorted({t.relative_path for t in targets})})"
        )
    return targets


def _instance_id(seed: SeedRepo, mutation_kind: MutationKind, suffix: str) -> str:
    repo_slug = re.sub(r"[^a-zA-Z0-9]+", "_", seed.repo).strip("_").lower()
    return f"synth__{repo_slug}__{mutation_kind}__{suffix}"


@dataclass
class SynthProducer:
    """Build synthetic_grounded candidates with inverse gold on green bases."""

    work_root: Path | None = None
    image_digest: str = _DEFAULT_IMAGE_DIGEST
    min_files: int = 2
    keep_workspaces: bool = True

    def produce(
        self,
        seed: SeedRepo,
        *,
        mutation_kind: MutationKind = MUTATION_MULTI_FAULT,
        base_path: Path | None = None,
        instance_suffix: str | None = None,
        problem_statement: str | None = None,
        fail_to_pass: Sequence[str] | None = None,
        pass_to_pass: Sequence[str] | None = None,
        run_stub_oracle: bool = True,
        diversification_index: int = 0,
        prefer_stems: Sequence[str] | None = None,
        exclude_stems: Sequence[str] | None = None,
    ) -> SynthCandidate:
        """Produce one labeled candidate from a green seed checkout.

        Offline-capable when seed has a local fixture path.
        ``provider_calls`` is always 0 for pure mutation paths.
        ``diversification_index`` rotates file/func targets for harvest diversity.
        """
        green_src = base_path or seed.resolve_local_path()
        if green_src is None or not Path(green_src).is_dir():
            raise SynthError(
                f"seed {seed.seed_id!r} has no local green base; "
                f"clone {seed.repo}@{seed.base_commit}"
            )
        green_src = Path(green_src)

        work_root = (
            Path(self.work_root) if self.work_root else Path(tempfile.mkdtemp(prefix="sdf-synth-"))
        )
        work_root.mkdir(parents=True, exist_ok=True)
        suffix = instance_suffix or uuid.uuid4().hex[:10]
        case_dir = work_root / f"{seed.seed_id}_{mutation_kind}_{suffix}"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        green_ws = case_dir / "green"
        broken_ws = case_dir / "broken"
        _copy_tree(green_src, green_ws)
        _copy_tree(green_src, broken_ws)

        targets = _apply_mutations(
            broken_ws,
            language=seed.language,
            mutation_kind=mutation_kind,
            min_files=self.min_files,
            diversification_index=int(diversification_index),
            prefer_stems=prefer_stems,
            exclude_stems=exclude_stems,
        )
        gold_patch = unified_diff_trees(broken_ws, green_ws)
        gold_files = tuple(count_files_in_patch(gold_patch))
        if len(gold_files) < self.min_files:
            raise SynthError(
                f"gold patch multi-file floor failed: files={list(gold_files)} "
                f"(targets={[t.relative_path for t in targets]})"
            )

        f2p = list(fail_to_pass) if fail_to_pass is not None else list(seed.f2p_commands)
        p2p = list(pass_to_pass) if pass_to_pass is not None else list(seed.p2p_commands)
        if not f2p:
            f2p = [seed.baseline_test_command or "python -m pytest -q"]
            p2p = []

        prompt = problem_statement or build_problem_statement(
            mutation_kind=mutation_kind,
            targets=targets,
            repo=seed.repo,
            language=seed.language,
            broken_workspace=broken_ws,
        )
        if not prompt.strip():
            raise SynthError("problem_statement must be non-empty")

        inverse_meta: dict[str, Any] = {
            "derivation": "inverse_of_synthetic_mutation",
            "mutation_kind": mutation_kind,
            "seed_id": seed.seed_id,
            "targets": [
                {
                    "path": t.relative_path,
                    "kind": t.kind,
                    "symbol": t.symbol,
                    "detail": t.detail,
                }
                for t in targets
            ],
            "gold_files": list(gold_files),
            "gold_is_inverse": True,
            "source_track": SourceTrack.SYNTHETIC_GROUNDED.value,
        }

        instance_id = _instance_id(seed, mutation_kind, suffix)
        task = TaskRecord.model_validate(
            {
                "instance_id": instance_id,
                "source_track": SourceTrack.SYNTHETIC_GROUNDED,
                "repo": seed.repo,
                "base_commit": seed.base_commit,
                "language": seed.language,
                "problem_statement": prompt,
                "fail_to_pass": f2p,
                "pass_to_pass": p2p,
                "gold_patch": gold_patch,
                "environment": EnvironmentMeta(image_digest=self.image_digest),
                "license": seed.license,
                "requirements": (
                    ",".join(seed.install_commands) if seed.install_commands else None
                ),
                "gate_proof": {
                    "inverse_meta": inverse_meta,
                    "producer": "synthetic_grounded",
                },
                "created_at": datetime.now(UTC),
            }
        )

        gates: GateResult | None = None
        if run_stub_oracle:
            gates = run_stub_gates(task, require_multi_file=True)
            if not gates.passed:
                raise SynthError(f"stub oracle rejected synthetic candidate: {gates.reason_codes}")
            task = task.model_copy(
                update={
                    "gate_proof": {
                        **(task.gate_proof or {}),
                        **gates.to_gate_proof(),
                        "inverse_meta": inverse_meta,
                    }
                }
            )

        (case_dir / "gold.patch").write_text(
            gold_patch if gold_patch.endswith("\n") else gold_patch + "\n",
            encoding="utf-8",
        )
        (case_dir / "inverse_meta.json").write_text(
            json.dumps(inverse_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return SynthCandidate(
            task=task,
            broken_workspace=broken_ws,
            green_workspace=green_ws if self.keep_workspaces else None,
            mutation_kind=mutation_kind,
            targets=tuple(targets),
            gold_files=gold_files,
            inverse_meta=inverse_meta,
            gates=gates,
            provider_calls=0,
        )

    def produce_and_certify(
        self,
        seed: SeedRepo,
        *,
        runner: OracleRunnerBackend,
        mutation_kind: MutationKind = MUTATION_MULTI_FAULT,
        base_path: Path | None = None,
        dual_runs: int = 2,
        audit_out: Path | None = None,
        **produce_kwargs: Any,
    ) -> SynthCandidate:
        """Produce candidate then run certified oracle (Fake or Docker backend)."""
        produce_kwargs.pop("run_stub_oracle", None)
        candidate = self.produce(
            seed,
            mutation_kind=mutation_kind,
            base_path=base_path,
            run_stub_oracle=True,
            **produce_kwargs,
        )
        result = run_certified_gates_for_task(
            candidate.task,
            workspace=candidate.broken_workspace,
            runner=runner,
            agent_workspace=None,
            require_multi_file=True,
            dual_runs=dual_runs,
            check_null_patch=True,
            check_leak=True,
        )
        if audit_out is not None:
            append_gate_audit(
                audit_out,
                result,
                candidate.task.instance_id,
                extra={"source_track": SourceTrack.SYNTHETIC_GROUNDED.value},
            )
        task = candidate.task.model_copy(
            update={
                "gate_proof": {
                    **(candidate.task.gate_proof or {}),
                    **result.to_gate_proof(),
                    "inverse_meta": candidate.inverse_meta,
                }
            }
        )
        if not result.passed:
            raise SynthError(
                f"certified oracle rejected synthetic candidate "
                f"{task.instance_id}: {result.reason_codes}"
            )
        return SynthCandidate(
            task=task,
            broken_workspace=candidate.broken_workspace,
            green_workspace=candidate.green_workspace,
            mutation_kind=candidate.mutation_kind,
            targets=candidate.targets,
            gold_files=candidate.gold_files,
            inverse_meta=candidate.inverse_meta,
            gates=result,
            provider_calls=0,
        )


def produce_from_green_fixture(
    *,
    mutation_kind: MutationKind = MUTATION_MULTI_FAULT,
    work_root: Path | None = None,
    run_stub_oracle: bool = True,
    seed_id: str = "fixture_tiny_green",
) -> SynthCandidate:
    """Convenience offline entry: seed allowlist local green fixture."""
    seed = get_seed(seed_id) if seed_id != TINY_GREEN.seed_id else TINY_GREEN
    return SynthProducer(work_root=work_root).produce(
        seed,
        mutation_kind=mutation_kind,
        run_stub_oracle=run_stub_oracle,
    )


__all__ = [
    "MUTATION_FUNCTION_REMOVAL",
    "MUTATION_MULTI_FAULT",
    "MutationKind",
    "MutationTarget",
    "SynthCandidate",
    "SynthError",
    "SynthProducer",
    "build_problem_statement",
    "mutate_go_function_removal",
    "mutate_go_multi_fault",
    "mutate_js_function_removal",
    "mutate_js_multi_fault",
    "mutate_python_function_removal",
    "mutate_python_multi_fault",
    "produce_from_green_fixture",
    "unified_diff_trees",
]
