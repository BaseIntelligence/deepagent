"""Base-commit SHA validation + tree pin helpers for real envbuild (VAL-ENVR-001)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
# Pure pad fixtures: aaaa…, 0000…001, or labeled green/fixture prefixes.
_PADDED_HEX_RE = re.compile(
    r"^(?:([0-9a-f])\1{29,}[0-9a-f]*|"  # repeated nibble pad (e.g. aaaa… / 0000…)
    r"green[0-9a-f]*|"
    r"fixture[0-9a-f]*)$",
    re.IGNORECASE,
)


class BaseCommitError(ValueError):
    """Raised when a base_commit is invalid for a real-SHA envbuild path."""


def is_full_sha(value: str) -> bool:
    return bool(_FULL_SHA_RE.match((value or "").strip()))


def looks_synthetic_sha(value: str) -> bool:
    """Heuristic: wholly padded hex fixtures / placeholders are synthetic.

    Real mixed hex (e.g. a1b2c3…) is accepted. Motor-style inventory SHAs
    like a1000…001 / b1000… / c1000… are treated as synthetic.
    """
    v = (value or "").strip().lower()
    if not v:
        return True
    if v in {"local", "pending", "unknown", "none"}:
        return True
    if v.startswith(("green", "fixture")):
        return True
    # Historical motor inventory hex: letter + 1000…pad
    if re.match(r"^[a-f]1000+[0-9a-f]*$", v):
        return True
    if _PADDED_HEX_RE.match(v):
        return True
    # Same nibble ≥30 times on a full hex string → pad fixture
    return len(v) == 40 and set(v) <= set("0123456789abcdef") and v.count(v[0]) >= 30


def require_full_sha(value: str, *, allow_synthetic: bool = False) -> str:
    """Return normalized 40-char lower SHA or raise BaseCommitError."""
    raw = (value or "").strip().lower()
    if not is_full_sha(raw):
        raise BaseCommitError(
            f"base_commit must be a full 40-char git SHA for real envbuild; got {value!r}"
        )
    if not allow_synthetic and looks_synthetic_sha(raw):
        raise BaseCommitError(
            f"base_commit looks synthetic/placeholder and is refused for real-SHA builds: {value!r}"
        )
    return raw


def git_rev_parse_head(path: Path | str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_status_porcelain(path: Path | str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "ERROR"
    return completed.stdout if completed.returncode == 0 else "ERROR"


def assert_head_matches(path: Path | str, expected: str) -> str:
    """Ensure working tree HEAD equals *expected* (full or unique prefix)."""
    head = git_rev_parse_head(path)
    if not head:
        raise BaseCommitError(f"cannot resolve HEAD in {path}")
    exp = expected.strip().lower()
    head_l = head.lower()
    if head_l == exp or head_l.startswith(exp) or (len(exp) >= 7 and exp.startswith(head_l)):
        return head
    raise BaseCommitError(f"checked-out HEAD {head!r} does not match pin {expected!r}")


def scrub_git_history(path: Path | str) -> list[str]:
    """Drop remotes/reflogs so post-base future history is not easy agent fodder.

    Working tree remains at current HEAD (must already be pinned).
    """
    root = Path(path)
    actions: list[str] = []
    commands: list[list[str]] = [
        ["git", "-C", str(root), "remote", "remove", "origin"],
        ["git", "-C", str(root), "reflog", "expire", "--expire=now", "--all"],
        ["git", "-C", str(root), "gc", "--prune=now"],
        ["git", "-C", str(root), "config", "core.hooksPath", "/dev/null"],
    ]
    for cmd in commands:
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120.0, check=False
            )
            actions.append(f"{' '.join(cmd[3:])}:rc={completed.returncode}")
        except (OSError, subprocess.SubprocessError) as exc:
            actions.append(f"{' '.join(cmd[3:])}:err={exc}")
    return actions


def write_base_commit_marker(path: Path | str, sha: str) -> Path:
    """Write ``.harbor_base_commit`` marker under the workspace."""
    dest = Path(path) / ".harbor_base_commit"
    dest.write_text(f"{sha.strip()}\n", encoding="utf-8")
    # Keep out of dirty porcelain if .git/info/exclude exists.
    git_info = Path(path) / ".git" / "info"
    if git_info.is_dir():
        exclude = git_info / "exclude"
        text = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        if ".harbor_base_commit" not in text:
            with exclude.open("a", encoding="utf-8") as fh:
                fh.write(".harbor_base_commit\n")
    return dest


def isolation_scan(path: Path | str) -> dict[str, object]:
    """Scan agent workspace for forbidden solution leakage (VAL-ENVR-005)."""
    root = Path(path)
    hits: list[str] = []
    forbidden_names = {
        "solution",
        "solution.patch",
        "gold.patch",
        "test.patch",
        "solve.sh",
    }
    # solution/ directory at root or common agent-leaked names
    for name in forbidden_names:
        candidate = root / name
        if candidate.exists():
            hits.append(str(candidate.relative_to(root)))
    # Nested solution/ folder rarely but still toxic
    for p in root.rglob("solution"):
        if p.is_dir() and p != root:
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                rel = str(p)
            if rel not in hits:
                hits.append(rel)
    porcelain = git_status_porcelain(root)
    head = git_rev_parse_head(root)
    return {
        "clean": not hits,
        "hits": hits,
        "porcelain_empty": porcelain == "",
        "porcelain": porcelain[:500],
        "head": head,
    }


__all__ = [
    "BaseCommitError",
    "assert_head_matches",
    "git_rev_parse_head",
    "git_status_porcelain",
    "isolation_scan",
    "is_full_sha",
    "looks_synthetic_sha",
    "require_full_sha",
    "scrub_git_history",
    "write_base_commit_marker",
]
