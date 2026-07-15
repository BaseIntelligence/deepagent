"""Clone and pin allowlisted seeds to immutable SHAs.

Certified keeps must pin immutable base_commit SHAs (not branch names).
This module clones remote seeds shallowly when needed, checks out a pin, and
records the resolved full SHA for task records.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from swe_factory.sources.allowlist import SeedRepo

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


class CloneError(RuntimeError):
    """Raised when a seed cannot be cloned or pinned to an immutable SHA."""


@dataclass(frozen=True, slots=True)
class PinnedCheckout:
    """A green base checkout pinned to an immutable git SHA."""

    seed_id: str
    repo: str
    path: Path
    base_commit: str  # full or long SHA
    source: str  # local_fixture | clone


def _run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def is_immutable_sha(value: str) -> bool:
    """Return True when value looks like a hex commit SHA (not a branch name)."""
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"master", "main", "head", "develop", "trunk"}:
        return False
    return bool(_SHA_RE.fullmatch(cleaned))


def resolve_full_sha(repo_path: Path, ref: str | None = None) -> str:
    """Resolve HEAD (or ref) to the full 40-char SHA."""
    args = ["rev-parse", ref] if ref else ["rev-parse", "HEAD"]
    proc = _run_git(args, cwd=repo_path)
    if proc.returncode != 0:
        raise CloneError(
            f"cannot resolve SHA for {repo_path} ref={ref!r}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    sha = proc.stdout.strip()
    if not is_immutable_sha(sha):
        raise CloneError(f"resolved non-immutable SHA {sha!r} for {repo_path}")
    return sha


def ensure_pinned_checkout(
    seed: SeedRepo,
    *,
    dest_root: Path,
    prefer_local: bool = True,
    depth: int = 50,
) -> PinnedCheckout:
    """Ensure seed is available locally, pinned to an immutable SHA.

    Prefer local fixtures when present (offline). Otherwise clone ``seed.repo``
    from GitHub and checkout ``seed.base_commit`` (must be SHA when keep-bound).
    """
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    if prefer_local:
        local = seed.resolve_local_path()
        if local is not None and local.is_dir():
            # Local fixtures may use synthetic green SHAs — accept if marked
            # hex-like; otherwise invent a deterministicized copy path with the
            # declared pin as the task base_commit.
            pin = seed.base_commit.strip()
            if not is_immutable_sha(pin):
                raise CloneError(
                    f"local seed {seed.seed_id!r} declares non-immutable "
                    f"base_commit={pin!r}; fix allowlist pin before keep"
                )
            # Copy into dest so later mutations are isolated.
            case = dest_root / f"{seed.seed_id}_pin"
            if case.exists():
                shutil.rmtree(case)
            shutil.copytree(
                local,
                case,
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
            return PinnedCheckout(
                seed_id=seed.seed_id,
                repo=seed.repo,
                path=case,
                base_commit=pin,
                source="local_fixture",
            )

    if not seed.repo or "/" not in seed.repo:
        raise CloneError(f"seed {seed.seed_id!r} has no cloneable repo id")

    pin = seed.base_commit.strip()
    if not is_immutable_sha(pin):
        raise CloneError(
            f"seed {seed.seed_id!r} base_commit={pin!r} is not an immutable SHA; "
            "pin a full commit before certified keep"
        )

    clone_dir = dest_root / f"{seed.seed_id}_clone"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    url = f"https://github.com/{seed.repo}.git"
    # Try shallow clone of the pin first; fall back to deeper history if needed.
    clone = _run_git(
        [
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            f"--depth={max(1, depth)}",
            url,
            str(clone_dir),
        ]
    )
    if clone.returncode != 0:
        # Retry without depth filter (some pins may be invisible shallow).
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        clone = _run_git(["clone", "--filter=blob:none", "--no-checkout", url, str(clone_dir)])
        if clone.returncode != 0:
            raise CloneError(
                f"git clone failed for {seed.repo}: {(clone.stderr or clone.stdout or '').strip()}"
            )

    # Fetch the pin explicitly.
    fetch = _run_git(["fetch", "--depth", str(max(1, depth)), "origin", pin], cwd=clone_dir)
    if fetch.returncode != 0:
        fetch = _run_git(["fetch", "origin", pin], cwd=clone_dir)
    if fetch.returncode != 0:
        # Last resort: full fetch of origin + checkout may still work if pin is
        # an ancestor already present.
        _run_git(["fetch", "--unshallow"], cwd=clone_dir)

    checkout = _run_git(["checkout", "--force", pin], cwd=clone_dir)
    if checkout.returncode != 0:
        raise CloneError(
            f"git checkout {pin} failed for {seed.repo}: "
            f"{(checkout.stderr or checkout.stdout or '').strip()}"
        )

    full = resolve_full_sha(clone_dir)
    return PinnedCheckout(
        seed_id=seed.seed_id,
        repo=seed.repo,
        path=clone_dir,
        base_commit=full,
        source="clone",
    )


__all__ = [
    "CloneError",
    "PinnedCheckout",
    "ensure_pinned_checkout",
    "is_immutable_sha",
    "resolve_full_sha",
]
