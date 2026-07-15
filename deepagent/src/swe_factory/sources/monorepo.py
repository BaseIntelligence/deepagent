"""Monorepo detection + skip gate for M9 funnel hardening.

Large multi-package workspaces (pnpm/lerna/nx turbo, go.work, Cargo workspaces,
multi-setup.py / multi-package.json trees) produce unreliable single-pack
envbuild yield and burn scale budget. When markers exceed thresholds we emit
``monorepo_skip`` with a documented reason rather than silent soft-fail.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.sources.skip_reasons import (
    SKIP_MONOREPO,
    SKIP_REPO_TOO_LARGE,
    SkipReason,
    describe_skip_reason,
)

# Workspace / multi-package markers (file names relative to repo root).
_WORKSPACE_MARKER_FILES: tuple[str, ...] = (
    "pnpm-workspace.yaml",
    "pnpm-workspace.yml",
    "lerna.json",
    "nx.json",
    "turbo.json",
    "rush.json",
    "go.work",
    "Cargo.toml",  # may be workspace root; inspected for [workspace]
    "package.json",  # inspected for workspaces field when many packages present
)

# Subdir package indicators counted when recursive scan is bounded.
_PACKAGE_INDICATORS: tuple[str, ...] = (
    "package.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Cargo.toml",
    "go.mod",
)

# Default thresholds for scale ≥70 envbuild (fail soft → skip, not crash).
DEFAULT_MAX_PACKAGE_MARKERS = 8
DEFAULT_MAX_TRACKED_FILES = 25_000
DEFAULT_MAX_TREE_BYTES = 800 * 1024 * 1024  # 800 MiB work-tree estimate

# Directories never walked when counting package indicators.
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        "target",
        "vendor",
        ".turbo",
        ".nx",
        ".yarn",
        ".pnpm-store",
    }
)


@dataclass(frozen=True, slots=True)
class MonorepoSignal:
    """Signals collected from a tree indicating multi-package layout."""

    markers_found: tuple[str, ...] = ()
    package_indicator_count: int = 0
    package_indicator_paths: tuple[str, ...] = ()
    has_explicit_workspace: bool = False
    tracked_file_estimate: int = 0
    tree_bytes_estimate: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "markers_found": list(self.markers_found),
            "package_indicator_count": self.package_indicator_count,
            "package_indicator_paths": list(self.package_indicator_paths[:40]),
            "has_explicit_workspace": self.has_explicit_workspace,
            "tracked_file_estimate": self.tracked_file_estimate,
            "tree_bytes_estimate": self.tree_bytes_estimate,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class MonorepoDecision:
    """Skip / keep decision for monorepo + size gates."""

    skip: bool
    reason_code: str = ""
    detail: str = ""
    signal: MonorepoSignal = field(default_factory=MonorepoSignal)

    def to_skip_reason(
        self,
        *,
        stage: str = "mine",
        repo: str = "",
        candidate_id: str = "",
    ) -> SkipReason | None:
        if not self.skip:
            return None
        code = self.reason_code or SKIP_MONOREPO
        return SkipReason(
            code=code,
            detail=self.detail or describe_skip_reason(code),
            stage=stage,
            repo=repo,
            candidate_id=candidate_id,
            meta=self.signal.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skip": self.skip,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "documentation": describe_skip_reason(self.reason_code) if self.reason_code else "",
            "signal": self.signal.to_dict(),
        }


def _read_text_prefix(path: Path, *, limit: int = 4096) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _cargo_is_workspace(path: Path) -> bool:
    text = _read_text_prefix(path)
    return "[workspace]" in text


def _package_json_has_workspaces(path: Path) -> bool:
    text = _read_text_prefix(path)
    # Cheap structural check — avoid full json parse for huge roots.
    return '"workspaces"' in text or "'workspaces'" in text


def scan_monorepo_signals(
    root: Path | str,
    *,
    max_walk_files: int = 50_000,
) -> MonorepoSignal:
    """Inspect *root* for multi-package / monorepo markers (bounded walk)."""
    root_path = Path(root)
    if not root_path.is_dir():
        return MonorepoSignal(notes=("root missing or not a directory",))

    markers: list[str] = []
    notes: list[str] = []
    explicit = False

    for name in _WORKSPACE_MARKER_FILES:
        candidate = root_path / name
        if not candidate.is_file():
            continue
        if name == "Cargo.toml":
            if _cargo_is_workspace(candidate):
                markers.append(name)
                explicit = True
                notes.append("Cargo.toml declares [workspace]")
            continue
        if name == "package.json":
            # Only treat root package.json as marker when workspaces field present;
            # multi-package is counted separately below.
            if _package_json_has_workspaces(candidate):
                markers.append(f"{name}:workspaces")
                explicit = True
                notes.append("package.json declares workspaces")
            continue
        markers.append(name)
        explicit = True

    package_paths: list[str] = []
    file_count = 0
    tree_bytes = 0
    for path in root_path.rglob("*"):
        if file_count >= max_walk_files:
            notes.append(f"walk truncated at {max_walk_files} entries")
            break
        # Skip heavy/vendor trees
        parts = set(path.parts)
        if parts & _SKIP_DIR_NAMES:
            continue
        if path.is_dir():
            continue
        file_count += 1
        with contextlib.suppress(OSError):
            tree_bytes += int(path.stat().st_size)
        if path.name in _PACKAGE_INDICATORS:
            # Store path relative to root when possible
            try:
                rel = str(path.relative_to(root_path))
            except ValueError:
                rel = str(path)
            package_paths.append(rel)

    return MonorepoSignal(
        markers_found=tuple(markers),
        package_indicator_count=len(package_paths),
        package_indicator_paths=tuple(package_paths[:80]),
        has_explicit_workspace=explicit,
        tracked_file_estimate=file_count,
        tree_bytes_estimate=tree_bytes,
        notes=tuple(notes),
    )


def evaluate_monorepo_gate(
    root: Path | str | None = None,
    *,
    signal: MonorepoSignal | None = None,
    max_package_markers: int = DEFAULT_MAX_PACKAGE_MARKERS,
    max_tracked_files: int = DEFAULT_MAX_TRACKED_FILES,
    max_tree_bytes: int = DEFAULT_MAX_TREE_BYTES,
    force_skip: bool = False,
) -> MonorepoDecision:
    """Decide whether to skip a candidate repo as monorepo / oversized.

    Order of severity:
    1. size budget (repo_too_large)
    2. explicit workspace markers or package-count over threshold (monorepo_skip)
    """
    if force_skip:
        sig = signal or MonorepoSignal(notes=("force_skip=True",))
        return MonorepoDecision(
            skip=True,
            reason_code=SKIP_MONOREPO,
            detail="forced monorepo skip for scale funnel policy",
            signal=sig,
        )

    sig = signal or (scan_monorepo_signals(Path(root)) if root is not None else MonorepoSignal())

    if sig.tracked_file_estimate > max_tracked_files or sig.tree_bytes_estimate > max_tree_bytes:
        detail = (
            f"repo exceeds size budget: files≈{sig.tracked_file_estimate} "
            f"(max {max_tracked_files}), bytes≈{sig.tree_bytes_estimate} "
            f"(max {max_tree_bytes})"
        )
        return MonorepoDecision(
            skip=True,
            reason_code=SKIP_REPO_TOO_LARGE,
            detail=detail,
            signal=sig,
        )

    if sig.has_explicit_workspace and (sig.package_indicator_count > 1 or bool(sig.markers_found)):
        detail = (
            "explicit multi-package workspace markers: "
            + ", ".join(sig.markers_found or ("(workspace)",))
            + f"; package_indicators={sig.package_indicator_count}"
        )
        return MonorepoDecision(
            skip=True,
            reason_code=SKIP_MONOREPO,
            detail=detail,
            signal=sig,
        )

    if sig.package_indicator_count > max_package_markers:
        detail = (
            f"package indicators ({sig.package_indicator_count}) exceed "
            f"modular threshold ({max_package_markers})"
        )
        return MonorepoDecision(
            skip=True,
            reason_code=SKIP_MONOREPO,
            detail=detail,
            signal=sig,
        )

    return MonorepoDecision(skip=False, signal=sig)


def paths_look_monorepo(paths: Sequence[str]) -> MonorepoDecision:
    """Lightweight path-list monorepo check (no filesystem walk).

    Useful when only changed paths / gold file lists are available.
    """
    lowered = [p.replace("\\", "/").lower() for p in paths]
    hits: list[str] = []
    for marker in (
        "pnpm-workspace.yaml",
        "pnpm-workspace.yml",
        "lerna.json",
        "nx.json",
        "turbo.json",
        "go.work",
        "rush.json",
    ):
        if any(p.endswith(marker) or f"/{marker}" in f"/{p}" for p in lowered):
            hits.append(marker)
    # Many package.json / go.mod under different top-level dirs
    tops: set[str] = set()
    for p in lowered:
        parts = [x for x in p.split("/") if x and x != "."]
        if not parts:
            continue
        if parts[-1] in {
            "package.json",
            "go.mod",
            "cargo.toml",
            "pyproject.toml",
            "setup.py",
        }:
            tops.add(parts[0] if len(parts) > 1 else parts[0])
    sig = MonorepoSignal(
        markers_found=tuple(hits),
        package_indicator_count=len(tops),
        package_indicator_paths=tuple(sorted(tops))[:40],
        has_explicit_workspace=bool(hits),
        notes=("path-list monorepo heuristic",),
    )
    if hits or len(tops) > DEFAULT_MAX_PACKAGE_MARKERS:
        return MonorepoDecision(
            skip=True,
            reason_code=SKIP_MONOREPO,
            detail=(f"path-list monorepo heuristic: markers={hits!r} package_tops={len(tops)}"),
            signal=sig,
        )
    return MonorepoDecision(skip=False, signal=sig)


__all__ = [
    "DEFAULT_MAX_PACKAGE_MARKERS",
    "DEFAULT_MAX_TRACKED_FILES",
    "DEFAULT_MAX_TREE_BYTES",
    "MonorepoDecision",
    "MonorepoSignal",
    "evaluate_monorepo_gate",
    "paths_look_monorepo",
    "scan_monorepo_signals",
]
