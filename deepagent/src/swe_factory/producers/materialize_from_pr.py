"""Materialize live accepted PR candidates into a durable materials tree.

Bridge from live mine (candidates / MergedPR) → ship-compatible materials root:

    materials_root/
      inventory.json
      {task_id}/
        solution.patch
        test.patch
        meta.json

The product materials root must **not** be ``fixtures/real_pr_ship`` (that
shortlist stays engineering/unit-only). Default live root is
``datasets/live_materials``.

Offline path: inject DictGitHubTransport via GitHubClient / MergedPR objects.
Live path: GitHub REST via PrMiner.fetch_merged_pr.

VAL-LMAT-001 / VAL-LMAT-004 / VAL-LMINE-008.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_factory.producers.hard_filter import measure_source_hunk_count
from swe_factory.producers.pr_miner import (
    MergedPR,
    PrMineError,
    PrMiner,
    RealPrCandidate,
    extract_gold_from_files,
)
from swe_factory.sources.discover import (
    extract_test_patch_from_files,
    is_full_sha,
    repository_url_for,
)
from swe_factory.sources.github import GitHubClient, GitHubError

# Product / live materials default — deliberately outside fixtures/real_pr_ship.
DEFAULT_LIVE_MATERIALS_ROOT = Path("datasets/live_materials")

# Engineering-only shortlist that must never be the live product materials authority.
FIXTURE_MATERIALS_MARKER = "fixtures/real_pr_ship"

_INVENTORY_NAME = "inventory.json"
_META_NAME = "meta.json"
_SOLUTION_NAME = "solution.patch"
_TEST_NAME = "test.patch"

# Fields expected by ship_real_pr.load_real_pr_materials + VAL-LMAT inventory.
INVENTORY_REQUIRED_FIELDS: tuple[str, ...] = (
    "task_id",
    "repo",
    "pr",
    "base",
    "language",
    "url",
    "license",
    "src",
    "tests",
    "title",
    "materials_dir",
)


class MaterializeError(RuntimeError):
    """Raised when materials-from-PR emission fails closed."""


@dataclass(frozen=True, slots=True)
class MaterializedTask:
    """One durable materials tree entry ready for product ship loaders."""

    task_id: str
    materials_root: str
    materials_dir: str
    repo: str
    pr_number: int
    base_sha: str
    language: str
    license: str
    solution_patch: str
    test_patch: str
    source_files: tuple[str, ...]
    test_files: tuple[str, ...]
    title: str
    inventory_row: dict[str, Any] = field(repr=False)
    # PR markdown/plain description for full agent prompts (VAL-DPRMPT-001).
    body: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "materials_root": self.materials_root,
            "materials_dir": self.materials_dir,
            "repo": self.repo,
            "pr": self.pr_number,
            "base": self.base_sha,
            "language": self.language,
            "license": self.license,
            "title": self.title,
            "body": self.body,
            "source_files": list(self.source_files),
            "test_files": list(self.test_files),
            "solution_bytes": len(self.solution_patch),
            "test_bytes": len(self.test_patch),
            "inventory_row": dict(self.inventory_row),
        }


@dataclass(frozen=True, slots=True)
class MaterializeReport:
    """Batch materialize result for accepted live candidates."""

    materials_root: str
    inventory_path: str
    tasks: tuple[MaterializedTask, ...]
    rejected: tuple[dict[str, Any], ...] = ()
    product_materials: bool = True
    engineering_fixture: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "materials_root": self.materials_root,
            "inventory_path": self.inventory_path,
            "count": len(self.tasks),
            "task_ids": [t.task_id for t in self.tasks],
            "rejected": list(self.rejected),
            "product_materials": self.product_materials,
            "engineering_fixture": self.engineering_fixture,
            "tasks": [t.to_dict() for t in self.tasks],
        }


def materials_task_id(repo: str, pr_number: int) -> str:
    """Stable task_id matching fixture shortlist style: realpr-{basename}-{pr}."""
    raw = (repo or "").strip().removesuffix(".git")
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower() or "repo"
    return f"realpr-{slug}-{int(pr_number)}"


def is_fixture_materials_root(path: Path | str | None) -> bool:
    """True when *path* resolves under or equals fixtures/real_pr_ship."""
    if path is None:
        return False
    text = str(path).replace("\\", "/")
    # Exact marker segment (unit fixtures only — never product live root).
    return FIXTURE_MATERIALS_MARKER in text or text.rstrip("/").endswith("real_pr_ship")


def assert_non_fixture_materials_root(path: Path | str) -> Path:
    """Fail closed if caller tries to use fixture shortlist as live materials root.

    VAL-LMINE-008 / VAL-LMAT-001: product live materials must live outside
    fixtures/real_pr_ship so ship can load a non-fixture inventory.
    """
    root = Path(path).expanduser()
    if is_fixture_materials_root(root):
        raise MaterializeError(
            f"materials root must not be the engineering fixture shortlist "
            f"({FIXTURE_MATERIALS_MARKER}); got {root}. "
            f"Use {DEFAULT_LIVE_MATERIALS_ROOT} (or another non-fixture path) "
            "for live materialize (VAL-LMAT-001 / VAL-LMINE-008)."
        )
    return root


def inventory_row_for_task(
    *,
    task_id: str,
    repo: str,
    pr_number: int,
    base_sha: str,
    language: str,
    license: str,
    url: str,
    source_files: Sequence[str],
    test_files: Sequence[str],
    title: str,
    materials_dir: str,
    source_hunk_count: int | None = None,
    discovery_path: str | None = None,
    body: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one inventory.json row (repo/pr/sha/lang + ship loader fields)."""
    row: dict[str, Any] = {
        "task_id": task_id,
        "repo": repo,
        "url": url,
        "language": language,
        "license": license,
        "pr": int(pr_number),
        "base": base_sha,
        "src": list(source_files),
        "tests": list(test_files),
        "title": title,
        "materials_dir": materials_dir,
    }
    if source_hunk_count is not None:
        row["source_hunk_count"] = int(source_hunk_count)
    if discovery_path:
        row["discovery_path"] = discovery_path
    # Optional PR body for DeepSWE-style full prompts (VAL-DPRMPT-001).
    if body is not None:
        row["body"] = str(body)
    if extra:
        for key, value in extra.items():
            if key not in row and value is not None:
                row[key] = value
    for field_name in INVENTORY_REQUIRED_FIELDS:
        if field_name not in row:
            raise MaterializeError(f"inventory row missing required field {field_name!r}")
    return row


def read_inventory(materials_root: Path | str) -> list[dict[str, Any]]:
    """Load inventory.json list (empty if missing)."""
    root = Path(materials_root)
    inv_path = root / _INVENTORY_NAME
    if not inv_path.is_file():
        return []
    raw = json.loads(inv_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise MaterializeError(f"inventory must be a JSON list at {inv_path}")
    return [row for row in raw if isinstance(row, dict)]


def write_inventory(materials_root: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Rewrite inventory.json stably sorted by task_id."""
    root = Path(materials_root)
    root.mkdir(parents=True, exist_ok=True)
    inv_path = root / _INVENTORY_NAME
    ordered = sorted((dict(r) for r in rows), key=lambda r: str(r.get("task_id") or ""))
    inv_path.write_text(json.dumps(ordered, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return inv_path


def list_task_dirs(materials_root: Path | str) -> list[Path]:
    """Return child directories under materials_root that look like task slots."""
    root = Path(materials_root)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        # Skip hidden / cache dirs
        if child.name.startswith(".") or child.name.startswith("_"):
            continue
        out.append(child)
    return out


def inventory_row_from_task_dir(
    task_dir: Path | str, *, materials_root: Path | str | None = None
) -> dict[str, Any] | None:
    """Build one inventory row from an on-disk task directory (meta + patches).

    Returns None when the directory lacks minimal ship-loadable patches
    (non-empty solution.patch + test.patch) or cannot derive required fields.
    Used to rebuild inventory.json when it was truncated anten vs task dirs.
    """
    tdir = Path(task_dir)
    if not tdir.is_dir():
        return None
    sol_path = tdir / _SOLUTION_NAME
    test_path = tdir / _TEST_NAME
    if not sol_path.is_file() or not test_path.is_file():
        return None
    try:
        sol = sol_path.read_text(encoding="utf-8", errors="replace")
        test_body = test_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not sol.strip() or not test_body.strip():
        return None

    meta: dict[str, Any] = {}
    meta_path = tdir / _META_NAME
    if meta_path.is_file():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = raw
        except (OSError, json.JSONDecodeError):
            meta = {}

    tid = str(meta.get("task_id") or tdir.name).strip()
    if not tid:
        return None
    repo = str(meta.get("repo") or "").strip()
    pr_raw = meta.get("pr", meta.get("pr_number"))
    base = str(meta.get("base") or meta.get("base_commit") or "").strip().lower()
    language = str(meta.get("language") or "python").strip().lower() or "python"
    license_name = str(meta.get("license") or "MIT").strip() or "MIT"
    title = str(meta.get("title") or f"PR #{pr_raw or tid}").strip()
    body = str(meta.get("body") or meta.get("pr_body") or "").strip()
    src = [str(p) for p in (meta.get("src") or meta.get("source_files") or []) if str(p).strip()]
    tests = [str(p) for p in (meta.get("tests") or meta.get("test_files") or []) if str(p).strip()]
    url = str(meta.get("url") or meta.get("repository_url") or "").strip()
    if not url and repo:
        url = repository_url_for(repo)
        if url.startswith("https://github.com/") and not url.endswith(".git"):
            url = url + ".git"
    if not repo and url:
        # Derive owner/name from github URL when meta partial
        cleaned = url.rstrip("/").removesuffix(".git")
        if "github.com/" in cleaned:
            repo = cleaned.split("github.com/", 1)[-1]
    # Fail soft if missing critical identity (PR#. base SHA still preferred).
    try:
        pr_number = int(pr_raw) if pr_raw is not None else 0
    except (TypeError, ValueError):
        pr_number = 0
    if not base:
        # Cannot ship without pin; omit from inventory
        return None
    if not is_full_sha(base):
        return None

    materials_dir = str(meta.get("materials_dir") or "")
    if not materials_dir:
        root = Path(materials_root) if materials_root is not None else tdir.parent
        try:
            materials_dir = str(tdir.relative_to(Path.cwd()))
        except ValueError:
            materials_dir = str(tdir)
        # Prefer relative-to-root style matching materialize_merged_pr
        if not materials_dir.startswith(str(root)) and (root / tid).is_dir():
            materials_dir = (
                str(root / tid)
                if Path(materials_dir).is_absolute()
                else str(Path(root.name) / tid if root.name else tdir)
            )

    # Prefer canonical relative path used elsewhere: <root>/<task_id>
    root_for_path = Path(materials_root) if materials_root is not None else tdir.parent
    materials_dir = str(root_for_path / tid)

    hunk_raw = meta.get("source_hunk_count")
    if hunk_raw is None:
        hunk_count = sum(1 for line in sol.splitlines() if line.startswith("@@")) or None
    else:
        try:
            hunk_count = int(hunk_raw)
        except (TypeError, ValueError):
            hunk_count = None

    extra: dict[str, Any] = {}
    for key in ("discovery_path", "live_mined", "product_n_evidence", "merged_at", "html_url"):
        if key in meta and meta[key] is not None:
            extra[key] = meta[key]

    try:
        row = inventory_row_for_task(
            task_id=tid,
            repo=repo or "unknown/unknown",
            pr_number=pr_number,
            base_sha=base,
            language=language,
            license=license_name,
            url=url or f"https://github.com/{repo or 'unknown/unknown'}.git",
            source_files=src,
            test_files=tests,
            title=title,
            materials_dir=materials_dir,
            source_hunk_count=hunk_count,
            discovery_path=str(meta.get("discovery_path") or "") or None,
            body=body or None,
            extra=extra or None,
        )
    except MaterializeError:
        return None
    return row


def rebuild_inventory_from_task_dirs(
    materials_root: Path | str,
    *,
    merge_existing: bool = True,
    write: bool = True,
) -> list[dict[str, Any]]:
    """Rebuild inventory.json so it covers every ship-loadable task directory.

    Root cause (m14 under-yield): ship ``load_real_pr_materials`` trusts a
    possibly **truncated** inventory.json when present and never scans sibling
    task dirs. Materialize may leave dirs with full patches/meta while only a
    subset landed in inventory (partial wave, overwrite, race).

    This rebuild:
    1. Scans every child task dir with non-empty solution.patch + test.patch.
    2. Optionally merges existing inventory rows (existing wins only when
       the task dir is missing; dir-derived row wins when both exist).
    3. Writes the complete inventory when ``write=True``.

    Returns the rebuilt row list (sorted by task_id).
    """
    root = Path(materials_root)
    if not root.is_dir():
        if write:
            write_inventory(root, [])
        return []

    by_id: dict[str, dict[str, Any]] = {}
    if merge_existing:
        for row in read_inventory(root):
            tid = str(row.get("task_id") or "").strip()
            if tid:
                by_id[tid] = dict(row)

    recovered = 0
    for tdir in list_task_dirs(root):
        dir_row = inventory_row_from_task_dir(tdir, materials_root=root)
        if dir_row is None:
            continue
        tid = str(dir_row["task_id"])
        # Dir-derived row always supersedes stale inventory (authoritative disk).
        if tid not in by_id:
            recovered += 1
        by_id[tid] = dir_row

    # Drop inventory-only rows whose task dir lacks ship-loadable patches.
    cleaned: list[dict[str, Any]] = []
    for tid, row in by_id.items():
        tdir = root / tid
        sol = tdir / _SOLUTION_NAME
        test_p = tdir / _TEST_NAME
        if sol.is_file() and test_p.is_file():
            try:
                if (
                    sol.read_text(encoding="utf-8", errors="replace").strip()
                    and test_p.read_text(encoding="utf-8", errors="replace").strip()
                ):
                    # Ensure materials_dir points at this tree
                    row = dict(row)
                    row.setdefault("materials_dir", str(root / tid))
                    if not row.get("materials_dir"):
                        row["materials_dir"] = str(root / tid)
                    cleaned.append(row)
            except OSError:
                continue

    ordered = sorted(cleaned, key=lambda r: str(r.get("task_id") or ""))
    if write:
        write_inventory(root, ordered)
    # Attach recovery count onto a module-level readable side channel via list
    # attribute for tests/logs (list is the return; callers may use stats).
    _ = recovered  # clarity for maintainers; tests use inventory_stats/len
    return ordered


def inventory_completeness(
    materials_root: Path | str,
) -> dict[str, Any]:
    """Report inventory vs task-dir coverage (diagnose truncated inventory)."""
    root = Path(materials_root)
    inv_ids = {str(r.get("task_id") or "") for r in read_inventory(root) if r.get("task_id")}
    dir_ids: set[str] = set()
    loadable_dirs: list[str] = []
    for tdir in list_task_dirs(root):
        row = inventory_row_from_task_dir(tdir, materials_root=root)
        if row is not None:
            tid = str(row["task_id"])
            dir_ids.add(tid)
            loadable_dirs.append(tid)
    missing_from_inv = sorted(dir_ids - inv_ids)
    orphan_inv = sorted(inv_ids - dir_ids)
    return {
        "materials_root": str(root),
        "inventory_count": len(inv_ids),
        "loadable_task_dirs": len(dir_ids),
        "missing_from_inventory": missing_from_inv,
        "orphan_inventory_ids": orphan_inv,
        "complete": len(missing_from_inv) == 0 and len(dir_ids) >= len(inv_ids),
        "task_ids_on_disk": sorted(dir_ids),
    }


def _upsert_inventory_row(
    materials_root: Path,
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tid = str(row.get("task_id") or "").strip()
    if not tid:
        raise MaterializeError("inventory row requires string task_id")
    existing = read_inventory(materials_root)
    kept = [r for r in existing if str(r.get("task_id") or "") != tid]
    kept.append(dict(row))
    write_inventory(materials_root, kept)
    return kept


def _normalize_patch(body: str) -> str:
    """Normalize patch body and repair create-file headers for ``git apply``.

    Ensures trailing newline and rewrites legacy pseudo-create headers
    (``--- a/path`` + ``@@ -0,0``) into ``new file mode`` + ``--- /dev/null``.
    """
    from swe_factory.producers.pr_miner import repair_pseudo_create_file_headers

    text = body if isinstance(body, str) else str(body or "")
    if not text.strip():
        return ""
    return repair_pseudo_create_file_headers(text)


def materialize_merged_pr(
    pr: MergedPR,
    materials_root: Path | str | None = None,
    *,
    task_id: str | None = None,
    discovery_path: str | None = None,
    allow_fixture_root: bool = False,
    require_test_patch: bool = True,
    materials_dir_style: str = "relative",
) -> MaterializedTask:
    """Write solution.patch + test.patch + inventory row for one MergedPR.

    Args:
        pr: Accepted merged PR (source + test file changes, patches attached).
        materials_root: Durable root (default ``datasets/live_materials``).
        task_id: Override stable id; default ``realpr-{repo}-{pr}``.
        discovery_path: Optional ``search`` / ``list_pulls`` passthrough onto inventory.
        allow_fixture_root: Test-only escape; product path must leave this False.
        require_test_patch: When True, empty test.diff fails closed (product bridge).
        materials_dir_style: ``relative`` stores path relative to CWD/package root;
            ``absolute`` stores resolved absolute path.
    """
    root = Path(materials_root) if materials_root is not None else DEFAULT_LIVE_MATERIALS_ROOT
    if not allow_fixture_root:
        root = assert_non_fixture_materials_root(root)
    else:
        root = Path(root).expanduser()

    if not pr.base_commit or not str(pr.base_commit).strip():
        raise MaterializeError(f"PR {pr.repo}#{pr.number} missing base_commit for materials")
    base_sha = str(pr.base_commit).strip().lower()
    if not is_full_sha(base_sha):
        # Still allow engineering pins, but live product prefers 40-char (soft warn: pointer).
        # For durable product bridge we fail closed on non-full SHA.
        raise MaterializeError(
            f"PR {pr.repo}#{pr.number} base_commit must be full 40-char hex SHA; "
            f"got {pr.base_commit!r}"
        )

    try:
        solution = _normalize_patch(extract_gold_from_files(pr.files))
        test_patch = _normalize_patch(extract_test_patch_from_files(pr.files))
    except PrMineError as exc:
        raise MaterializeError(str(exc)) from exc

    if not solution.strip():
        raise MaterializeError(
            f"empty solution.patch for supposed keep {pr.repo}#{pr.number} (VAL-LMAT-001)"
        )
    if require_test_patch and not test_patch.strip():
        raise MaterializeError(
            f"empty test.patch for supposed keep {pr.repo}#{pr.number} "
            "(candidate had no usable test-path changes; VAL-LMAT-001)"
        )

    tid = (task_id or materials_task_id(pr.repo, pr.number)).strip()
    if not tid:
        raise MaterializeError("task_id must be non-empty")

    task_dir = root / tid
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / _SOLUTION_NAME).write_text(solution, encoding="utf-8")
    (task_dir / _TEST_NAME).write_text(test_patch, encoding="utf-8")

    url = repository_url_for(pr.repo)
    # Prefer .git form for clone URLs when github HTTPS.
    if url.startswith("https://github.com/") and not url.endswith(".git"):
        url = url + ".git"

    if materials_dir_style == "absolute":
        materials_dir_str = str(task_dir.resolve())
    else:
        materials_dir_str = str(task_dir)

    hunk_count = int(pr.source_hunk_count or 0) or measure_source_hunk_count(pr.files)
    pr_body = (pr.body or "").strip()
    meta = {
        "task_id": tid,
        "repo": pr.repo,
        "url": url,
        "language": pr.language,
        "license": pr.license or "MIT",
        "pr": int(pr.number),
        "base": base_sha,
        "src": list(pr.source_files),
        "tests": list(pr.test_files),
        "title": pr.title or f"PR #{pr.number}",
        "body": pr_body,
        "source_hunk_count": hunk_count,
        "merged_at": pr.merged_at,
        "html_url": pr.html_url,
        "materialized_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "materials_root": str(root),
    }
    if discovery_path:
        meta["discovery_path"] = discovery_path
    (task_dir / _META_NAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    row = inventory_row_for_task(
        task_id=tid,
        repo=pr.repo,
        pr_number=int(pr.number),
        base_sha=base_sha,
        language=pr.language,
        license=pr.license or "MIT",
        url=url,
        source_files=pr.source_files,
        test_files=pr.test_files,
        title=pr.title or f"PR #{pr.number}",
        materials_dir=materials_dir_str,
        source_hunk_count=hunk_count,
        discovery_path=discovery_path,
        body=pr_body,
    )
    _upsert_inventory_row(root, row)

    # Sanity: solution must exist and be non-empty after write.
    sol_path = task_dir / _SOLUTION_NAME
    if not sol_path.is_file() or not sol_path.read_text(encoding="utf-8").strip():
        raise MaterializeError(f"failed to sink solution.patch under {task_dir}")

    return MaterializedTask(
        task_id=tid,
        materials_root=str(root),
        materials_dir=materials_dir_str,
        repo=pr.repo,
        pr_number=int(pr.number),
        base_sha=base_sha,
        language=pr.language,
        license=pr.license or "MIT",
        solution_patch=solution,
        test_patch=test_patch,
        source_files=tuple(pr.source_files),
        test_files=tuple(pr.test_files),
        title=pr.title or f"PR #{pr.number}",
        inventory_row=row,
        body=pr_body,
    )


def materialize_from_candidate(
    candidate: RealPrCandidate,
    materials_root: Path | str | None = None,
    *,
    discovery_path: str | None = None,
    allow_fixture_root: bool = False,
) -> MaterializedTask:
    """Bridge a produced RealPrCandidate into the durable materials tree."""
    return materialize_merged_pr(
        candidate.pr,
        materials_root,
        discovery_path=discovery_path,
        allow_fixture_root=allow_fixture_root,
        require_test_patch=True,
    )


def materialize_from_pr_number(
    client: GitHubClient,
    repo: str,
    number: int,
    materials_root: Path | str | None = None,
    *,
    language: str | None = None,
    license: str | None = None,
    discovery_path: str | None = None,
    product_mode: bool = False,
    allow_fixture_root: bool = False,
) -> MaterializedTask:
    """Fetch one live/mock merged PR via *client* and materialize patches+inventory.

    Offline: inject DictGitHubTransport into *client*.
    Live: use GitHubClient.from_env() with a token.
    """
    miner = PrMiner(client=client, product_mode=product_mode, license=license or "MIT")
    try:
        pr = miner.fetch_merged_pr(
            repo,
            int(number),
            language=language,
            license=license,
            require_full_base_sha=True,
            enforce_license=True,
            product_hard_filter=product_mode,
        )
    except (PrMineError, GitHubError) as exc:
        raise MaterializeError(str(exc)) from exc
    return materialize_merged_pr(
        pr,
        materials_root,
        discovery_path=discovery_path,
        allow_fixture_root=allow_fixture_root,
        require_test_patch=True,
    )


def materialize_accepted_candidates(
    client: GitHubClient | None,
    candidates: Sequence[Mapping[str, Any] | MergedPR],
    materials_root: Path | str | None = None,
    *,
    product_mode: bool = False,
    allow_fixture_root: bool = False,
    stop_on_error: bool = False,
) -> MaterializeReport:
    """Materialize a batch of accepted live candidates into a non-fixture materials tree.

    Each mapping may be a candidates.jsonl row (``repo``, ``pr_number``/``pr``,
    optional ``discovery_path``/``language``/``license``) or a MergedPR.
    """
    root = Path(materials_root) if materials_root is not None else DEFAULT_LIVE_MATERIALS_ROOT
    if not allow_fixture_root:
        root = assert_non_fixture_materials_root(root)

    tasks: list[MaterializedTask] = []
    rejected: list[dict[str, Any]] = []

    for item in candidates:
        try:
            if isinstance(item, MergedPR):
                task = materialize_merged_pr(
                    item,
                    root,
                    allow_fixture_root=allow_fixture_root,
                )
            else:
                row = dict(item)
                repo = str(row.get("repo") or "").strip()
                pr_raw = row.get("pr_number", row.get("pr"))
                if not repo or pr_raw is None:
                    raise MaterializeError(
                        f"candidate row missing repo/pr: keys={sorted(row.keys())}"
                    )
                discovery = str(row.get("discovery_path") or "") or None
                language = str(row.get("language") or "") or None
                license_name = str(row.get("license") or "") or None
                if client is None:
                    raise MaterializeError(
                        "GitHubClient required to materialize mapping candidates "
                        "(or pass MergedPR objects)"
                    )
                task = materialize_from_pr_number(
                    client,
                    repo,
                    int(pr_raw),
                    root,
                    language=language,
                    license=license_name,
                    discovery_path=discovery,
                    product_mode=product_mode,
                    allow_fixture_root=allow_fixture_root,
                )
            tasks.append(task)
        except (MaterializeError, PrMineError, GitHubError, TypeError, ValueError) as exc:
            rej: dict[str, Any] = {
                "disposition": "reject",
                "reason": "materialize_failed",
                "detail": str(exc),
            }
            if isinstance(item, Mapping):
                rej["repo"] = item.get("repo")
                rej["pr_number"] = item.get("pr_number", item.get("pr"))
            elif isinstance(item, MergedPR):
                rej["repo"] = item.repo
                rej["pr_number"] = item.number
            rejected.append(rej)
            if stop_on_error:
                raise MaterializeError(str(exc)) from exc

    if not tasks:
        raise MaterializeError(
            f"no materials materialized under {root} "
            f"(rejected={len(rejected)}); empty materials root fails VAL-LMAT-001"
        )

    # Atomic completeness: after every batch write, rebuild inventory so it
    # covers every ship-loadable task dir (not just this batch upserts).
    rebuild_inventory_from_task_dirs(root, merge_existing=True, write=True)
    inv_path = root / _INVENTORY_NAME
    return MaterializeReport(
        materials_root=str(root),
        inventory_path=str(inv_path),
        tasks=tuple(tasks),
        rejected=tuple(rejected),
        product_materials=not is_fixture_materials_root(root),
        engineering_fixture=is_fixture_materials_root(root),
    )


# Re-export helper for callers that only need inventory statistics.
def inventory_stats(materials_root: Path | str) -> dict[str, Any]:
    """Summary of a materials tree (count, languages, fixture flag)."""
    root = Path(materials_root)
    rows = read_inventory(root) if root.is_dir() else []
    langs = sorted({str(r.get("language") or "") for r in rows if r.get("language")})
    completeness: dict[str, Any]
    if root.is_dir():
        completeness = inventory_completeness(root)
    else:
        completeness = {
            "complete": True,
            "missing_from_inventory": [],
            "loadable_task_dirs": 0,
        }
    missing_raw = completeness.get("missing_from_inventory") or []
    missing_list = list(missing_raw) if isinstance(missing_raw, list) else list(missing_raw or [])
    loadable_raw = completeness.get("loadable_task_dirs") or 0
    return {
        "materials_root": str(root),
        "count": len(rows),
        "languages": langs,
        "is_fixture_root": is_fixture_materials_root(root),
        "inventory_path": str(root / _INVENTORY_NAME),
        "task_ids": [str(r.get("task_id")) for r in rows if r.get("task_id")],
        "complete": bool(completeness.get("complete")),
        "missing_from_inventory": missing_list,
        "loadable_task_dirs": int(loadable_raw),
    }


__all__ = [
    "DEFAULT_LIVE_MATERIALS_ROOT",
    "FIXTURE_MATERIALS_MARKER",
    "INVENTORY_REQUIRED_FIELDS",
    "MaterializeError",
    "MaterializeReport",
    "MaterializedTask",
    "assert_non_fixture_materials_root",
    "inventory_completeness",
    "inventory_row_for_task",
    "inventory_row_from_task_dir",
    "inventory_stats",
    "is_fixture_materials_root",
    "list_task_dirs",
    "materialize_accepted_candidates",
    "materialize_from_candidate",
    "materialize_from_pr_number",
    "materialize_merged_pr",
    "materials_task_id",
    "read_inventory",
    "rebuild_inventory_from_task_dirs",
    "write_inventory",
]
