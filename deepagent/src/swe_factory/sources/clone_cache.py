"""Cached git clones for M9 scale mining / envbuild.

Reuses bare or working-tree clones under a durable cache root so mining
≥70 candidates does not re-clone the same public remotes repeatedly.

Safety:
- Never touches off-limits docker names (cache is filesystem only).
- Cache keys are seed id + sanitized repo slug.
- Existing directories with a valid ``.git`` are refreshed via ``git fetch`` when
  possible; corrupt trees are rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_factory.sources.allowlist import SeedRepo
from swe_factory.sources.clone import CloneError, is_immutable_sha, resolve_full_sha

_REPO_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_DEFAULT_CACHE_ROOT = Path("/tmp/sdf-clone-cache")
_INDEX_NAME = "clone_cache_index.json"

# Module-level lock map so parallel envbuild/mine share the same seed safely.
_locks_guard = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def default_clone_cache_root() -> Path:
    """Return the default durable clone cache root (under /tmp)."""
    return _DEFAULT_CACHE_ROOT


def _path_lock(key: str) -> threading.Lock:
    with _locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _path_locks[key] = lock
        return lock


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def cache_key_for(seed: SeedRepo | str, *, repo: str | None = None) -> str:
    """Stable filesystem-safe cache key for a seed or repo slug."""
    if isinstance(seed, SeedRepo):
        seed_id = seed.seed_id
        repo_slug = seed.repo
    else:
        seed_id = str(seed)
        repo_slug = repo or str(seed)
    slug = _REPO_SLUG_RE.sub("_", (repo_slug or seed_id).strip().lower())[:80]
    digest = hashlib.sha1(f"{seed_id}:{repo_slug}".encode()).hexdigest()[:10]
    safe_seed = _REPO_SLUG_RE.sub("_", seed_id.strip())[:60] or "seed"
    return f"{safe_seed}__{slug}__{digest}"


@dataclass
class CloneCacheEntry:
    """One cached clone row."""

    key: str
    seed_id: str
    repo: str
    repository_url: str
    path: str
    last_fetch_at: str = ""
    head_sha: str = ""
    hits: int = 0
    source: str = "clone"  # clone | reused | local_fixture
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CloneCacheStats:
    """Cache hit / miss funnel counters."""

    hits: int = 0
    misses: int = 0
    refreshes: int = 0
    rebuilds: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CloneCache:
    """Filesystem clone cache with per-key locks and index JSON."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        depth: int = 80,
    ) -> None:
        self.root = Path(root) if root is not None else default_clone_cache_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.depth = max(1, int(depth))
        self.stats = CloneCacheStats()
        self._index_path = self.root / _INDEX_NAME
        self._index: dict[str, dict[str, Any]] = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self._index_path.is_file():
            return {}
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        entries = raw.get("entries")
        return dict(entries) if isinstance(entries, dict) else {}

    def _save_index(self) -> None:
        payload = {
            "entries": self._index,
            "stats": self.stats.to_dict(),
            "updated_at": datetime.now(UTC).isoformat(),
            "root": str(self.root),
        }
        self._index_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def entry_path(self, key: str) -> Path:
        return self.root / key

    def get_entry(self, key: str) -> CloneCacheEntry | None:
        row = self._index.get(key)
        if not row:
            return None
        try:
            return CloneCacheEntry(
                key=str(row.get("key") or key),
                seed_id=str(row.get("seed_id") or ""),
                repo=str(row.get("repo") or ""),
                repository_url=str(row.get("repository_url") or ""),
                path=str(row.get("path") or self.entry_path(key)),
                last_fetch_at=str(row.get("last_fetch_at") or ""),
                head_sha=str(row.get("head_sha") or ""),
                hits=int(row.get("hits") or 0),
                source=str(row.get("source") or "clone"),
                meta=dict(row.get("meta") or {}),
            )
        except (TypeError, ValueError):
            return None

    def list_entries(self) -> list[CloneCacheEntry]:
        out: list[CloneCacheEntry] = []
        for key in sorted(self._index):
            entry = self.get_entry(key)
            if entry is not None:
                out.append(entry)
        return out

    def ensure_seed(
        self,
        seed: SeedRepo,
        *,
        refresh: bool = True,
        prefer_local: bool = False,
    ) -> CloneCacheEntry:
        """Ensure clone for *seed* exists under the cache; return entry.

        Thread-safe per key. Refreshes with ``git fetch`` when *refresh* and
        a prior clone exists.
        """
        key = cache_key_for(seed)
        lock = _path_lock(key)
        with lock:
            return self._ensure_seed_unlocked(
                seed,
                key=key,
                refresh=refresh,
                prefer_local=prefer_local,
            )

    def _ensure_seed_unlocked(
        self,
        seed: SeedRepo,
        *,
        key: str,
        refresh: bool,
        prefer_local: bool,
    ) -> CloneCacheEntry:
        dest = self.entry_path(key)

        if prefer_local:
            local = seed.resolve_local_path()
            if local is not None and local.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(
                    local,
                    dest,
                    ignore=shutil.ignore_patterns(
                        "__pycache__",
                        "*.pyc",
                        ".venv",
                        "node_modules",
                        ".pytest_cache",
                        "*.egg-info",
                    ),
                )
                entry = CloneCacheEntry(
                    key=key,
                    seed_id=seed.seed_id,
                    repo=seed.repo,
                    repository_url=seed.repository_url,
                    path=str(dest),
                    last_fetch_at=datetime.now(UTC).isoformat(),
                    head_sha=seed.base_commit if is_immutable_sha(seed.base_commit) else "",
                    hits=1,
                    source="local_fixture",
                )
                self.stats.misses += 1
                self._record(entry)
                return entry

        if dest.is_dir() and (dest / ".git").exists():
            self.stats.hits += 1
            if refresh:
                fetch = _run_git(
                    ["fetch", "--depth", str(self.depth), "origin"],
                    cwd=dest,
                    timeout=300,
                )
                if fetch.returncode != 0:
                    _run_git(["fetch", "origin"], cwd=dest, timeout=600)
                self.stats.refreshes += 1
            try:
                head = resolve_full_sha(dest)
            except CloneError:
                head = ""
            prev = self.get_entry(key)
            hits = (prev.hits + 1) if prev else 1
            entry = CloneCacheEntry(
                key=key,
                seed_id=seed.seed_id,
                repo=seed.repo,
                repository_url=seed.repository_url,
                path=str(dest),
                last_fetch_at=datetime.now(UTC).isoformat(),
                head_sha=head,
                hits=hits,
                source="reused",
            )
            self._record(entry)
            return entry

        # Rebuild
        if dest.exists():
            shutil.rmtree(dest)
            self.stats.rebuilds += 1

        url = seed.repository_url
        if not url.startswith("http"):
            url = f"https://github.com/{seed.repo}.git"
        if not url.endswith(".git") and "github.com" in url:
            url = url.rstrip("/") + ".git"

        self.stats.misses += 1
        clone = _run_git(
            [
                "clone",
                "--filter=blob:none",
                f"--depth={self.depth}",
                "--no-single-branch",
                url,
                str(dest),
            ],
            timeout=600,
        )
        if clone.returncode != 0:
            if dest.exists():
                shutil.rmtree(dest)
            clone = _run_git(
                ["clone", "--filter=blob:none", url, str(dest)],
                timeout=900,
            )
            if clone.returncode != 0:
                self.stats.errors += 1
                raise CloneError(
                    f"cache clone failed for {seed.repo}: "
                    f"{(clone.stderr or clone.stdout or '').strip()[:400]}"
                )

        try:
            head = resolve_full_sha(dest)
        except CloneError:
            head = ""
        entry = CloneCacheEntry(
            key=key,
            seed_id=seed.seed_id,
            repo=seed.repo,
            repository_url=seed.repository_url,
            path=str(dest),
            last_fetch_at=datetime.now(UTC).isoformat(),
            head_sha=head,
            hits=1,
            source="clone",
        )
        self._record(entry)
        return entry

    def _record(self, entry: CloneCacheEntry) -> None:
        self._index[entry.key] = entry.to_dict()
        self._save_index()

    def stats_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "entry_count": len(self._index),
            **self.stats.to_dict(),
        }

    def materialize_worktree(
        self,
        seed: SeedRepo,
        dest: Path | str,
        *,
        base_commit: str | None = None,
        refresh: bool = False,
    ) -> Path:
        """Copy (or hardlink-free tree-copy) a cached clone into *dest* @ optional pin."""
        entry = self.ensure_seed(seed, refresh=refresh)
        src = Path(entry.path)
        dest_path = Path(dest)
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(
            src,
            dest_path,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".venv",
                "node_modules",
                ".pytest_cache",
            ),
        )
        pin = (base_commit or seed.base_commit or "").strip()
        if pin and is_immutable_sha(pin) and (dest_path / ".git").exists():
            checkout = _run_git(["checkout", "--force", pin], cwd=dest_path)
            if checkout.returncode != 0:
                fetch = _run_git(["fetch", "origin", pin], cwd=dest_path, timeout=300)
                if fetch.returncode == 0:
                    _run_git(["checkout", "--force", pin], cwd=dest_path)
        return dest_path


def ensure_cached_clone(
    seed: SeedRepo,
    *,
    cache_root: Path | str | None = None,
    refresh: bool = True,
    prefer_local: bool = False,
) -> CloneCacheEntry:
    """Module-level helper: ensure clone in default or provided cache root."""
    cache = CloneCache(root=cache_root)
    return cache.ensure_seed(seed, refresh=refresh, prefer_local=prefer_local)


def load_cache_stats(cache_root: Path | str | None = None) -> Mapping[str, Any]:
    """Read cache index stats without opening clones."""
    cache = CloneCache(root=cache_root)
    return cache.stats_dict()


__all__ = [
    "CloneCache",
    "CloneCacheEntry",
    "CloneCacheStats",
    "cache_key_for",
    "default_clone_cache_root",
    "ensure_cached_clone",
    "load_cache_stats",
]
