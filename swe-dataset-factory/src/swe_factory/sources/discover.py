"""Real-repo candidate discover for DeepSWE mining (M7+).

Authority split (architecture non-negotiable):
- **Git** (local clone/fetch) owns history, blobs, multi-file diffs, base SHAs.
- **Oxylabs** may be used only for github.com **HTTP** metadata (pages/raw) via
  ``source=universal`` — never as a substitute for git history.

Discovery rules (VAL-MINE-*):
1. Multi-file: ≥2 non-test product source files + ≥1 test path (VAL-MINE-001)
2. ``base_commit`` full 40-char hex SHA (VAL-MINE-002)
3. Copyleft licenses fatal (VAL-MINE-003)
4. Offline fixture path remains network-free (VAL-MINE-004)
5. Hybrid curated still needs real HTTPS ``repository_url`` + real SHA (VAL-MINE-005)
6. Gold = source-only multi-file; tests → held-out ``test.patch`` (VAL-MINE-006)
7. Live history via git clone/diff (VAL-MINE-007)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from swe_factory.oracle.gates import MULTI_FILE_FLOOR, count_files_in_patch
from swe_factory.producers.pr_miner import (
    PrFileChange,
    PrMineError,
    build_problem_statement,
    extract_gold_from_files,
    is_test_path,
    multi_file_source_filter,
    offline_fixture_pr,
    wrap_file_diff,
)
from swe_factory.sources.clone import CloneError, is_immutable_sha
from swe_factory.sources.license_gate import (
    LicenseGateError,
    assert_permissive_license,
)

# Full 40-char hex required for DeepSWE keep / discover (stricter than short SHA).
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)

CandidateKind = Literal["real_pr", "curated", "offline_fixture"]
RejectReason = str

# Synthetic/motor markers that cannot enter certified DeepSWE keep state.
_FAKE_REPO_PREFIXES = (
    "file://",
    "fixtures/",
    "fixture/",
    "local/",
    "synthetic/",
)
_FAKE_SHA_PREFIXES = (
    "fixture",
    "a1000",
    "b1000",
    "c1000",
    "0000000",
)


class DiscoverError(RuntimeError):
    """Raised when discover cannot construct a certifiable candidate."""


@dataclass(frozen=True, slots=True)
class DiscoverCandidate:
    """Sanitized discover output destined for envbuild / export staging.

    Always carries multi-file gold, held-out test patch (when tests changed),
    a full 40-char ``base_commit``, and license + repository_url provenance.
    """

    candidate_id: str
    kind: CandidateKind
    repository_url: str
    repo: str
    base_commit: str
    language: str
    license: str
    title: str
    problem_statement: str
    source_files: tuple[str, ...]
    test_files: tuple[str, ...]
    gold_patch: str
    test_patch: str
    gold_files: tuple[str, ...]
    history_authority: Literal["git", "offline_fixture", "api_metadata_only"]
    http_metadata_source: Literal["none", "oxylabs", "github_api"] = "none"
    merge_commit_sha: str | None = None
    pr_number: int | None = None
    html_url: str = ""
    reject_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.reject_reason is None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        # keep tuples JSON-serializable
        row["source_files"] = list(self.source_files)
        row["test_files"] = list(self.test_files)
        row["gold_files"] = list(self.gold_files)
        return row


@dataclass(frozen=True, slots=True)
class DiscoverReject:
    """Audit row for a rejected candidate (file under-size, license, SHA, …)."""

    repo: str
    reason_code: str
    detail: str
    repository_url: str = ""
    base_commit: str | None = None
    pr_number: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiscoverReport:
    """Funnel summary for a discover run."""

    kept: tuple[DiscoverCandidate, ...]
    rejected: tuple[DiscoverReject, ...]
    provider_calls: int = 0
    network_required: bool = False
    offline: bool = False
    history_authority: str = "git"

    @property
    def keep_count(self) -> int:
        return len(self.kept)

    @property
    def reject_count(self) -> int:
        return len(self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "keep_count": self.keep_count,
            "reject_count": self.reject_count,
            "provider_calls": self.provider_calls,
            "network_required": self.network_required,
            "offline": self.offline,
            "history_authority": self.history_authority,
            "kept": [c.to_dict() for c in self.kept],
            "rejected": [r.to_dict() for r in self.rejected],
            "reject_reasons": _tally_reasons(self.rejected),
        }


def _tally_reasons(rows: Sequence[DiscoverReject]) -> dict[str, int]:
    tallies: dict[str, int] = {}
    for row in rows:
        tallies[row.reason_code] = tallies.get(row.reason_code, 0) + 1
    return tallies


def is_full_sha(value: str | None) -> bool:
    """True when value is an immutable full 40-char hex commit SHA."""
    if not value:
        return False
    cleaned = value.strip()
    return bool(_FULL_SHA_RE.fullmatch(cleaned))


def require_full_sha(value: str | None, *, field_name: str = "base_commit") -> str:
    """Return stripped 40-char lowercase hex or raise DiscoverError."""
    cleaned = (value or "").strip().lower()
    if not is_full_sha(cleaned):
        raise DiscoverError(f"{field_name} must be a full 40-char hex SHA; got {value!r}")
    # Prefer lowercase hex for durable pins.
    return cleaned


def _looks_like_motor_identity(value: str | None) -> bool:
    """True for harbor motor / hybrid fixture tree identities."""
    if not value:
        return False
    lower = str(value).strip().lower()
    markers = (
        "fixtures/harbor_motors",
        "harbor_motors/",
        "fixtures/tiny_green",
        "python_orders",
        "go_kvstore",
        "ts_registry",
        "orderlib/",
    )
    return any(m in lower for m in markers)


def is_real_repository_url(url: str | None) -> bool:
    """True for public HTTPS (or git@) host repo URLs, false for fixture/file."""
    if not url or not str(url).strip():
        return False
    raw = str(url).strip()
    lower = raw.lower()
    if any(lower.startswith(p) for p in _FAKE_REPO_PREFIXES):
        return False
    if lower.startswith("git@"):
        # git@github.com:owner/name.git
        return ":" in raw and "/" in raw.split(":", 1)[-1]
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc or not parsed.path or parsed.path in {"/", ""}:
        return False
    # Reject local hosts
    host = (parsed.hostname or "").lower()
    return not (host in {"localhost", "127.0.0.1", "0.0.0.0"} or host.endswith(".local"))


def repository_url_for(repo: str, *, override: str | None = None) -> str:
    """Normalize owner/name or URL into a canonical HTTPS clone URL."""
    if override and is_real_repository_url(override):
        return override.rstrip("/")
    cleaned = (repo or "").strip().removesuffix(".git")
    if is_real_repository_url(cleaned):
        return cleaned.rstrip("/")
    if cleaned.startswith("https://github.com/") or cleaned.startswith("http://github.com/"):
        return cleaned.rstrip("/")
    if "/" in cleaned and not cleaned.startswith("fixtures/"):
        owner, name = cleaned.split("/", 1)
        owner, name = owner.strip(), name.strip().removesuffix(".git")
        if owner and name and not any(ch in owner + name for ch in " \t"):
            return f"https://github.com/{owner}/{name}"
    # Fixture / non-remote identifiers stay as non-real markers
    return f"file://{cleaned}" if cleaned else ""


def looks_like_fake_sha(value: str | None) -> bool:
    """Heuristic: synthetic harbor motor / offline fixture SHAs."""
    if not value:
        return True
    cleaned = value.strip().lower()
    if not is_full_sha(cleaned) and not is_immutable_sha(cleaned):
        return True
    return any(cleaned.startswith(p) for p in _FAKE_SHA_PREFIXES)


def extract_test_patch_from_files(files: Sequence[PrFileChange]) -> str:
    """Build held-out unified test.patch from PR test-path changes only.

    VAL-MINE-006: solution/gold omits tests; verifier receives test.patch.
    """
    chunks: list[str] = []
    for change in files:
        if not change.is_test:
            continue
        # Keep removals / empty hunks out of held-out suite.
        if not change.patch or not change.patch.strip():
            continue
        chunks.append(wrap_file_diff(change.path, change.patch, status=change.status).rstrip("\n"))
    if not chunks:
        return ""
    body = "\n".join(chunks)
    return body if body.endswith("\n") else body + "\n"


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_numstat(output: str) -> list[tuple[str, int, int]]:
    """Parse ``git diff --numstat`` lines → (path, added, deleted)."""
    rows: list[tuple[str, int, int]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        add_s, del_s, path = parts[0], parts[1], parts[2]
        try:
            added = 0 if add_s == "-" else int(add_s)
            deleted = 0 if del_s == "-" else int(del_s)
        except ValueError:
            added, deleted = 0, 0
        path = path.strip()
        if path:
            rows.append((path, added, deleted))
    return rows


def parse_name_status(output: str) -> list[tuple[str, str]]:
    """Parse ``git diff --name-status`` → (status, path)."""
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0].strip()
        # renames: R100\told\tnew — take the new path
        path = parts[-1].strip()
        if path:
            rows.append((status, path))
    return rows


def git_show_file_patch(
    repo_path: Path,
    base: str,
    head: str,
    path: str,
) -> str | None:
    """Return unified diff body for one path via git (history authority)."""
    proc = _run_git(
        ["diff", "--unified=3", f"{base}..{head}", "--", path],
        cwd=repo_path,
    )
    if proc.returncode != 0:
        return None
    text = proc.stdout or ""
    if not text.strip():
        return None
    return text if text.endswith("\n") else text + "\n"


def collect_pr_file_changes_from_git(
    repo_path: Path,
    *,
    base: str,
    head: str,
) -> list[PrFileChange]:
    """Build PrFileChange list from local git history (VAL-MINE-007).

    Uses ``git diff base..head`` only — never Oxylabs — for patch bodies.
    """
    if not Path(repo_path).is_dir():
        raise DiscoverError(f"git repo path missing: {repo_path}")
    base_sha = require_full_sha(base, field_name="base")
    # head may be short if resolved; expand if possible
    head_proc = _run_git(["rev-parse", head], cwd=repo_path)
    if head_proc.returncode != 0:
        raise DiscoverError(
            f"cannot resolve head {head!r} in {repo_path}: "
            f"{(head_proc.stderr or head_proc.stdout or '').strip()}"
        )
    head_sha = head_proc.stdout.strip()
    if not is_full_sha(head_sha) and not is_immutable_sha(head_sha):
        raise DiscoverError(f"resolved head is not a SHA: {head_sha!r}")

    status_proc = _run_git(
        ["diff", "--name-status", f"{base_sha}..{head_sha}"],
        cwd=repo_path,
    )
    if status_proc.returncode != 0:
        raise DiscoverError(
            f"git diff --name-status failed: "
            f"{(status_proc.stderr or status_proc.stdout or '').strip()}"
        )
    num_proc = _run_git(
        ["diff", "--numstat", f"{base_sha}..{head_sha}"],
        cwd=repo_path,
    )
    num_map = {
        path: (added, deleted) for path, added, deleted in parse_numstat(num_proc.stdout or "")
    }

    status_map = {path: status for status, path in parse_name_status(status_proc.stdout or "")}
    changes: list[PrFileChange] = []
    for path, status in status_map.items():
        if not path:
            continue
        added, deleted = num_map.get(path, (0, 0))
        # Map git status letter → pr-style status
        st = status.upper()
        if st.startswith("A"):
            pr_status = "added"
        elif st.startswith("D"):
            pr_status = "removed"
        elif st.startswith("R"):
            pr_status = "renamed"
        else:
            pr_status = "modified"
        raw_patch = git_show_file_patch(repo_path, base_sha, head_sha, path)
        # git diff already includes headers; store as-is or None
        patch_body = raw_patch
        # If patch already has headers, extract_gold_from_files/wrap is idempotent
        # via wrap_file_diff which detects "diff --git".
        changes.append(
            PrFileChange(
                path=path,
                status=pr_status,
                patch=patch_body,
                additions=added,
                deletions=deleted,
            )
        )
    return changes


def discover_merge_range_from_git(
    repo_path: Path,
    *,
    base: str,
    head: str,
    repo: str,
    repository_url: str | None = None,
    license: str,
    language: str | None = None,
    title: str = "",
    body: str = "",
    pr_number: int | None = None,
    html_url: str = "",
    merge_commit_sha: str | None = None,
    min_source_files: int = MULTI_FILE_FLOOR,
    require_tests: bool = True,
    kind: CandidateKind = "real_pr",
) -> DiscoverCandidate:
    """Discover one candidate from local git range base..head.

    Enforces multi-file floor, full SHA pin, license gate, held-out tests.
    """
    base_sha = require_full_sha(base)
    license_decision = assert_permissive_license(license, repo=repo)
    url = repository_url_for(repo, override=repository_url)
    if kind == "real_pr" and (_looks_like_motor_identity(repo) or _looks_like_motor_identity(url)):
        raise DiscoverError(
            f"motor_or_hybrid_rejected: real_pr git discover refuses motor/fixture "
            f"identity repo={repo!r} url={url!r} (VAL-RPR-002/005)"
        )
    # Hybrid/curated real path must not accept fake motors
    if kind == "curated":
        validate_hybrid_curated(repository_url=url, base_commit=base_sha, license=license)

    try:
        files = collect_pr_file_changes_from_git(repo_path, base=base_sha, head=head)
    except CloneError as exc:
        raise DiscoverError(str(exc)) from exc
    except DiscoverError:
        raise

    if not multi_file_source_filter(
        files,
        min_source_files=min_source_files,
        require_tests=require_tests,
    ):
        sources = sorted({f.path for f in files if f.is_source})
        tests = sorted({f.path for f in files if f.is_test})
        raise DiscoverError(
            f"multi-file/tests floor failed for {repo} "
            f"(sources={sources}, tests={tests}, min_source={min_source_files})"
        )

    gold = extract_gold_from_files(files)
    gold_files = tuple(count_files_in_patch(gold))
    if len(gold_files) < min_source_files:
        raise DiscoverError(
            f"gold multi-file floor failed: gold_files={list(gold_files)} (min={min_source_files})"
        )
    # Gold must not include test paths
    for gpath in gold_files:
        if is_test_path(gpath):
            raise DiscoverError(f"gold_patch must be source-only; found test path {gpath!r}")

    test_patch = extract_test_patch_from_files(files)
    test_files = tuple(sorted({f.path for f in files if f.is_test}))
    source_files = tuple(sorted({f.path for f in files if f.is_source}))
    if require_tests and not test_files:
        raise DiscoverError("require_tests=True but no test files in range")
    # When tests changed, held-out patch must be non-empty
    if test_files and not test_patch.strip():
        # Allow empty only when test files have no textual patch (binary); still
        # require ≥1 test path was touched for the filter.
        pass

    lang = language or _language_from_paths(files)
    # Lightweight MergedPR-like shim for problem statement reuse
    from swe_factory.producers.pr_miner import MergedPR

    pr_stub = MergedPR(
        repo=repo,
        number=int(pr_number or 0),
        title=title or f"Range {base_sha[:8]}..{head}",
        body=body or "",
        base_commit=base_sha,
        merge_commit_sha=merge_commit_sha,
        language=lang,
        html_url=html_url or "",
        files=tuple(files),
        license=license_decision.license_raw or license,
    )
    prompt = build_problem_statement(pr=pr_stub, source_files=source_files)

    candidate_id = (
        f"discover__{re.sub(r'[^a-z0-9]+', '_', repo.lower()).strip('_')}"
        f"__{base_sha[:12]}__{uuid.uuid4().hex[:8]}"
    )
    return DiscoverCandidate(
        candidate_id=candidate_id,
        kind=kind,
        repository_url=url,
        repo=repo,
        base_commit=base_sha,
        language=lang,
        license=license_decision.license_raw or license,
        title=pr_stub.title,
        problem_statement=prompt,
        source_files=source_files,
        test_files=test_files,
        gold_patch=gold,
        test_patch=test_patch,
        gold_files=gold_files,
        history_authority="git",
        http_metadata_source="none",
        merge_commit_sha=merge_commit_sha,
        pr_number=pr_number,
        html_url=html_url or "",
        meta={
            "license_normalized": license_decision.normalized,
            "license_reason": license_decision.reason_code,
            "produced_at": datetime.now(UTC).isoformat(),
            "git_base": base_sha,
            "git_head": head,
        },
    )


def _language_from_paths(files: Sequence[PrFileChange], default: str = "python") -> str:
    counts: dict[str, int] = {}
    for change in files:
        if not change.is_source and not change.is_test:
            continue
        suffix = Path(change.path).suffix.lower()
        if suffix == ".py":
            lang = "python"
        elif suffix in {".js", ".jsx"}:
            lang = "javascript"
        elif suffix in {".ts", ".tsx"}:
            lang = "typescript"
        elif suffix == ".go":
            lang = "go"
        elif suffix == ".rs":
            lang = "rust"
        else:
            continue
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda kv: kv[1])[0]


def validate_hybrid_curated(
    *,
    repository_url: str,
    base_commit: str,
    license: str | None = None,
) -> None:
    """Ensure curated/long-horizon tasks still carry real repo + real SHA.

    VAL-MINE-005: motors with file:// or fake SHAs cannot enter certified keep.
    """
    if not is_real_repository_url(repository_url):
        raise DiscoverError(
            f"hybrid curated candidate requires real public repository_url; got {repository_url!r}"
        )
    if not is_full_sha(base_commit) or looks_like_fake_sha(base_commit):
        raise DiscoverError(
            f"hybrid curated candidate requires real full 40-char base_commit; got {base_commit!r}"
        )
    if license is not None:
        assert_permissive_license(license, repo=repository_url)


def sanitize_api_candidate(
    *,
    repo: str,
    base_commit: str,
    files: Sequence[PrFileChange] | Sequence[Mapping[str, Any]],
    license: str,
    repository_url: str | None = None,
    language: str | None = None,
    title: str = "",
    body: str = "",
    pr_number: int | None = None,
    html_url: str = "",
    merge_commit_sha: str | None = None,
    min_source_files: int = MULTI_FILE_FLOOR,
    require_tests: bool = True,
    kind: CandidateKind = "real_pr",
    history_authority: Literal["git", "offline_fixture", "api_metadata_only"] = "api_metadata_only",
    http_metadata_source: Literal["none", "oxylabs", "github_api"] = "github_api",
) -> DiscoverCandidate:
    """Sanitize an API/HTTP-metadata candidate (patches already in file payloads).

    Still requires full 40-char base, multi-file floor, license, held-out tests.
    Prefer :func:`discover_merge_range_from_git` when a local clone exists so git
    remains history authority for gold/test patches.

    For ``kind=real_pr`` (product path): harbor motors / fixture trees are
    rejected (VAL-RPR-002 / VAL-RPR-005). Use ``kind=curated`` only for
    documented hybrid validation paths — never product ship.
    """
    from swe_factory.producers.pr_miner import _as_file_change

    base_sha = require_full_sha(base_commit)
    license_decision = assert_permissive_license(license, repo=repo)
    url = repository_url_for(repo, override=repository_url)

    if kind == "real_pr":
        # Product real_pr path: never accept hybrid motor fixture trees.
        motor_hit = _looks_like_motor_identity(repo) or _looks_like_motor_identity(url)
        if motor_hit:
            raise DiscoverError(
                f"motor_or_hybrid_rejected: real_pr discover refuses motor/fixture "
                f"identity repo={repo!r} url={url!r} (VAL-RPR-002/005)"
            )
        if not is_real_repository_url(url):
            # Offline fixture uses kind=offline_fixture, not real_pr.
            raise DiscoverError(f"real_pr discover requires public repository_url; got {url!r}")

    if kind == "curated":
        validate_hybrid_curated(repository_url=url, base_commit=base_sha, license=license)

    changes = tuple(
        item if isinstance(item, PrFileChange) else _as_file_change(dict(item)) for item in files
    )

    if not multi_file_source_filter(
        changes,
        min_source_files=min_source_files,
        require_tests=require_tests,
    ):
        sources = sorted({c.path for c in changes if c.is_source})
        tests = sorted({c.path for c in changes if c.is_test})
        raise DiscoverError(
            f"multi-file/tests floor failed for {repo} (sources={sources}, tests={tests})"
        )

    gold = extract_gold_from_files(changes)
    gold_files = tuple(count_files_in_patch(gold))
    if len(gold_files) < min_source_files:
        raise DiscoverError(f"gold multi-file floor failed: {list(gold_files)}")
    for gpath in gold_files:
        if is_test_path(gpath):
            raise DiscoverError(f"gold_patch must be source-only; found test path {gpath!r}")

    test_patch = extract_test_patch_from_files(changes)
    test_files = tuple(sorted({c.path for c in changes if c.is_test}))
    source_files = tuple(sorted({c.path for c in changes if c.is_source}))
    lang = language or _language_from_paths(changes)

    from swe_factory.producers.pr_miner import MergedPR

    pr_stub = MergedPR(
        repo=repo,
        number=int(pr_number or 0),
        title=title or f"PR {pr_number or ''}".strip() or "candidate",
        body=body or "",
        base_commit=base_sha,
        merge_commit_sha=merge_commit_sha,
        language=lang,
        html_url=html_url or "",
        files=changes,
        license=license_decision.license_raw or license,
    )
    prompt = build_problem_statement(pr=pr_stub, source_files=source_files)
    candidate_id = (
        f"discover__{re.sub(r'[^a-z0-9]+', '_', repo.lower()).strip('_')}"
        f"__{pr_number or base_sha[:12]}__{uuid.uuid4().hex[:8]}"
    )
    return DiscoverCandidate(
        candidate_id=candidate_id,
        kind=kind,
        repository_url=url,
        repo=repo,
        base_commit=base_sha,
        language=lang,
        license=license_decision.license_raw or license,
        title=pr_stub.title,
        problem_statement=prompt,
        source_files=source_files,
        test_files=test_files,
        gold_patch=gold,
        test_patch=test_patch,
        gold_files=gold_files,
        history_authority=history_authority,
        http_metadata_source=http_metadata_source,
        merge_commit_sha=merge_commit_sha,
        pr_number=pr_number,
        html_url=html_url or "",
        meta={
            "license_normalized": license_decision.normalized,
            "license_reason": license_decision.reason_code,
            "produced_at": datetime.now(UTC).isoformat(),
        },
    )


def discover_offline_fixture(
    *,
    work_root: Path | None = None,
) -> DiscoverReport:
    """Offline discover path: multi-file fixture, no network (VAL-MINE-004).

    Emits candidate structure with gold source-only patch and held-out
    test.patch. Base SHA form remains full 40 hex-compatible for schema checks
    (fixture prefix is allowed only on this offline path).
    """
    pr = offline_fixture_pr()
    # Offline fixture uses a 40-char synthetic pin (not a live git object).
    # Live/certified discover still requires [0-9a-f]{40} via is_full_sha.
    base = pr.base_commit.strip()
    if len(base) != 40:
        raise DiscoverError(f"offline fixture base_commit must be 40 chars; got {base!r}")

    gold = extract_gold_from_files(pr.files)
    gold_files = tuple(count_files_in_patch(gold))
    test_patch = extract_test_patch_from_files(pr.files)
    if not test_patch.strip():
        raise DiscoverError("offline fixture must emit non-empty held-out test.patch")
    if len(gold_files) < MULTI_FILE_FLOOR:
        raise DiscoverError("offline fixture gold must be multi-file")

    # Offline licenses are MIT permissive
    assert_permissive_license(pr.license, repo=pr.repo)

    prompt = build_problem_statement(pr=pr)
    candidate = DiscoverCandidate(
        candidate_id=f"discover_offline__{pr.number}__{uuid.uuid4().hex[:8]}",
        kind="offline_fixture",
        repository_url=f"https://github.com/{pr.repo}",  # structural URL, fixture id
        repo=pr.repo,
        base_commit=base,
        language=pr.language,
        license=pr.license,
        title=pr.title,
        problem_statement=prompt,
        source_files=pr.source_files,
        test_files=pr.test_files,
        gold_patch=gold,
        test_patch=test_patch,
        gold_files=gold_files,
        history_authority="offline_fixture",
        http_metadata_source="none",
        merge_commit_sha=pr.merge_commit_sha,
        pr_number=pr.number,
        html_url=pr.html_url,
        meta={
            "offline": True,
            "provider_calls": 0,
            "network_required": False,
            "fixture": "tiny_real_pr",
            "note": (
                "offline fixture is not a certified deepswe keep; proves discover wiring only"
            ),
        },
    )

    if work_root is not None:
        write_candidate_artifacts(candidate, Path(work_root))

    return DiscoverReport(
        kept=(candidate,),
        rejected=(),
        provider_calls=0,
        network_required=False,
        offline=True,
        history_authority="offline_fixture",
    )


def write_candidate_artifacts(candidate: DiscoverCandidate, out_dir: Path) -> Path:
    """Write gold.patch, test.patch, candidate.json under out_dir/<id>/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = out_dir / candidate.candidate_id
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True, exist_ok=True)
    (case / "gold.patch").write_text(candidate.gold_patch, encoding="utf-8")
    (case / "solution.patch").write_text(candidate.gold_patch, encoding="utf-8")
    (case / "test.patch").write_text(candidate.test_patch or "", encoding="utf-8")
    (case / "candidate.json").write_text(
        json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (case / "instruction.md").write_text(
        candidate.problem_statement.rstrip() + "\n",
        encoding="utf-8",
    )
    return case


def try_discover_from_files(
    *,
    repo: str,
    base_commit: str,
    files: Sequence[PrFileChange] | Sequence[Mapping[str, Any]],
    license: str,
    **kwargs: Any,
) -> tuple[DiscoverCandidate | None, DiscoverReject | None]:
    """Non-raising sanitize wrapper returning keep or reject audit row."""
    try:
        kept = sanitize_api_candidate(
            repo=repo,
            base_commit=base_commit,
            files=files,
            license=license,
            **kwargs,
        )
        return kept, None
    except (DiscoverError, LicenseGateError, PrMineError) as exc:
        reason = getattr(exc, "reason_code", None) or _reason_from_message(str(exc))
        return None, DiscoverReject(
            repo=repo,
            reason_code=reason,
            detail=str(exc),
            repository_url=repository_url_for(repo, override=kwargs.get("repository_url")),
            base_commit=base_commit,
            pr_number=kwargs.get("pr_number"),
            meta={"error_type": type(exc).__name__},
        )


def _reason_from_message(message: str) -> str:
    lower = message.lower()
    if "motor_or_hybrid" in lower or "harbor motor" in lower:
        return "motor_or_hybrid_rejected"
    if "copyleft" in lower or "license" in lower:
        if "copyleft" in lower:
            return "license_copyleft_rejected"
        if "missing" in lower:
            return "license_missing_rejected"
        return "license_rejected"
    if "40-char" in lower or "full" in lower and "sha" in lower:
        return "base_commit_not_full_sha"
    if "multi-file" in lower or "floor" in lower:
        return "multi_file_floor_rejected"
    if "source-only" in lower or "test path" in lower:
        return "gold_includes_tests"
    if "repository_url" in lower or "hybrid" in lower or "curated" in lower:
        return "hybrid_requires_real_repo"
    if "fake" in lower:
        return "fake_sha_or_repo_rejected"
    return "discover_rejected"


def build_local_git_repo_with_range(
    dest: Path,
    *,
    base_files: Mapping[str, str],
    head_files: Mapping[str, str],
) -> tuple[str, str]:
    """Create an ephemeral git repo with base + head commits for offline tests.

    Returns ``(base_sha, head_sha)`` as full 40-char hex.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
        return _run_git(
            ["-c", "user.email=test@example.com", "-c", "user.name=test", *args], cwd=dest
        )

    init = _git(["init"])
    if init.returncode != 0:
        raise DiscoverError(f"git init failed: {(init.stderr or init.stdout or '').strip()}")
    _git(["checkout", "-b", "main"])

    for path, content in base_files.items():
        full = dest / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    _git(["add", "-A"])
    commit_base = _git(["commit", "-m", "base"])
    if commit_base.returncode != 0:
        raise DiscoverError(
            f"base commit failed: {(commit_base.stderr or commit_base.stdout or '').strip()}"
        )
    base_sha_proc = _git(["rev-parse", "HEAD"])
    base_sha = base_sha_proc.stdout.strip()

    # Apply head tree
    for path, content in head_files.items():
        full = dest / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    # Remove files present only on base when not in head
    for path in base_files:
        if path not in head_files:
            target = dest / path
            if target.exists():
                target.unlink()
    _git(["add", "-A"])
    commit_head = _git(["commit", "-m", "head"])
    # empty commit if tree unchanged is fine for tests that expect filter rejects later
    if commit_head.returncode != 0 and "nothing to commit" not in (
        commit_head.stdout + commit_head.stderr
    ):
        raise DiscoverError(
            f"head commit failed: {(commit_head.stderr or commit_head.stdout or '').strip()}"
        )
    head_sha_proc = _git(["rev-parse", "HEAD"])
    head_sha = head_sha_proc.stdout.strip()
    if not is_full_sha(base_sha) or not is_full_sha(head_sha):
        raise DiscoverError(f"local test repo produced non-full SHAs {base_sha!r}/{head_sha!r}")
    return base_sha, head_sha


__all__ = [
    "DiscoverCandidate",
    "DiscoverError",
    "DiscoverReject",
    "DiscoverReport",
    "build_local_git_repo_with_range",
    "collect_pr_file_changes_from_git",
    "discover_merge_range_from_git",
    "discover_offline_fixture",
    "extract_test_patch_from_files",
    "is_full_sha",
    "is_real_repository_url",
    "looks_like_fake_sha",
    "repository_url_for",
    "require_full_sha",
    "sanitize_api_candidate",
    "try_discover_from_files",
    "validate_hybrid_curated",
    "write_candidate_artifacts",
]
