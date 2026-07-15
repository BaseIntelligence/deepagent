"""Real PR miner: multi-file merged PRs with tests → labeled ``real_pr`` tasks.

Select multi-file merged PRs that touch both source and test files, pin
``base_commit`` to the PR base SHA (pre-merge), extract gold as the unified
patch, derive F2P/P2P or applicable test-command split, label
``source_track=real_pr``, and run oracle (stub or certified).

Offline path uses injectable GitHub transport + optional local base fixtures.
Live path uses GITHUB_TOKEN when present (never logged).
"""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_factory.oracle.docker_run import OracleRunnerBackend
from swe_factory.oracle.gates import (
    MULTI_FILE_FLOOR,
    GateResult,
    append_gate_audit,
    count_files_in_patch,
    run_certified_gates_for_task,
    run_stub_gates,
)
from swe_factory.producers.hard_filter import (
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
    SOFT_MULTI_FILE_FLOOR,
    evaluate_product_hard_filter,
    measure_source_hunk_count,
)
from swe_factory.schema import VALID_SOURCE_TRACKS, EnvironmentMeta, SourceTrack, TaskRecord
from swe_factory.sources.github import GitHubClient, GitHubError
from swe_factory.sources.license_gate import LicenseGateError, assert_permissive_license

_DEFAULT_IMAGE_DIGEST = "sha256:real_pr_pending"

_TEST_PATH_HINTS = (
    "/tests/",
    "/test/",
    "/__tests__/",
    "/spec/",
    "test_",
    "_test.",
    ".test.",
    ".spec.",
    "conftest.py",
    "tests.py",
)

_SOURCE_EXTS = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".cs",
        ".kt",
        ".swift",
    }
)

_SKIP_PATH_PARTS = frozenset(
    {
        "vendor",
        "node_modules",
        "dist",
        "build",
        ".git",
        "__pycache__",
        ".venv",
        "coverage",
    }
)


class PrMineError(RuntimeError):
    """Raised when real_pr candidate construction fails."""


@dataclass(frozen=True, slots=True)
class PrFileChange:
    """One file change from a PR with optional patch body."""

    path: str
    status: str
    patch: str | None = None
    additions: int = 0
    deletions: int = 0

    @property
    def is_test(self) -> bool:
        return is_test_path(self.path)

    @property
    def is_source(self) -> bool:
        return is_source_path(self.path) and not self.is_test


@dataclass(frozen=True, slots=True)
class MergedPR:
    """A merged multi-file PR candidate after selection filters."""

    repo: str
    number: int
    title: str
    body: str
    base_commit: str
    merge_commit_sha: str | None
    language: str
    html_url: str
    files: tuple[PrFileChange, ...]
    license: str = "MIT"
    merged_at: str | None = None
    source_hunk_count: int = 0

    @property
    def source_files(self) -> tuple[str, ...]:
        return tuple(sorted({f.path for f in self.files if f.is_source}))

    @property
    def test_files(self) -> tuple[str, ...]:
        return tuple(sorted({f.path for f in self.files if f.is_test}))

    @property
    def files_touched(self) -> tuple[str, ...]:
        return tuple(sorted({f.path for f in self.files if f.path}))


@dataclass(frozen=True, slots=True)
class RealPrCandidate:
    """Labeled real_pr task with gold provenance metadata."""

    task: TaskRecord
    pr: MergedPR
    gold_files: tuple[str, ...]
    workspace: Path | None
    provenance: dict[str, Any]
    gates: GateResult | None = None
    provider_calls: int = 0
    test_patch: str = ""
    repository_url: str = ""

    @property
    def source_track(self) -> str:
        track = self.task.source_track
        return track.value if hasattr(track, "value") else str(track)


def is_test_path(path: str) -> bool:
    """Heuristic: path looks like a test / spec fixture file."""
    norm = path.replace("\\", "/")
    lower = norm.lower()
    name = Path(norm).name.lower()
    if name in {"conftest.py", "tests.py", "test.js", "test.ts"}:
        return True
    if name.endswith("_test.py") or name.endswith("_test.go"):
        return True
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if ".test." in name or ".spec." in name:
        return True
    for h in _TEST_PATH_HINTS:
        if h in f"/{lower}/" or h in lower:
            # avoid over-matching plain "test" inside package names except path segments
            if h in {"test_", "_test.", ".test.", ".spec."}:
                if h in name:
                    return True
            elif h.strip("/") in lower.split("/") or h.startswith("/") and h in f"/{lower}/":
                return True
    return False


def is_source_path(path: str) -> bool:
    """True if path is a non-skip source-code file we care about."""
    if not path or path.endswith("/"):
        return False
    parts = Path(path.replace("\\", "/")).parts
    if any(p in _SKIP_PATH_PARTS for p in parts):
        return False
    suffix = Path(path).suffix.lower()
    return suffix in _SOURCE_EXTS


def multi_file_source_filter(
    files: Sequence[PrFileChange | dict[str, Any]],
    *,
    min_source_files: int = MULTI_FILE_FLOOR,
    require_tests: bool = True,
) -> bool:
    """Accept PR when ≥min_source_files non-test sources touch (+ optional tests).

    VAL expected: multi-file PR filter rejects single-file or test-only PRs.
    """
    changes = [_as_file_change(item) for item in files]
    sources = {c.path for c in changes if c.is_source}
    tests = {c.path for c in changes if c.is_test}
    if len(sources) < min_source_files:
        return False
    return not (require_tests and not tests)


def _as_file_change(item: PrFileChange | dict[str, Any]) -> PrFileChange:
    if isinstance(item, PrFileChange):
        return item
    path = str(item.get("filename") or item.get("path") or "")
    status = str(item.get("status") or "modified")
    patch = item.get("patch")
    patch_s = patch if isinstance(patch, str) else None
    return PrFileChange(
        path=path,
        status=status,
        patch=patch_s,
        additions=int(item.get("additions") or 0),
        deletions=int(item.get("deletions") or 0),
    )


_ADD_STATUSES = frozenset({"added", "add", "new", "created"})
_CREATE_HUNK_RE = re.compile(r"^@@ -0(?:,0)? \+\d+(?:,\d+)? @@", re.MULTILINE)


def _looks_like_create_hunks(hunks: str) -> bool:
    """True when a GitHub-style hunk body is a pure new-file create (``@@ -0,0``)."""
    body = hunks or ""
    if not body.strip():
        return False
    # Pure create hunks start with @@ -0,0 (optionally -0) and only + lines thereafter.
    if not _CREATE_HUNK_RE.search(body):
        return False
    # Reject if any other (non-create) hunk is present.
    for line in body.splitlines():
        if line.startswith("@@ ") and not _CREATE_HUNK_RE.match(line):
            return False
    return True


def wrap_file_diff(
    path: str,
    hunks: str,
    *,
    status: str | None = None,
) -> str:
    """Wrap GitHub's header-less per-file hunks into a git-applyable diff.

    GitHub REST's ``files[].patch`` omits ``diff --git`` / file headers. For
    ``status=added`` (and for pure ``@@ -0,0`` create hunks) we must emit
    ``new file mode`` + ``--- /dev/null`` so ``git apply`` can create missing
    paths. Emitting ``--- a/<path>`` for creates makes apply fail with
    ``No such file or directory`` inside HarborDockerVerifier (while the pure
    dual-run fallback may still succeed).
    """
    cleaned_path = (path or "").strip().lstrip("./")
    if not cleaned_path:
        raise PrMineError("wrap_file_diff requires a non-empty path")
    body = hunks if hunks.endswith("\n") else hunks + "\n"
    if body.lstrip().startswith("diff --git"):
        # Repair already-wrapped pseudo-create headers (legacy materials).
        return repair_pseudo_create_file_headers(body)
    status_l = (status or "").strip().lower()
    is_add = status_l in _ADD_STATUSES or _looks_like_create_hunks(body)
    if is_add:
        return (
            f"diff --git a/{cleaned_path} b/{cleaned_path}\n"
            f"new file mode 100644\n"
            f"--- /dev/null\n"
            f"+++ b/{cleaned_path}\n"
            f"{body}"
        )
    return (
        f"diff --git a/{cleaned_path} b/{cleaned_path}\n"
        f"--- a/{cleaned_path}\n"
        f"+++ b/{cleaned_path}\n"
        f"{body}"
    )


def repair_pseudo_create_file_headers(patch_text: str) -> str:
    """Rewrite create-style file diffs that use ``--- a/path`` instead of ``/dev/null``.

    Durable materials emitted before wrap_file_diff respected ``status=added``
    may still carry pseudo-create headers. HarborDocker `git apply` requires
    proper ``new file mode`` + ``--- /dev/null`` for those hunks.
    """
    text = patch_text if isinstance(patch_text, str) else str(patch_text or "")
    if not text.strip():
        return ""
    if not text.endswith("\n"):
        text = text + "\n"
    # Split on diff --git boundaries, keep separators.
    parts: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            parts.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("".join(current))

    repaired: list[str] = []
    for chunk in parts:
        if not chunk.strip():
            continue
        if "--- /dev/null" in chunk or "new file mode" in chunk:
            repaired.append(chunk if chunk.endswith("\n") else chunk + "\n")
            continue
        if not _looks_like_create_hunks(chunk):
            repaired.append(chunk if chunk.endswith("\n") else chunk + "\n")
            continue
        # Extract path from diff --git or +++ b/
        path: str | None = None
        m = re.match(r"diff --git a/(\S+) b/(\S+)", chunk)
        if m:
            path = m.group(2)
        if path is None:
            for line in chunk.splitlines():
                if line.startswith("+++ b/"):
                    path = line[6:].strip()
                    break
        if not path or path == "/dev/null":
            repaired.append(chunk if chunk.endswith("\n") else chunk + "\n")
            continue
        # Rebuild headers; keep hunk body from first @@ line.
        hunk_start = None
        lines = chunk.splitlines(keepends=True)
        for idx, line in enumerate(lines):
            if line.startswith("@@ "):
                hunk_start = idx
                break
        if hunk_start is None:
            repaired.append(chunk if chunk.endswith("\n") else chunk + "\n")
            continue
        hunk_body = "".join(lines[hunk_start:])
        if not hunk_body.endswith("\n"):
            hunk_body += "\n"
        repaired.append(
            f"diff --git a/{path} b/{path}\n"
            f"new file mode 100644\n"
            f"--- /dev/null\n"
            f"+++ b/{path}\n"
            f"{hunk_body}"
        )
    body = "".join(repaired)
    return body if body.endswith("\n") else body + "\n"


def extract_gold_from_files(files: Sequence[PrFileChange]) -> str:
    """Build multi-file unified gold patch from PR source file changes only.

    Tests that the PR introduced stay in the base tree (pin base_commit to PR base
    so those tests exist after we rehabilitate sources via gold). Including large
    test-only hunks in gold risks agent-side leak / G4 confusion; gold is source fix.
    """
    chunks: list[str] = []
    for change in files:
        if not change.is_source:
            continue
        if change.status in {"removed", "deleted"}:
            continue
        if not change.patch or not change.patch.strip():
            continue
        chunks.append(wrap_file_diff(change.path, change.patch, status=change.status).rstrip("\n"))
    if not chunks:
        # Fallback: include any modified text with patch (still require multi-file later)
        for change in files:
            if change.status in {"removed", "deleted"}:
                continue
            if not change.patch or not change.patch.strip():
                continue
            if is_test_path(change.path):
                continue
            chunks.append(
                wrap_file_diff(change.path, change.patch, status=change.status).rstrip("\n")
            )
    if not chunks:
        raise PrMineError("PR has no usable source patches for gold extraction")
    gold = "\n".join(chunks)
    return gold if gold.endswith("\n") else gold + "\n"


def derive_test_commands(
    *,
    language: str,
    test_files: Sequence[str],
    f2p_override: Sequence[str] | None = None,
    p2p_override: Sequence[str] | None = None,
    baseline_command: str | None = None,
) -> tuple[list[str], list[str]]:
    """Derive fail_to_pass / pass_to_pass commands from language + PR test paths.

    Prefer running the PR's test files as F2P; P2P is a broader baseline that
    excludes F2P paths when a runner supports path selection.
    """
    if f2p_override is not None:
        f2p = [c for c in f2p_override if str(c).strip()]
        p2p = list(p2p_override) if p2p_override is not None else []
        if not f2p:
            raise PrMineError("fail_to_pass override is empty")
        return f2p, [c for c in p2p if str(c).strip()]

    lang = language.strip().lower()
    tests = [t for t in test_files if t.strip()]
    if lang in {"python", "py"}:
        if tests:
            quoted = " ".join(tests)
            f2p = [f"python -m pytest {quoted} -q"]
            # P2P: broader baseline command (live env may narrow further)
            p2p = [baseline_command or "python -m pytest -q"]
        else:
            f2p = [baseline_command or "python -m pytest -q"]
            p2p = []
    elif lang in {"javascript", "js", "typescript", "ts"}:
        if tests:
            f2p = [f"npx jest {' '.join(tests)}"]
            p2p = [baseline_command or "npm test"]
        else:
            f2p = [baseline_command or "npm test"]
            p2p = []
    elif lang == "go":
        packages = sorted({str(Path(t).parent).replace("\\", "/") or "." for t in tests}) or ["."]
        pkgs = " ".join(
            "./..." if p in {".", "./"} else f"./{p.lstrip('./')}/..." for p in packages
        )
        f2p = [f"go test {pkgs}"]
        p2p = [baseline_command or "go test ./..."]
    else:
        f2p = [baseline_command or "python -m pytest -q"]
        p2p = []
    return f2p, p2p


def build_problem_statement(
    *,
    pr: MergedPR,
    source_files: Sequence[str] | None = None,
) -> str:
    """Agent-facing prompt from PR title/body (no gold leakage)."""
    sources = list(source_files) if source_files is not None else list(pr.source_files)
    files = ", ".join(sources[:12]) if sources else "(multiple modules)"
    title = (pr.title or "").strip() or f"PR #{pr.number}"
    body = (pr.body or "").strip()
    # Strip common secrecy markers / dump lengths
    body_snip = re.sub(r"\n{3,}", "\n\n", body)
    if len(body_snip) > 1200:
        body_snip = body_snip[:1200].rstrip() + "…"
    parts = [
        f"Repository `{pr.repo}` has a regression fixed by PR #{pr.number}: {title}.",
        f"Affected source modules include: {files}.",
        "Restore the intended multi-file behaviour so the fail_to_pass tests pass "
        "without weakening pass_to_pass regressions.",
    ]
    if body_snip:
        parts.append("PR description:\n" + body_snip)
    if pr.html_url:
        parts.append(f"Context: {pr.html_url}")
    prompt = "\n\n".join(parts)
    if not prompt.strip():
        raise PrMineError("problem_statement builder produced empty string")
    return prompt


def _instance_id(repo: str, pr_number: int, suffix: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", repo).strip("_").lower()
    return f"real_pr__{slug}__{pr_number}__{suffix}"


def _language_from_paths(files: Sequence[PrFileChange], default: str = "python") -> str:
    """Infer dominant source language from PR paths (VAL-MLANG-001 ``.rs`` bias)."""
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


def ensure_source_track(value: object) -> SourceTrack:
    """Reject invalid/missing source_track values (VAL-PROD-004)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PrMineError(
            f"source_track missing or empty; must be one of {sorted(VALID_SOURCE_TRACKS)}"
        )
    if isinstance(value, SourceTrack):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        # Hybrid/motor labels are never valid product tracks for pr_miner
        if cleaned in {
            "hybrid_curated",
            "hybrid",
            "motor",
            "harbor_motor",
            "synthetic_motor",
        }:
            raise PrMineError(
                f"invalid source_track {value!r}; hybrid/motor not allowed "
                f"(product track is real_pr only; VAL-RPR-005)"
            )
        if cleaned not in VALID_SOURCE_TRACKS:
            raise PrMineError(
                f"invalid source_track {value!r}; must be one of {sorted(VALID_SOURCE_TRACKS)}"
            )
        return SourceTrack(cleaned)
    raise PrMineError(f"invalid source_track type {type(value)!r}")


_MOTOR_REPO_MARKERS = (
    "fixtures/harbor_motors",
    "harbor_motors/",
    "fixtures/tiny_green",
    "orderlib/",
    "python_orders",
    "go_kvstore",
    "ts_registry",
)


def _reject_motor_repo(repo: str) -> None:
    """Refuse harbor motors / synthetic fixture trees as real_pr product inputs."""
    raw = (repo or "").strip().lower()
    if not raw:
        return
    if any(marker in raw for marker in _MOTOR_REPO_MARKERS):
        raise PrMineError(
            f"motor_or_hybrid_rejected: {repo!r} looks like a hybrid motor/fixture; "
            "real_pr mine only accepts merged public GitHub PRs (VAL-RPR-002/005)"
        )
    if raw.startswith("file://"):
        raise PrMineError(
            f"motor_or_hybrid_rejected: file:// repo {repo!r} is not a public PR source"
        )


@dataclass
class PrMiner:
    """Mine multi-file merged PRs and emit labeled real_pr TaskRecords."""

    client: GitHubClient
    work_root: Path | None = None
    image_digest: str = _DEFAULT_IMAGE_DIGEST
    min_source_files: int = MULTI_FILE_FLOOR
    require_tests: bool = True
    license: str = "MIT"
    keep_workspaces: bool = True
    # Product keep uses LHARD floors; soft MULTI_FILE_FLOOR stays default for offline.
    product_mode: bool = False
    min_source_hunks: int = PRODUCT_SOURCE_HUNK_FLOOR

    def select_from_files(
        self,
        *,
        repo: str,
        number: int,
        title: str,
        body: str,
        base_commit: str,
        merge_commit_sha: str | None,
        html_url: str,
        files_payload: Sequence[dict[str, Any]],
        language: str | None = None,
        license: str | None = None,
        require_full_base_sha: bool = False,
        enforce_license: bool = False,
        merged_at: str | None = None,
        product_hard_filter: bool | None = None,
    ) -> MergedPR:
        """Build MergedPR after multi-file + tests filters (no network).

        Refuses hybrid motor fixtures as product candidates (VAL-RPR-002/005).
        When ``product_hard_filter`` (or ``self.product_mode``) is true, apply
        M14 product hard floors (≥10 source hunks, docs/chore, license, suite,
        and always-required real ``merged_at``; no invented merge markers).
        """
        from swe_factory.sources.discover import is_full_sha

        _reject_motor_repo(repo)
        if not base_commit or not str(base_commit).strip():
            raise PrMineError(f"PR {repo}#{number} missing base_commit pin")
        pin = str(base_commit).strip()
        if require_full_base_sha and not is_full_sha(pin):
            raise PrMineError(
                f"PR {repo}#{number} base_commit must be full 40-char hex SHA; got {pin!r}"
            )
        lic = license or self.license
        if enforce_license:
            try:
                assert_permissive_license(lic, repo=repo)
            except LicenseGateError as exc:
                raise PrMineError(str(exc)) from exc
        changes = tuple(_as_file_change(item) for item in files_payload)
        use_product = self.product_mode if product_hard_filter is None else product_hard_filter
        if use_product:
            # Product path always requires a real merge signal: never invent
            # merged_at="product" or force merged=True when callers omit merged_at.
            hard = evaluate_product_hard_filter(
                files=changes,
                base_commit=pin,
                merged_at=merged_at,
                language=language,
                license=lic,
                repo=repo,
                title=title,
                require_merged=True,
                require_full_base_sha=require_full_base_sha or self.product_mode,
                require_license=enforce_license or self.product_mode,
                min_source_hunks=self.min_source_hunks,
                min_source_files=max(self.min_source_files, PRODUCT_MULTI_FILE_FLOOR),
                require_tests=self.require_tests,
            )
            if not hard.accepted:
                raise PrMineError(
                    f"PR {repo}#{number} rejected by product hard filter "
                    f"({hard.reason_code}): {hard.detail} "
                    f"[source_hunk_count={hard.stats.source_hunk_count}, "
                    f"license={hard.stats.license!r}]"
                )
            lang = language or hard.stats.language or _language_from_paths(changes)
            hunk_count = hard.stats.source_hunk_count
        else:
            if not multi_file_source_filter(
                changes,
                min_source_files=self.min_source_files,
                require_tests=self.require_tests,
            ):
                sources = sorted({c.path for c in changes if c.is_source})
                tests = sorted({c.path for c in changes if c.is_test})
                raise PrMineError(
                    f"PR {repo}#{number} rejected by multi-file/tests filter "
                    f"(source_files={sources}, test_files={tests}, "
                    f"min_source_files={self.min_source_files}, require_tests={self.require_tests})"
                )
            lang = language or _language_from_paths(changes)
            hunk_count = measure_source_hunk_count(changes)
        return MergedPR(
            repo=repo,
            number=int(number),
            title=title or f"PR #{number}",
            body=body or "",
            base_commit=pin.lower() if is_full_sha(pin) else pin,
            merge_commit_sha=merge_commit_sha,
            language=lang,
            html_url=html_url or "",
            files=changes,
            license=lic,
            merged_at=merged_at,
            source_hunk_count=hunk_count,
        )

    def fetch_merged_pr(
        self,
        repo: str,
        number: int,
        *,
        language: str | None = None,
        license: str | None = None,
        require_full_base_sha: bool = True,
        enforce_license: bool = True,
        product_hard_filter: bool | None = None,
    ) -> MergedPR:
        """Load one PR via GitHub API and apply selection filters.

        Live/API path requires full 40-char base SHA (VAL-MINE-002) and
        permissive license (VAL-MINE-003) by default.
        Rejects harbor motors / hybrid fixtures as product candidates
        (VAL-RPR-002, VAL-RPR-005).
        Product live-mine enables M14 hard floors (VAL-LHARD-*) when
        ``product_hard_filter`` / ``product_mode`` is set.
        """
        _reject_motor_repo(repo)
        try:
            meta = self.client.get_pull(repo, number)
            files_payload = self.client.list_all_pull_files(repo, number)
        except GitHubError as exc:
            raise PrMineError(str(exc)) from exc
        if not meta.get("merged_at"):
            raise PrMineError(f"PR {repo}#{number} is not merged")
        base_obj = meta.get("base")
        base: dict[str, Any] = base_obj if isinstance(base_obj, dict) else {}
        base_sha = str(base.get("sha") or "").strip()
        if not base_sha:
            raise PrMineError(f"PR {repo}#{number} has no base.sha")
        merge_sha = str(meta.get("merge_commit_sha") or "").strip() or None
        use_product = self.product_mode if product_hard_filter is None else product_hard_filter
        return self.select_from_files(
            repo=repo,
            number=number,
            title=str(meta.get("title") or ""),
            body=str(meta.get("body") or ""),
            base_commit=base_sha,
            merge_commit_sha=merge_sha,
            html_url=str(meta.get("html_url") or ""),
            files_payload=files_payload,
            language=language,
            license=license,
            require_full_base_sha=require_full_base_sha,
            enforce_license=enforce_license,
            merged_at=str(meta.get("merged_at") or "") or None,
            product_hard_filter=use_product,
        )

    def list_candidate_prs(
        self,
        repo: str,
        *,
        max_scan: int = 30,
        max_keep: int = 5,
        language: str | None = None,
    ) -> list[MergedPR]:
        """Scan recent closed PRs and keep those that pass multi-file/tests filter."""
        kept: list[MergedPR] = []
        page = 1
        scanned = 0
        while scanned < max_scan and len(kept) < max_keep:
            try:
                batch = self.client.list_pulls(repo, state="closed", page=page, per_page=30)
            except GitHubError as exc:
                raise PrMineError(str(exc)) from exc
            if not batch:
                break
            for meta in batch:
                if scanned >= max_scan or len(kept) >= max_keep:
                    break
                scanned += 1
                if not meta.get("merged_at"):
                    continue
                number = int(meta.get("number") or 0)
                if number <= 0:
                    continue
                try:
                    kept.append(self.fetch_merged_pr(repo, number, language=language))
                except PrMineError:
                    continue
            page += 1
            if len(batch) < 30:
                break
        return kept

    def produce(
        self,
        pr: MergedPR,
        *,
        instance_suffix: str | None = None,
        problem_statement: str | None = None,
        fail_to_pass: Sequence[str] | None = None,
        pass_to_pass: Sequence[str] | None = None,
        baseline_command: str | None = None,
        base_workspace: Path | None = None,
        run_stub_oracle: bool = True,
        source_track: object = SourceTrack.REAL_PR,
    ) -> RealPrCandidate:
        """Materialize labeled real_pr TaskRecord + gold from a selected MergedPR."""
        track = ensure_source_track(source_track)
        if track != SourceTrack.REAL_PR:
            # Allow only real_pr label from this miner (never silent mix)
            raise PrMineError(f"pr_miner only emits source_track=real_pr; got {track.value!r}")

        from swe_factory.sources.discover import (
            extract_test_patch_from_files,
            repository_url_for,
        )

        gold_patch = extract_gold_from_files(pr.files)
        gold_files = tuple(count_files_in_patch(gold_patch))
        if len(gold_files) < self.min_source_files:
            raise PrMineError(
                f"gold multi-file floor failed for {pr.repo}#{pr.number}: files={list(gold_files)}"
            )
        # VAL-MINE-006: gold is source-only; tests held out as test.patch
        for gpath in gold_files:
            if is_test_path(gpath):
                raise PrMineError(
                    f"gold_patch must be source-only; found test path {gpath!r} "
                    f"in {pr.repo}#{pr.number}"
                )
        test_patch = extract_test_patch_from_files(pr.files)
        repo_url = repository_url_for(pr.repo)

        f2p, p2p = derive_test_commands(
            language=pr.language,
            test_files=pr.test_files,
            f2p_override=fail_to_pass,
            p2p_override=pass_to_pass,
            baseline_command=baseline_command,
        )
        prompt = problem_statement or build_problem_statement(pr=pr)
        if not prompt.strip():
            raise PrMineError("problem_statement must be non-empty")

        suffix = instance_suffix or uuid.uuid4().hex[:10]
        instance_id = _instance_id(pr.repo, pr.number, suffix)

        hunk_count = int(pr.source_hunk_count or 0) or measure_source_hunk_count(pr.files)
        provenance: dict[str, Any] = {
            "producer": "real_pr",
            "source_track": SourceTrack.REAL_PR.value,
            "repo": pr.repo,
            "repository_url": repo_url,
            "pr_number": pr.number,
            "pr_url": pr.html_url,
            "pr_title": pr.title,
            "base_commit": pr.base_commit,
            "merge_commit_sha": pr.merge_commit_sha,
            "merged_at": pr.merged_at,
            "source_files": list(pr.source_files),
            "test_files": list(pr.test_files),
            "gold_files": list(gold_files),
            "gold_provenance": "pr_unified_diff_source_files",
            "test_patch_nonempty": bool(test_patch.strip()),
            "language": pr.language,
            "license": pr.license or self.license,
            "source_hunk_count": hunk_count,
            "product_source_hunk_floor": PRODUCT_SOURCE_HUNK_FLOOR,
            "product_multi_file_floor": PRODUCT_MULTI_FILE_FLOOR,
            "soft_multi_file_floor": SOFT_MULTI_FILE_FLOOR,
            "product_mode": bool(self.product_mode),
        }

        task = TaskRecord.model_validate(
            {
                "instance_id": instance_id,
                "source_track": SourceTrack.REAL_PR,
                "repo": pr.repo,
                "base_commit": pr.base_commit,
                "language": pr.language,
                "problem_statement": prompt,
                "fail_to_pass": f2p,
                "pass_to_pass": p2p,
                "gold_patch": gold_patch,
                "environment": EnvironmentMeta(image_digest=self.image_digest),
                "license": pr.license or self.license,
                "gate_proof": {
                    "real_pr_meta": provenance,
                    "producer": "real_pr",
                },
                "created_at": datetime.now(UTC),
            }
        )

        gates: GateResult | None = None
        if run_stub_oracle:
            gates = run_stub_gates(task, require_multi_file=True)
            if not gates.passed:
                raise PrMineError(f"stub oracle rejected real_pr candidate: {gates.reason_codes}")
            task = task.model_copy(
                update={
                    "gate_proof": {
                        **(task.gate_proof or {}),
                        **gates.to_gate_proof(),
                        "real_pr_meta": provenance,
                    }
                }
            )

        workspace: Path | None = None
        if base_workspace is not None or self.work_root is not None:
            work_root = (
                Path(self.work_root)
                if self.work_root
                else Path(tempfile.mkdtemp(prefix="sdf-realpr-"))
            )
            work_root.mkdir(parents=True, exist_ok=True)
            case_dir = work_root / f"pr_{pr.number}_{suffix}"
            if case_dir.exists():
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "gold.patch").write_text(gold_patch, encoding="utf-8")
            if base_workspace is not None and Path(base_workspace).is_dir():
                dest = case_dir / "workspace"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(
                    base_workspace,
                    dest,
                    ignore=shutil.ignore_patterns(
                        ".git",
                        "__pycache__",
                        "*.pyc",
                        ".venv",
                        "node_modules",
                        ".pytest_cache",
                    ),
                )
                workspace = dest
            # Held-out test.patch for verifier (VAL-MINE-006)
            (case_dir / "test.patch").write_text(test_patch or "", encoding="utf-8")
            (case_dir / "solution.patch").write_text(gold_patch, encoding="utf-8")
            if self.keep_workspaces:
                import json as _json

                (case_dir / "real_pr_meta.json").write_text(
                    _json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

        return RealPrCandidate(
            task=task,
            pr=pr,
            gold_files=gold_files,
            workspace=workspace,
            provenance=provenance,
            gates=gates,
            provider_calls=0,
            test_patch=test_patch,
            repository_url=repo_url,
        )

    def produce_from_pr_number(
        self,
        repo: str,
        number: int,
        **produce_kwargs: Any,
    ) -> RealPrCandidate:
        """Fetch + select + produce for one PR number."""
        language = produce_kwargs.pop("language", None)
        pr = self.fetch_merged_pr(repo, number, language=language)
        return self.produce(pr, **produce_kwargs)

    def produce_and_certify(
        self,
        pr: MergedPR,
        *,
        runner: OracleRunnerBackend,
        workspace: Path,
        dual_runs: int = 2,
        audit_out: Path | None = None,
        **produce_kwargs: Any,
    ) -> RealPrCandidate:
        """Produce labeled candidate then run certified oracle (Fake or Docker)."""
        produce_kwargs.pop("run_stub_oracle", None)
        produce_kwargs["base_workspace"] = workspace
        candidate = self.produce(pr, run_stub_oracle=True, **produce_kwargs)
        ws = candidate.workspace or Path(workspace)
        result = run_certified_gates_for_task(
            candidate.task,
            workspace=ws,
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
                extra={"source_track": SourceTrack.REAL_PR.value},
            )
        task = candidate.task.model_copy(
            update={
                "gate_proof": {
                    **(candidate.task.gate_proof or {}),
                    **result.to_gate_proof(),
                    "real_pr_meta": candidate.provenance,
                }
            }
        )
        if not result.passed:
            raise PrMineError(
                f"certified oracle rejected real_pr candidate "
                f"{task.instance_id}: {result.reason_codes}"
            )
        return RealPrCandidate(
            task=task,
            pr=candidate.pr,
            gold_files=candidate.gold_files,
            workspace=candidate.workspace,
            provenance=candidate.provenance,
            gates=result,
            provider_calls=0,
            test_patch=candidate.test_patch,
            repository_url=candidate.repository_url,
        )


def offline_fixture_pr(
    *,
    repo: str = "fixtures/tiny_real_pr",
    number: int = 1,
    language: str = "python",
) -> MergedPR:
    """Build a multi-file PR fixture that mirrors tiny_offline gold (tests offline)."""
    files = (
        PrFileChange(
            path="demo_pkg/math_ops.py",
            status="modified",
            patch=(
                "@@ -1,3 +1,3 @@\n"
                " def add(a: int, b: int) -> int:\n"
                "-    return a - b\n"
                "+    return a + b\n"
            ),
        ),
        PrFileChange(
            path="demo_pkg/text_ops.py",
            status="modified",
            patch=(
                "@@ -1,3 +1,3 @@\n"
                " def reverse_words(text: str) -> str:\n"
                "-    return text\n"
                '+    return " ".join(reversed(text.split()))\n'
            ),
        ),
        PrFileChange(
            path="tests/test_math.py",
            status="modified",
            patch=(
                "@@ -0,0 +1,5 @@\n+def test_add():\n+    from demo_pkg.math_ops import add\n"
                "+    assert add(1, 2) == 3\n"
            ),
        ),
        PrFileChange(
            path="tests/test_text.py",
            status="modified",
            patch=(
                "@@ -0,0 +1,5 @@\n"
                "+def test_reverse_words():\n"
                "+    from demo_pkg.text_ops import reverse_words\n"
                '+    assert reverse_words("a b") == "b a"\n'
            ),
        ),
    )
    return MergedPR(
        repo=repo,
        number=number,
        title="Fix multi-module add + reverse_words regressions",
        body=(
            "Restore demo_pkg math_ops.add and text_ops.reverse_words so the "
            "multi-file suite passes."
        ),
        # Offline pins are 40-char labels (hex-compatible length for schema);
        # not real git objects and not DeepSWE certified keeps.
        base_commit="fixture000000000000000000000000000000001",
        merge_commit_sha="fixturefffffffffffffffffffffffffffffffff",
        language=language,
        html_url=f"https://github.com/{repo}/pull/{number}",
        files=files,
        license="MIT",
    )


def produce_offline_fixture(
    *,
    work_root: Path | None = None,
    run_stub_oracle: bool = True,
    base_workspace: Path | None = None,
) -> RealPrCandidate:
    """Convenience offline entry: offline fixture PR without GitHub network."""
    from swe_factory.sources.github import DictGitHubTransport

    # Client never used for offline_fixture path, but constructor needs one
    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client, work_root=work_root)
    pr = offline_fixture_pr()
    return miner.produce(
        pr,
        instance_suffix="offline",
        fail_to_pass=["python -m pytest tests/test_math.py tests/test_text.py -q"],
        pass_to_pass=["python -m pytest tests/test_ok.py -q"],
        base_workspace=base_workspace,
        run_stub_oracle=run_stub_oracle,
    )


__all__ = [
    "MULTI_FILE_FLOOR",
    "PRODUCT_MULTI_FILE_FLOOR",
    "PRODUCT_SOURCE_HUNK_FLOOR",
    "SOFT_MULTI_FILE_FLOOR",
    "MergedPR",
    "PrFileChange",
    "PrMineError",
    "PrMiner",
    "RealPrCandidate",
    "build_problem_statement",
    "derive_test_commands",
    "ensure_source_track",
    "evaluate_product_hard_filter",
    "extract_gold_from_files",
    "is_source_path",
    "is_test_path",
    "measure_source_hunk_count",
    "multi_file_source_filter",
    "offline_fixture_pr",
    "produce_offline_fixture",
    "repair_pseudo_create_file_headers",
    "wrap_file_diff",
]
