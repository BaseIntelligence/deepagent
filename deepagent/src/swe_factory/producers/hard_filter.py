"""Product hard filter for live-mined real_pr keeps (M14 + M27 DeepSWE-median).

Product keep requires **all** of:

- ``merged_at`` present (merged PR only)
- Full 40-char hex base commit SHA
- ``source_hunk_count >= PRODUCT_SOURCE_HUNK_FLOOR`` (default **14**, DeepSWE p50)
- Multi-file product source ≥ ``PRODUCT_MULTI_FILE_FLOOR`` (default **4**, DeepSWE p25)
- ≥1 real test path change
- Suite reporter detectability for the PR language
- Permissive license (fail-closed on missing / copyleft / unknown)
- Refuse pure docs / chore, motor / hybrid markers

The soft offline floor (``MULTI_FILE_FLOOR=2``) remains for engineering fixtures
via :func:`multi_file_source_filter` and offline pool paths. Product live keep
uses :func:`apply_product_hard_filter` / :func:`evaluate_product_hard_filter`.

Source-hunk definition (fixed + unit-tested):
count of unified-diff ``@@`` hunk headers across product source file patches
only (non-test, non-docs/chore paths with a code extension). Path renames or
patch-less metadata rows do not contribute hunks.

M27 DeepSWE-median band (public sample N≈48): files p50≈6 / p25≈4, hunks
p50≈14. Gold added-lines floor (≥400) is enforced post-materialize by
:mod:`swe_factory.pipeline.hardness_floors` (VAL-DMED-001).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.oracle.gates import MULTI_FILE_FLOOR
from swe_factory.producers.suite_reporters import (
    list_reporter_languages,
    normalize_reporter_lang,
    reporter_info,
)
from swe_factory.sources.license_gate import classify_license

# Path / SHA helpers are defined locally so this module stays import-cycle free
# vs sources.discover (which imports pr_miner) and producers.pr_miner.

_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

_MOTOR_HYBRID_MARKERS: tuple[str, ...] = (
    "fixtures/harbor_motors",
    "harbor_motors/",
    "fixtures/tiny_green",
    "fixtures/tiny_offline",
    "orderlib",
    "kvstore",
    "ts_registry",
    "python_orders",
    "go_kvstore",
)
_HYBRID_TRACKS = frozenset(
    {
        "hybrid_curated",
        "hybrid",
        "motor",
        "harbor_motor",
        "synthetic_motor",
    }
)


def is_full_sha(value: str | None) -> bool:
    """True when value is an immutable full 40-char hex commit SHA."""
    if not value:
        return False
    return bool(_FULL_SHA_RE.fullmatch(value.strip()))


def is_motor_or_hybrid_identity(
    identity: str | None,
    *,
    source_track: str | None = None,
    seed_id: str | None = None,
    kind: str | None = None,
) -> tuple[bool, str]:
    """Return (is_banned, reason) for product real_pr mine paths (local copy)."""
    track = (source_track or "").strip().lower()
    if track in _HYBRID_TRACKS:
        return True, f"source_track={track!r} is hybrid/motor; product path requires real_pr"
    kind_n = (kind or "").strip().lower()
    if kind_n in {"curated", "hybrid", "motor", "hybrid_curated"}:
        return True, f"kind={kind_n!r} is not allowed on real_pr-only product path"
    if seed_id and (seed_id.startswith("harbor_") or "motor" in seed_id.lower()):
        return True, f"seed_id={seed_id!r} looks like a harbor motor"
    raw = (identity or "").strip().lower()
    if not raw:
        return False, ""
    if any(marker in raw for marker in _MOTOR_HYBRID_MARKERS):
        return True, f"identity {identity!r} matches motor/fixture marker"
    if raw.startswith("file://"):
        return True, f"file:// repo {identity!r} is not a public PR source"
    return False, ""


# ---------------------------------------------------------------------------
# Product floors (hard keep). Soft MULTI_FILE_FLOOR stays for offline only.
# ---------------------------------------------------------------------------

#: Product keep minimum unified-diff source hunks (M27 DeepSWE p50; VAL-DMED-001).
PRODUCT_SOURCE_HUNK_FLOOR: int = 14

#: Product keep minimum distinct product-source files (M27 DeepSWE p25; VAL-DMED-001).
PRODUCT_MULTI_FILE_FLOOR: int = 4

#: Hybrid multi-file DeepSWE-min branch (VAL-DMED-012): files≥3 admits when
#: source additions ≥ 500 and hunks ≥ PRODUCT_SOURCE_HUNK_FLOOR.
PRODUCT_HYBRID_MIN_SOURCE_FILES: int = 3
PRODUCT_HYBRID_MIN_ADDED_LINES: int = 500

#: Soft offline engineering floor re-export (never promote product below hard).
SOFT_MULTI_FILE_FLOOR: int = MULTI_FILE_FLOOR

# Stable reject reason codes for candidates ledgers / gate_audit.
REASON_NOT_MERGED: str = "not_merged"
REASON_BASE_SHA_INVALID: str = "base_commit_not_full_sha"
REASON_SOURCE_HUNKS_BELOW_FLOOR: str = "source_hunks_below_floor"
REASON_MULTI_FILE_FLOOR: str = "multi_file_floor_rejected"
REASON_TESTS_MISSING: str = "tests_missing"
REASON_SUITE_REPORTER_UNAVAILABLE: str = "suite_reporter_unavailable"
REASON_DOCS_CHORE_ONLY: str = "docs_chore_only"
REASON_MOTOR_OR_HYBRID: str = "motor_or_hybrid_rejected"
REASON_LICENSE_REJECTED: str = "license_rejected"
REASON_LICENSE_COPYLEFT: str = "license_copyleft_rejected"
REASON_LICENSE_MISSING: str = "license_missing_rejected"
REASON_LICENSE_UNKNOWN: str = "license_unknown_rejected"
REASON_PURE_CHORE_MARKERS: str = "chore_only_rejected"
REASON_OK: str = "product_hard_filter_ok"

_HUNK_HEADER_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@", re.MULTILINE)

_DOCS_EXTS = frozenset(
    {
        ".md",
        ".mdx",
        ".rst",
        ".txt",
        ".adoc",
        ".org",
        ".html",
        ".htm",
    }
)
_DOCS_NAMES = frozenset(
    {
        "readme",
        "changelog",
        "changes",
        "history",
        "license",
        "copying",
        "authors",
        "contributors",
        "code_of_conduct",
        "security",
        "notice",
        "news",
        "todo",
    }
)
_DOCS_DIR_PARTS = frozenset(
    {
        "docs",
        "doc",
        "documentation",
        "examples",
        "example",
        "tutorial",
        "tutorials",
        "guides",
        "guide",
        "website",
        "site",
        "pages",
    }
)
_CHORE_DIR_PARTS = frozenset(
    {
        ".github",
        ".circleci",
        ".gitlab",
        ".azure",
        ".jenkins",
        "ci",
        "ops",
        "deploy",
        "deployment",
        "scripts",
        "script",
        "tooling",
        "infra",
        "infrastructure",
        "terraform",
        "k8s",
        "kubernetes",
        "helm",
        "charts",
    }
)
_CHORE_NAMES = frozenset(
    {
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
        ".pre-commit-config.yaml",
        ".prettierrc",
        ".eslintrc",
        ".eslintrc.js",
        ".eslintrc.cjs",
        ".eslintrc.json",
        ".nvmrc",
        ".node-version",
        ".python-version",
        ".tool-versions",
        "makefile",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "procfile",
        "renovate.json",
        "dependabot.yml",
        "codecov.yml",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "go.mod",
        "go.sum",
        "cargo.toml",
        "cargo.lock",
        "tsconfig.json",
        "jsconfig.json",
        "webpack.config.js",
        "vite.config.js",
        "vite.config.ts",
        "rollup.config.js",
        "babel.config.js",
        "tox.ini",
        "noxfile.py",
        "manifest.in",
        "cmakelists.txt",
    }
)
_CHORE_EXTS = frozenset(
    {
        ".yml",
        ".yaml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".lock",
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".webp",
        ".map",
    }
)

# Title / path markers that imply pure maintenance PRs even when mixed config.
_CHORE_TITLE_MARKERS = (
    "chore:",
    "chore(",
    "docs:",
    "docs(",
    "ci:",
    "ci(",
    "build:",
    "build(",
    "style:",
    "style(",
    "deps:",
    "dependency",
    "bump version",
    "release-only",
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


@dataclass(frozen=True, slots=True)
class HardFilterStats:
    """Measurable stats for candidate ledgers (source hunks, files, license)."""

    source_hunk_count: int = 0
    source_file_count: int = 0
    test_file_count: int = 0
    docs_chore_file_count: int = 0
    other_file_count: int = 0
    source_files: tuple[str, ...] = ()
    test_files: tuple[str, ...] = ()
    docs_chore_files: tuple[str, ...] = ()
    source_added_lines: int = 0
    license: str = ""
    language: str = ""
    base_commit: str = ""
    merged: bool = False
    suite_reporter: str = ""
    suite_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HardFilterResult:
    """Product hard-filter decision with stable reason codes."""

    accepted: bool
    reason_codes: tuple[str, ...]
    detail: str
    stats: HardFilterStats
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def reason_code(self) -> str:
        """Primary reason code (ok when accepted; first reject otherwise)."""
        if self.accepted:
            return REASON_OK
        return self.reason_codes[0] if self.reason_codes else "hard_filter_rejected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "stats": self.stats.to_dict(),
            "meta": dict(self.meta),
            "source_hunk_count": self.stats.source_hunk_count,
            "license": self.stats.license,
        }

    def to_ledger_fields(self) -> dict[str, Any]:
        """Minimal ledger fields required for product candidates.jsonl rows."""
        return {
            "source_hunk_count": self.stats.source_hunk_count,
            "source_file_count": self.stats.source_file_count,
            "test_file_count": self.stats.test_file_count,
            "license": self.stats.license,
            "language": self.stats.language,
            "base_commit": self.stats.base_commit,
            "merged": self.stats.merged,
            "suite_reporter": self.stats.suite_reporter,
            "hard_filter_accepted": self.accepted,
            "hard_filter_reason": self.reason_code,
            "hard_filter_reasons": list(self.reason_codes),
            "product_source_hunk_floor": PRODUCT_SOURCE_HUNK_FLOOR,
            "product_multi_file_floor": PRODUCT_MULTI_FILE_FLOOR,
            "soft_multi_file_floor": SOFT_MULTI_FILE_FLOOR,
        }


def is_docs_path(path: str) -> bool:
    """True when path is documentation / narrative content only."""
    if not path or path.endswith("/"):
        return False
    norm = path.replace("\\", "/").strip("/")
    lower = norm.lower()
    parts = [p for p in lower.split("/") if p]
    name = Path(lower).name
    stem = Path(lower).stem
    suffix = Path(lower).suffix
    if suffix in _DOCS_EXTS:
        return True
    if stem in _DOCS_NAMES or name in _DOCS_NAMES:
        return True
    if any(p in _DOCS_DIR_PARTS for p in parts[:-1]):
        # File under docs/ tree (even code samples treated as docs product-refuse for pure docs)
        if suffix in _DOCS_EXTS or stem in _DOCS_NAMES:
            return True
        if suffix not in {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".rb",
            ".php",
        }:
            return True
    return False


def is_chore_path(path: str) -> bool:
    """True when path is CI/config/chore tooling, not product source."""
    if not path or path.endswith("/"):
        return False
    norm = path.replace("\\", "/").strip("/")
    lower = norm.lower()
    parts = [p for p in lower.split("/") if p]
    name = Path(lower).name
    suffix = Path(lower).suffix
    if name in _CHORE_NAMES:
        return True
    if any(p in _CHORE_DIR_PARTS for p in parts):
        return True
    if suffix in _CHORE_EXTS and not is_source_path(path) and not is_test_path(path):
        return True
    # Root shell/batch maintenance scripts
    return len(parts) == 1 and suffix in {".sh", ".bash", ".ps1", ".bat", ".cmd"}


def is_docs_or_chore_path(path: str) -> bool:
    return is_docs_path(path) or is_chore_path(path)


def is_product_source_path(path: str) -> bool:
    """Product source = code source extension, not test, not docs/chore, not skip vendored."""
    if not path:
        return False
    if is_test_path(path):
        return False
    if is_docs_or_chore_path(path):
        return False
    return is_source_path(path)


def count_unified_diff_hunks(patch: str | None) -> int:
    """Count ``@@`` hunk headers in a unified-diff body (GitHub per-file form OK)."""
    if not patch or not str(patch).strip():
        return 0
    return len(_HUNK_HEADER_RE.findall(str(patch)))


@dataclass(frozen=True, slots=True)
class _FileView:
    path: str
    status: str
    patch: str | None
    additions: int = 0
    deletions: int = 0


def _as_file_change(item: Mapping[str, Any] | Any) -> _FileView:
    if isinstance(item, Mapping):
        path = str(item.get("filename") or item.get("path") or "")
        status = str(item.get("status") or "modified")
        patch = item.get("patch")
        patch_s = patch if isinstance(patch, str) else None
        return _FileView(
            path=path,
            status=status,
            patch=patch_s,
            additions=int(item.get("additions") or 0),
            deletions=int(item.get("deletions") or 0),
        )
    # Duck type (PrFileChange / similar)
    path = str(getattr(item, "path", "") or getattr(item, "filename", "") or "")
    status = str(getattr(item, "status", "modified") or "modified")
    patch = getattr(item, "patch", None)
    patch_s = patch if isinstance(patch, str) else None
    return _FileView(
        path=path,
        status=status,
        patch=patch_s,
        additions=int(getattr(item, "additions", 0) or 0),
        deletions=int(getattr(item, "deletions", 0) or 0),
    )


def measure_source_hunk_count(
    files: Sequence[Mapping[str, Any] | Any],
) -> int:
    """Sum unified-diff hunks across **product source** paths only.

    Definition was fixed for product keep (VAL-LHARD-001 / VAL-DMED-001): each
    ``@@`` header in a product-source file patch counts as one source hunk. Test,
    docs, chore, and vendor patches do not count. Boundary (M27): 13 → reject,
    14 → keep-eligible.
    """
    total = 0
    for raw in files:
        change = _as_file_change(raw)
        if not is_product_source_path(change.path):
            continue
        if change.status in {"removed", "deleted"} and not (change.patch or "").strip():
            continue
        total += count_unified_diff_hunks(change.patch)
    return total


def classify_pr_files(
    files: Sequence[Mapping[str, Any] | Any],
) -> HardFilterStats:
    """Bucket file changes and measure product source hunks."""
    sources: list[str] = []
    tests: list[str] = []
    docs_chore: list[str] = []
    other = 0
    hunks = 0
    added = 0
    for raw in files:
        change = _as_file_change(raw)
        path = change.path
        if not path:
            other += 1
            continue
        if is_test_path(path):
            if path not in tests:
                tests.append(path)
        elif is_docs_or_chore_path(path):
            if path not in docs_chore:
                docs_chore.append(path)
        elif is_product_source_path(path):
            if path not in sources:
                sources.append(path)
            hunks += count_unified_diff_hunks(change.patch)
            # Prefer GitHub additions when present; fall back to plus-line count.
            if change.additions > 0:
                added += int(change.additions)
            elif change.patch:
                added += sum(
                    1
                    for ln in str(change.patch).splitlines()
                    if ln.startswith("+") and not ln.startswith("+++")
                )
        else:
            other += 1
    return HardFilterStats(
        source_hunk_count=hunks,
        source_file_count=len(sources),
        test_file_count=len(tests),
        docs_chore_file_count=len(docs_chore),
        other_file_count=other,
        source_files=tuple(sorted(sources)),
        test_files=tuple(sorted(tests)),
        docs_chore_files=tuple(sorted(docs_chore)),
        source_added_lines=added,
    )


def suite_reporter_detectable(language: str | None) -> tuple[bool, str, str]:
    """Return (ok, reporter_id, suite_command-ish label) for product dual-run."""
    if not language or not str(language).strip():
        return False, "", ""
    try:
        info = reporter_info(str(language))
    except Exception:
        return False, "", ""
    lang = normalize_reporter_lang(str(language))
    if lang not in list_reporter_languages():
        return False, "", ""
    # Command strings mirror real_dual_run defaults without importing dual-run (cycle).
    cmd_by_lang = {
        "python": "python -m pytest -q",
        "go": "go test ./...",
        "typescript": "npx jest",
        "javascript": "npx jest",
        "rust": "cargo test",
    }
    return True, info.reporter_id, cmd_by_lang.get(lang, info.tool_label)


def _license_reason(license_value: str | None) -> tuple[bool, str, str]:
    """Return (ok, reason_code, normalized label)."""
    decision = classify_license(license_value)
    if decision.permitted:
        return True, "license_permissive_ok", decision.normalized or str(license_value or "")
    # Map license_gate reasons onto hard-filter ledger codes.
    code = decision.reason_code
    if code.endswith("copyleft_rejected") or "copyleft" in code:
        return False, REASON_LICENSE_COPYLEFT, decision.normalized
    if code.endswith("missing_rejected") or "missing" in code:
        return False, REASON_LICENSE_MISSING, decision.normalized
    if code.endswith("unknown_rejected") or "unknown" in code:
        return False, REASON_LICENSE_UNKNOWN, decision.normalized
    return False, REASON_LICENSE_REJECTED, decision.normalized


def _looks_like_chore_title(title: str | None) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    return any(m in t for m in _CHORE_TITLE_MARKERS)


def evaluate_product_hard_filter(
    *,
    files: Sequence[Mapping[str, Any] | Any],
    base_commit: str | None,
    merged_at: str | None = None,
    merged: bool | None = None,
    language: str | None = None,
    license: str | None = None,
    repo: str | None = None,
    source_track: str | None = None,
    seed_id: str | None = None,
    title: str | None = None,
    require_merged: bool = True,
    require_full_base_sha: bool = True,
    require_license: bool = True,
    require_suite_reporter: bool = True,
    min_source_hunks: int = PRODUCT_SOURCE_HUNK_FLOOR,
    min_source_files: int = PRODUCT_MULTI_FILE_FLOOR,
    require_tests: bool = True,
    refuse_motor_hybrid: bool = True,
) -> HardFilterResult:
    """Evaluate product keep hard floors; return accepted + reason codes + stats.

    Soft offline floor (``MULTI_FILE_FLOOR=2``) is **not** used here; callers that
    need soft offline should use :func:`multi_file_source_filter` instead.
    """
    stats = classify_pr_files(files)
    lang = (language or "").strip()
    if not lang:
        # Infer coarse language from product source suffixes.
        counts: dict[str, int] = {}
        for path in stats.source_files:
            suffix = Path(path).suffix.lower()
            if suffix == ".py":
                counts["python"] = counts.get("python", 0) + 1
            elif suffix in {".js", ".jsx"}:
                counts["javascript"] = counts.get("javascript", 0) + 1
            elif suffix in {".ts", ".tsx"}:
                counts["typescript"] = counts.get("typescript", 0) + 1
            elif suffix == ".go":
                counts["go"] = counts.get("go", 0) + 1
            elif suffix == ".rs":
                counts["rust"] = counts.get("rust", 0) + 1
        if counts:
            lang = max(counts.items(), key=lambda kv: kv[1])[0]

    is_merged = bool(merged) if merged is not None else bool(merged_at)
    pin = (base_commit or "").strip()
    suite_ok, suite_id, suite_cmd = suite_reporter_detectable(lang)
    lic_raw = (license or "").strip()
    lic_ok, lic_reason, lic_norm = _license_reason(lic_raw if require_license else "MIT")

    stats = HardFilterStats(
        source_hunk_count=stats.source_hunk_count,
        source_file_count=stats.source_file_count,
        test_file_count=stats.test_file_count,
        docs_chore_file_count=stats.docs_chore_file_count,
        other_file_count=stats.other_file_count,
        source_files=stats.source_files,
        test_files=stats.test_files,
        docs_chore_files=stats.docs_chore_files,
        source_added_lines=stats.source_added_lines,
        license=lic_raw or lic_norm,
        language=lang,
        base_commit=pin.lower() if is_full_sha(pin) else pin,
        merged=is_merged,
        suite_reporter=suite_id,
        suite_command=suite_cmd,
    )

    reasons: list[str] = []
    details: list[str] = []
    hybrid_admit = False

    if refuse_motor_hybrid:
        banned, ban_detail = is_motor_or_hybrid_identity(
            repo,
            source_track=source_track,
            seed_id=seed_id,
        )
        if banned:
            reasons.append(REASON_MOTOR_OR_HYBRID)
            details.append(ban_detail or "motor/hybrid identity refused")

    if require_merged and not is_merged:
        reasons.append(REASON_NOT_MERGED)
        details.append("merged_at missing / PR not merged")

    if require_full_base_sha and not is_full_sha(pin):
        reasons.append(REASON_BASE_SHA_INVALID)
        details.append(f"base_commit must be 40-char hex SHA; got {pin!r}")

    if require_license and not lic_ok:
        reasons.append(lic_reason)
        details.append(f"license gate refused {lic_raw!r} ({lic_reason})")

    # Pure docs/chore: no product source files (or title clearly docs/chore and zeros).
    pure_docs_chore = stats.source_file_count == 0 and (
        stats.docs_chore_file_count > 0 or stats.test_file_count == 0
    )
    title_chore = _looks_like_chore_title(title) and stats.source_file_count == 0
    if pure_docs_chore or title_chore:
        reasons.append(REASON_DOCS_CHORE_ONLY)
        details.append(
            "candidate is pure docs/chore (no product source files); "
            f"docs_chore={list(stats.docs_chore_files)[:8]}"
        )
    elif stats.source_file_count == 0:
        reasons.append(REASON_DOCS_CHORE_ONLY)
        details.append("no product source files after docs/chore/vendor exclusion")

    # Multi-file DeepSWE-min hybrid (VAL-DMED-012):
    # files ≥ min_source_files OR (files ≥ 3 AND added ≥ 500 AND hunks ≥ floor).
    if REASON_DOCS_CHORE_ONLY not in reasons:
        if stats.source_file_count >= min_source_files:
            pass
        elif (
            stats.source_file_count >= PRODUCT_HYBRID_MIN_SOURCE_FILES
            and stats.source_added_lines >= PRODUCT_HYBRID_MIN_ADDED_LINES
            and stats.source_hunk_count >= min_source_hunks
        ):
            hybrid_admit = True
        else:
            reasons.append(REASON_MULTI_FILE_FLOOR)
            details.append(
                f"product source files {stats.source_file_count} < floor {min_source_files} "
                f"and hybrid branch failed (need files≥{PRODUCT_HYBRID_MIN_SOURCE_FILES} AND "
                f"added≥{PRODUCT_HYBRID_MIN_ADDED_LINES} AND hunks≥{min_source_hunks}; "
                f"got added={stats.source_added_lines}, hunks={stats.source_hunk_count})"
            )

    if require_tests and stats.test_file_count < 1:
        reasons.append(REASON_TESTS_MISSING)
        details.append("product keep requires ≥1 real test path change")

    if stats.source_hunk_count < min_source_hunks:
        reasons.append(REASON_SOURCE_HUNKS_BELOW_FLOOR)
        details.append(
            f"source_hunk_count={stats.source_hunk_count} < product floor {min_source_hunks}"
        )

    if require_suite_reporter and not suite_ok:
        reasons.append(REASON_SUITE_REPORTER_UNAVAILABLE)
        details.append(
            f"no suite reporter for language {lang!r}; supported={list(list_reporter_languages())}"
        )

    multi_meta = {
        "product_source_hunk_floor": min_source_hunks,
        "product_multi_file_floor": min_source_files,
        "soft_multi_file_floor": SOFT_MULTI_FILE_FLOOR,
        "multi_file_rule": "files_ge_4_or_hybrid_3",
        "multi_file_hybrid_admit": hybrid_admit,
        "hybrid_min_source_files": PRODUCT_HYBRID_MIN_SOURCE_FILES,
        "hybrid_min_added_lines": PRODUCT_HYBRID_MIN_ADDED_LINES,
        "source_added_lines": stats.source_added_lines,
    }

    if reasons:
        return HardFilterResult(
            accepted=False,
            reason_codes=tuple(dict.fromkeys(reasons)),
            detail="; ".join(details),
            stats=stats,
            meta=multi_meta,
        )

    return HardFilterResult(
        accepted=True,
        reason_codes=(REASON_OK,),
        detail="product hard filter accepted",
        stats=stats,
        meta=multi_meta,
    )


def apply_product_hard_filter(
    *,
    files: Sequence[Mapping[str, Any] | Any],
    base_commit: str | None,
    merged_at: str | None = None,
    merged: bool | None = None,
    language: str | None = None,
    license: str | None = None,
    repo: str | None = None,
    source_track: str | None = None,
    seed_id: str | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> HardFilterResult:
    """Alias of :func:`evaluate_product_hard_filter` for call-site clarity."""
    return evaluate_product_hard_filter(
        files=files,
        base_commit=base_commit,
        merged_at=merged_at,
        merged=merged,
        language=language,
        license=license,
        repo=repo,
        source_track=source_track,
        seed_id=seed_id,
        title=title,
        **kwargs,
    )


def reject_ledger_row(
    result: HardFilterResult,
    *,
    repo: str,
    pr_number: int | None = None,
    discovery_path: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a durable reject/keep ledger row with source_hunk_count + license."""
    row: dict[str, Any] = {
        "repo": repo,
        "pr_number": pr_number,
        "disposition": "accept" if result.accepted else "reject",
        "reason_code": result.reason_code,
        "reason_codes": list(result.reason_codes),
        "detail": result.detail,
        "discovery_path": discovery_path,
        **result.to_ledger_fields(),
    }
    if extra:
        row.update(dict(extra))
    return row


__all__ = [
    "PRODUCT_HYBRID_MIN_ADDED_LINES",
    "PRODUCT_HYBRID_MIN_SOURCE_FILES",
    "PRODUCT_MULTI_FILE_FLOOR",
    "PRODUCT_SOURCE_HUNK_FLOOR",
    "REASON_BASE_SHA_INVALID",
    "REASON_DOCS_CHORE_ONLY",
    "REASON_LICENSE_COPYLEFT",
    "REASON_LICENSE_MISSING",
    "REASON_LICENSE_REJECTED",
    "REASON_LICENSE_UNKNOWN",
    "REASON_MOTOR_OR_HYBRID",
    "REASON_MULTI_FILE_FLOOR",
    "REASON_NOT_MERGED",
    "REASON_OK",
    "REASON_PURE_CHORE_MARKERS",
    "REASON_SOURCE_HUNKS_BELOW_FLOOR",
    "REASON_SUITE_REPORTER_UNAVAILABLE",
    "REASON_TESTS_MISSING",
    "SOFT_MULTI_FILE_FLOOR",
    "HardFilterResult",
    "HardFilterStats",
    "apply_product_hard_filter",
    "classify_pr_files",
    "count_unified_diff_hunks",
    "evaluate_product_hard_filter",
    "is_chore_path",
    "is_docs_or_chore_path",
    "is_docs_path",
    "is_product_source_path",
    "measure_source_hunk_count",
    "reject_ledger_row",
    "suite_reporter_detectable",
]
