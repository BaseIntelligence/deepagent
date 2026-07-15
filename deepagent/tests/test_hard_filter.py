"""Product hard filter unit tests (M14 VAL-LHARD / VAL-LMINE).

Covers:
- Boundary source_hunk_count 9 reject / 10 accept (VAL-LHARD-001)
- docs/chore refuse (VAL-LHARD-003)
- multi-file + test path floor (VAL-LHARD-004)
- merged_at + 40-char base SHA (VAL-LMINE-002)
- motor/hybrid refuse (VAL-LMINE-005)
- permissive license fail-closed (VAL-LHARD-005)
- ledger fields include source_hunk_count + license
"""

from __future__ import annotations

from typing import Any

import pytest

from swe_factory.oracle.gates import MULTI_FILE_FLOOR
from swe_factory.producers.hard_filter import (
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
    REASON_BASE_SHA_INVALID,
    REASON_DOCS_CHORE_ONLY,
    REASON_LICENSE_COPYLEFT,
    REASON_LICENSE_MISSING,
    REASON_LICENSE_UNKNOWN,
    REASON_MOTOR_OR_HYBRID,
    REASON_MULTI_FILE_FLOOR,
    REASON_NOT_MERGED,
    REASON_OK,
    REASON_SOURCE_HUNKS_BELOW_FLOOR,
    REASON_SUITE_REPORTER_UNAVAILABLE,
    REASON_TESTS_MISSING,
    SOFT_MULTI_FILE_FLOOR,
    count_unified_diff_hunks,
    evaluate_product_hard_filter,
    is_chore_path,
    is_docs_path,
    is_product_source_path,
    measure_source_hunk_count,
    reject_ledger_row,
)
from swe_factory.producers.pr_miner import PrFileChange, multi_file_source_filter

_BASE = "a" * 40


def _hunk(n: int = 1, body: str = "line") -> str:
    """Build n unified-diff hunks with minimal +/- lines."""
    parts: list[str] = []
    for i in range(n):
        parts.append(f"@@ -{i + 1},1 +{i + 1},1 @@\n-old_{body}_{i}\n+new_{body}_{i}\n")
    return "".join(parts)


def _src(path: str, hunks: int = 1) -> dict[str, Any]:
    return {
        "filename": path,
        "status": "modified",
        "patch": _hunk(hunks, path.replace("/", "_")),
        "additions": hunks,
        "deletions": hunks,
    }


def _test(path: str = "tests/test_mod.py") -> dict[str, Any]:
    return {
        "filename": path,
        "status": "added",
        "patch": "@@ -0,0 +1,3 @@\n+def test_ok():\n+    assert True\n",
        "additions": 3,
        "deletions": 0,
    }


def _hard_eligible_files(*, source_hunks: int) -> list[dict[str, Any]]:
    """Two product source files totaling exactly ``source_hunks`` hunks + one test."""
    if source_hunks < 2:
        raise ValueError("need at least 2 hunks to split across two source files")
    left = source_hunks // 2
    right = source_hunks - left
    return [
        _src("pkg/a.py", left),
        _src("pkg/b.py", right),
        _test(),
    ]


def test_product_floors_constants() -> None:
    assert PRODUCT_SOURCE_HUNK_FLOOR == 10
    assert PRODUCT_MULTI_FILE_FLOOR >= 2
    assert SOFT_MULTI_FILE_FLOOR == MULTI_FILE_FLOOR == 2
    # Soft floor still available for offline engineering filters.
    soft_ok = multi_file_source_filter(
        [
            PrFileChange(path="a.py", status="modified", patch=_hunk(1)),
            PrFileChange(path="b.py", status="modified", patch=_hunk(1)),
            PrFileChange(path="tests/t.py", status="added", patch=_hunk(1)),
        ],
        min_source_files=SOFT_MULTI_FILE_FLOOR,
        require_tests=True,
    )
    assert soft_ok is True


def test_count_unified_diff_hunks() -> None:
    assert count_unified_diff_hunks(None) == 0
    assert count_unified_diff_hunks("") == 0
    assert count_unified_diff_hunks(_hunk(1)) == 1
    assert count_unified_diff_hunks(_hunk(9)) == 9
    assert count_unified_diff_hunks(_hunk(10)) == 10
    assert count_unified_diff_hunks(_hunk(3) + "\n" + _hunk(2)) == 5


def test_measure_source_hunks_ignores_tests_and_docs() -> None:
    files = [
        _src("pkg/a.py", 4),
        _src("pkg/b.py", 3),
        _test("tests/test_a.py"),
        {
            "filename": "README.md",
            "status": "modified",
            "patch": _hunk(20, "readme"),
        },
        {
            "filename": ".github/workflows/ci.yml",
            "status": "modified",
            "patch": _hunk(5, "ci"),
        },
    ]
    # Only product source hunks: 4+3 = 7
    assert measure_source_hunk_count(files) == 7


def test_rejects_9_source_hunks() -> None:
    """VAL-LHARD-001 boundary: 9 source hunks hard-reject."""
    files = _hard_eligible_files(source_hunks=9)
    assert measure_source_hunks_sum(files) == 9
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="MIT",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_SOURCE_HUNKS_BELOW_FLOOR in result.reason_codes
    assert result.stats.source_hunk_count == 9
    assert result.stats.source_hunk_count < PRODUCT_SOURCE_HUNK_FLOOR


def test_accepts_10_source_hunks() -> None:
    """VAL-LHARD-001 boundary: 10 source hunks keep-eligible."""
    files = _hard_eligible_files(source_hunks=10)
    assert measure_source_hunk_count(files) == 10
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="MIT",
        repo="owner/demo",
    )
    assert result.accepted is True
    assert result.reason_code == REASON_OK
    assert result.stats.source_hunk_count == 10
    assert result.stats.source_file_count >= 2
    assert result.stats.test_file_count >= 1
    assert result.stats.license.lower() == "mit"


def measure_source_hunks_sum(files: list[dict[str, Any]]) -> int:
    return measure_source_hunk_count(files)


def test_rejects_not_merged() -> None:
    files = _hard_eligible_files(source_hunks=10)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at=None,
        merged=False,
        language="python",
        license="MIT",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_NOT_MERGED in result.reason_codes


def test_rejects_short_base_sha() -> None:
    files = _hard_eligible_files(source_hunks=10)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit="deadbeef",  # not 40-char
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="MIT",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_BASE_SHA_INVALID in result.reason_codes


def test_docs_only_rejected() -> None:
    files = [
        {
            "filename": "README.md",
            "status": "modified",
            "patch": _hunk(12, "docs"),
        },
        {
            "filename": "docs/guide.rst",
            "status": "added",
            "patch": _hunk(5, "guide"),
        },
    ]
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="MIT",
        repo="owner/demo",
        title="docs: fix typo",
    )
    assert result.accepted is False
    assert REASON_DOCS_CHORE_ONLY in result.reason_codes
    assert result.stats.source_hunk_count == 0


def test_chore_only_rejected() -> None:
    files = [
        {
            "filename": ".github/workflows/ci.yml",
            "status": "modified",
            "patch": _hunk(3, "ci"),
        },
        {
            "filename": "package.json",
            "status": "modified",
            "patch": _hunk(2, "pkg"),
        },
    ]
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="javascript",
        license="MIT",
        repo="owner/demo",
        title="chore: bump deps",
    )
    assert result.accepted is False
    assert REASON_DOCS_CHORE_ONLY in result.reason_codes


def test_single_source_file_rejected() -> None:
    files = [
        _src("pkg/only.py", 12),
        _test(),
    ]
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="MIT",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_MULTI_FILE_FLOOR in result.reason_codes
    assert result.stats.source_file_count == 1


def test_missing_tests_rejected() -> None:
    files = [
        _src("pkg/a.py", 6),
        _src("pkg/b.py", 6),
    ]
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="MIT",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_TESTS_MISSING in result.reason_codes


def test_copyleft_license_rejected() -> None:
    files = _hard_eligible_files(source_hunks=10)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="GPL-3.0",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_LICENSE_COPYLEFT in result.reason_codes


def test_missing_license_fail_closed() -> None:
    files = _hard_eligible_files(source_hunks=10)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_LICENSE_MISSING in result.reason_codes


def test_unknown_license_fail_closed() -> None:
    files = _hard_eligible_files(source_hunks=10)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="Proprietary-Internal-Only",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_LICENSE_UNKNOWN in result.reason_codes


def test_motor_hybrid_identity_rejected() -> None:
    files = _hard_eligible_files(source_hunks=10)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="MIT",
        repo="fixtures/harbor_motors/python_orders",
        seed_id="harbor_python_orders",
        source_track="hybrid_curated",
    )
    assert result.accepted is False
    assert REASON_MOTOR_OR_HYBRID in result.reason_codes


def test_suite_reporter_required() -> None:
    files = _hard_eligible_files(source_hunks=10)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="cobol",  # no reporter
        license="MIT",
        repo="owner/demo",
    )
    assert result.accepted is False
    assert REASON_SUITE_REPORTER_UNAVAILABLE in result.reason_codes


def test_reject_ledger_row_includes_hunks_and_license() -> None:
    files = _hard_eligible_files(source_hunks=9)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="python",
        license="Apache-2.0",
        repo="owner/demo",
    )
    row = reject_ledger_row(
        result,
        repo="owner/demo",
        pr_number=42,
        discovery_path="search",
    )
    assert row["source_hunk_count"] == 9
    assert row["license"]
    assert "apache" in row["license"].lower() or row["license"] == "Apache-2.0"
    assert row["disposition"] == "reject"
    assert row["reason_code"] == REASON_SOURCE_HUNKS_BELOW_FLOOR
    assert row["discovery_path"] == "search"
    assert row["base_commit"] == _BASE
    assert row["product_source_hunk_floor"] == 10


def test_accept_ledger_row_includes_hunks_and_license() -> None:
    files = _hard_eligible_files(source_hunks=12)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at="2026-01-01T00:00:00Z",
        language="go",
        license="BSD-3-Clause",
        repo="owner/demo",
    )
    assert result.accepted is True
    row = reject_ledger_row(result, repo="owner/demo", pr_number=7, discovery_path="list_pulls")
    assert row["source_hunk_count"] >= 10
    assert row["license"]
    assert row["disposition"] == "accept"
    assert row["hard_filter_accepted"] is True
    assert row["discovery_path"] == "list_pulls"


def test_path_classifiers() -> None:
    assert is_docs_path("README.md")
    assert is_docs_path("docs/intro.rst")
    assert is_chore_path(".github/workflows/ci.yml")
    assert is_chore_path("package.json")
    assert is_product_source_path("pkg/core.py")
    assert not is_product_source_path("tests/test_core.py")
    assert not is_product_source_path("README.md")


def test_soft_floor_does_not_equal_product_hunk_floor() -> None:
    """Soft MULTI_FILE_FLOOR=2 must not be confused with product 10-hunk hard floor."""
    assert SOFT_MULTI_FILE_FLOOR == 2
    assert PRODUCT_SOURCE_HUNK_FLOOR == 10
    assert PRODUCT_SOURCE_HUNK_FLOOR > SOFT_MULTI_FILE_FLOOR


def test_pr_miner_product_mode_rejects_9_accepts_10() -> None:
    """PrMiner product_mode applies hard floor with fixed source_hunk definition."""
    from swe_factory.producers.pr_miner import PrMineError, PrMiner
    from swe_factory.sources.github import DictGitHubTransport, GitHubClient

    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client, product_mode=True, license="MIT")

    def _select(hunks: int) -> Any:
        return miner.select_from_files(
            repo="owner/demo",
            number=100 + hunks,
            title="Hard multi-file fix",
            body="real product PR",
            base_commit=_BASE,
            merge_commit_sha="c" * 40,
            html_url="https://github.com/owner/demo/pull/1",
            files_payload=_hard_eligible_files(source_hunks=hunks),
            language="python",
            license="MIT",
            require_full_base_sha=True,
            enforce_license=True,
            merged_at="2026-01-01T00:00:00Z",
            product_hard_filter=True,
        )

    with pytest.raises(PrMineError, match="source_hunk|hard filter"):
        _select(9)

    pr = _select(10)
    assert pr.source_hunk_count == 10
    assert pr.license == "MIT"
    assert pr.merged_at
    candidate = miner.produce(pr, instance_suffix="hard10", run_stub_oracle=True)
    assert candidate.provenance["source_hunk_count"] == 10
    assert candidate.provenance["license"] == "MIT"


def test_pr_miner_product_mode_rejects_missing_merged_at() -> None:
    """product_mode must require a real merged_at; never invent merged/merged_at."""
    from swe_factory.producers.pr_miner import PrMineError, PrMiner
    from swe_factory.sources.github import DictGitHubTransport, GitHubClient

    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client, product_mode=True, license="MIT")
    with pytest.raises(PrMineError, match=r"not_merged|merged") as excinfo:
        miner.select_from_files(
            repo="owner/demo",
            number=501,
            title="Looks hard but not merged",
            body="open PR must not pass product hard filter",
            base_commit=_BASE,
            merge_commit_sha=None,
            html_url="https://github.com/owner/demo/pull/501",
            files_payload=_hard_eligible_files(source_hunks=10),
            language="python",
            license="MIT",
            require_full_base_sha=True,
            enforce_license=True,
            # intentionally omit merged_at
            product_hard_filter=True,
        )
    msg = str(excinfo.value)
    assert REASON_NOT_MERGED in msg


def test_pr_miner_product_mode_rejects_empty_merged_at() -> None:
    """Empty/falsey merged_at must fail product hard filter with REASON_NOT_MERGED."""
    from swe_factory.producers.pr_miner import PrMineError, PrMiner
    from swe_factory.sources.github import DictGitHubTransport, GitHubClient

    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client, product_mode=True, license="MIT")

    for bad_merged_at in ("", None):
        with pytest.raises(PrMineError, match=r"not_merged|merged"):
            miner.select_from_files(
                repo="owner/demo",
                number=502,
                title="Empty merge timestamp",
                body="falsey merge cannot pass product filter",
                base_commit=_BASE,
                merge_commit_sha=None,
                html_url="https://github.com/owner/demo/pull/502",
                files_payload=_hard_eligible_files(source_hunks=10),
                language="python",
                license="MIT",
                require_full_base_sha=True,
                enforce_license=True,
                merged_at=bad_merged_at,  # type: ignore[arg-type]
                product_hard_filter=True,
            )


def test_evaluate_product_require_merged_true_rejects_missing() -> None:
    """Direct hard-filter path: require_merged=True + missing merge → NOT_MERGED."""
    files = _hard_eligible_files(source_hunks=10)
    result = evaluate_product_hard_filter(
        files=files,
        base_commit=_BASE,
        merged_at=None,
        merged=None,
        language="python",
        license="MIT",
        repo="owner/demo",
        require_merged=True,
    )
    assert result.accepted is False
    assert REASON_NOT_MERGED in result.reason_codes
    assert result.reason_code == REASON_NOT_MERGED
