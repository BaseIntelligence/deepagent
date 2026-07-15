"""Unit tests for real-repo discover (VAL-MINE-001..007)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from swe_factory.oracle.gates import count_files_in_patch
from swe_factory.producers.pr_miner import (
    PrFileChange,
    is_test_path,
    multi_file_source_filter,
)
from swe_factory.sources.discover import (
    DiscoverError,
    build_local_git_repo_with_range,
    discover_merge_range_from_git,
    discover_offline_fixture,
    extract_test_patch_from_files,
    is_full_sha,
    is_real_repository_url,
    looks_like_fake_sha,
    require_full_sha,
    sanitize_api_candidate,
    try_discover_from_files,
    validate_hybrid_curated,
    write_candidate_artifacts,
)
from swe_factory.sources.license_gate import LicenseGateError

_FULL_SHA = "b" * 40
_FULL_SHA_B = "c" * 40


def _source(path: str, patch: str = "@@ -1 +1 @@\n-a\n+b\n") -> PrFileChange:
    return PrFileChange(path=path, status="modified", patch=patch)


def _test(path: str, patch: str = "@@ -0,0 +1 @@\n+assert True\n") -> PrFileChange:
    return PrFileChange(path=path, status="modified", patch=patch)


# --- VAL-MINE-002: full 40-char SHA ---


def test_is_full_sha_requires_40_hex() -> None:
    assert is_full_sha("a" * 40)
    assert is_full_sha("0123456789abcdef" * 2 + "01234567")
    assert not is_full_sha("abc123")  # short
    assert not is_full_sha("a" * 39)
    assert not is_full_sha("a" * 41)
    assert not is_full_sha("fixture000000000000000000000000000000001")  # non-hex letters
    assert not is_full_sha("main")
    with pytest.raises(DiscoverError, match="40-char"):
        require_full_sha("deadbee")
    assert require_full_sha("A" * 40) == "a" * 40


def test_looks_like_fake_sha() -> None:
    assert looks_like_fake_sha("a100000000000000000000000000000000000001")
    assert looks_like_fake_sha("fixture" + "0" * 33)
    assert not looks_like_fake_sha("979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe")


# --- VAL-MINE-001 / 006: multi-file + held-out tests ---


def test_sanitize_accepts_multi_file_source_plus_tests() -> None:
    files = [
        _source("pkg/a.py", "@@ -1 +1 @@\n-old_a\n+new_a\n"),
        _source("pkg/b.py", "@@ -1 +1 @@\n-old_b\n+new_b\n"),
        _test("tests/test_a.py"),
    ]
    candidate = sanitize_api_candidate(
        repo="owner/demo",
        base_commit=_FULL_SHA,
        files=files,
        license="MIT",
        language="python",
        pr_number=12,
    )
    assert candidate.accepted
    assert is_full_sha(candidate.base_commit)
    assert candidate.base_commit == _FULL_SHA
    assert len(candidate.gold_files) >= 2
    assert candidate.test_files
    assert candidate.test_patch.strip()
    assert all(not is_test_path(p) for p in candidate.gold_files)
    assert candidate.repository_url.startswith("https://github.com/")
    assert "owner/demo" in candidate.repository_url
    # gold touches product sources only
    gold_paths = count_files_in_patch(candidate.gold_patch)
    assert "pkg/a.py" in gold_paths
    assert "pkg/b.py" in gold_paths
    assert not any(is_test_path(p) for p in gold_paths)
    test_paths = count_files_in_patch(candidate.test_patch)
    assert any(is_test_path(p) for p in test_paths)


def test_sanitize_rejects_single_source() -> None:
    files = [_source("only.py"), _test("tests/test_only.py")]
    with pytest.raises(DiscoverError, match="multi-file"):
        sanitize_api_candidate(
            repo="owner/demo",
            base_commit=_FULL_SHA,
            files=files,
            license="MIT",
        )


def test_sanitize_rejects_test_only() -> None:
    files = [_test("tests/test_a.py"), _test("tests/test_b.py")]
    with pytest.raises(DiscoverError, match="multi-file"):
        sanitize_api_candidate(
            repo="owner/demo",
            base_commit=_FULL_SHA,
            files=files,
            license="MIT",
        )


def test_sanitize_rejects_short_sha() -> None:
    files = [_source("a.py"), _source("b.py"), _test("tests/t.py")]
    with pytest.raises(DiscoverError, match="40-char"):
        sanitize_api_candidate(
            repo="owner/demo",
            base_commit="abc1234",
            files=files,
            license="MIT",
        )


# --- VAL-MINE-003: copyleft ---


def test_sanitize_rejects_copyleft_license() -> None:
    files = [_source("a.py"), _source("b.py"), _test("tests/t.py")]
    with pytest.raises((DiscoverError, LicenseGateError), match="copyleft|license"):
        sanitize_api_candidate(
            repo="owner/gpl-demo",
            base_commit=_FULL_SHA,
            files=files,
            license="GPL-3.0",
        )
    # try_ wrapper returns reject reason_code
    kept, rejected = try_discover_from_files(
        repo="owner/gpl-demo",
        base_commit=_FULL_SHA,
        files=files,
        license="GPL-3.0",
    )
    assert kept is None
    assert rejected is not None
    assert "license" in rejected.reason_code
    assert "copyleft" in rejected.reason_code or "license" in rejected.reason_code


def test_sanitize_rejects_missing_license() -> None:
    files = [_source("a.py"), _source("b.py"), _test("tests/t.py")]
    with pytest.raises((DiscoverError, LicenseGateError)):
        sanitize_api_candidate(
            repo="owner/demo",
            base_commit=_FULL_SHA,
            files=files,
            license="",
        )


# --- VAL-MINE-004: offline fixture ---


def test_discover_offline_fixture_no_network(tmp_path: Path) -> None:
    report = discover_offline_fixture(work_root=tmp_path / "out")
    assert report.offline is True
    assert report.network_required is False
    assert report.provider_calls == 0
    assert report.keep_count == 1
    candidate = report.kept[0]
    assert candidate.kind == "offline_fixture"
    assert len(candidate.gold_files) >= 2
    assert candidate.test_patch.strip()
    assert candidate.history_authority == "offline_fixture"
    # artifact layout
    cases = list((tmp_path / "out").iterdir())
    assert cases
    case = cases[0]
    assert (case / "gold.patch").is_file()
    assert (case / "test.patch").is_file()
    assert (case / "candidate.json").is_file()
    meta = json.loads((case / "candidate.json").read_text(encoding="utf-8"))
    assert meta.get("meta", {}).get("provider_calls") == 0
    assert meta.get("meta", {}).get("network_required") is False
    assert (case / "test.patch").read_text(encoding="utf-8").strip()
    gold = (case / "gold.patch").read_text(encoding="utf-8")
    for path in count_files_in_patch(gold):
        assert not is_test_path(path)


# --- VAL-MINE-005: hybrid curated real URL + SHA ---


def test_hybrid_curated_requires_real_repo_and_sha() -> None:
    with pytest.raises(DiscoverError, match="repository_url"):
        validate_hybrid_curated(
            repository_url="file://fixtures/tiny_green",
            base_commit=_FULL_SHA,
            license="MIT",
        )
    with pytest.raises(DiscoverError, match="base_commit"):
        validate_hybrid_curated(
            repository_url="https://github.com/mahmoud/boltons",
            base_commit="a100000000000000000000000000000000000001",
            license="MIT",
        )
    with pytest.raises(DiscoverError, match="base_commit|40"):
        validate_hybrid_curated(
            repository_url="https://github.com/mahmoud/boltons",
            base_commit="deadbee",
            license="MIT",
        )
    # real public + real full SHA ok
    validate_hybrid_curated(
        repository_url="https://github.com/mahmoud/boltons",
        base_commit="979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe",
        license="BSD-3-Clause",
    )


def test_hybrid_curated_motor_rejected_as_discover_candidate() -> None:
    files = [_source("a.py"), _source("b.py"), _test("tests/t.py")]
    with pytest.raises(DiscoverError, match="repository_url|base_commit|hybrid|curated|fake"):
        sanitize_api_candidate(
            repo="fixtures/harbor_motors/python_orders",
            repository_url="file://fixtures/harbor_motors/python_orders",
            base_commit="a100000000000000000000000000000000000001",
            files=files,
            license="MIT",
            kind="curated",
        )


def test_is_real_repository_url() -> None:
    assert is_real_repository_url("https://github.com/owner/name")
    assert is_real_repository_url("https://github.com/owner/name.git")
    assert not is_real_repository_url("file://foo")
    assert not is_real_repository_url("fixtures/tiny_green")
    assert not is_real_repository_url("")


# --- VAL-MINE-007: git history authority ---


def test_discover_from_local_git_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base_files = {
        "pkg/a.py": "def a():\n    return 1\n",
        "pkg/b.py": "def b():\n    return 2\n",
        "tests/test_a.py": "def test_placeholder():\n    assert True\n",
    }
    head_files = {
        "pkg/a.py": "def a():\n    return 10\n",
        "pkg/b.py": "def b():\n    return 20\n",
        "tests/test_a.py": "from pkg.a import a\n\ndef test_a():\n    assert a() == 10\n",
        "tests/test_b.py": "from pkg.b import b\n\ndef test_b():\n    assert b() == 20\n",
    }
    base_sha, head_sha = build_local_git_repo_with_range(
        repo, base_files=base_files, head_files=head_files
    )
    assert is_full_sha(base_sha) and is_full_sha(head_sha)
    assert base_sha != head_sha

    candidate = discover_merge_range_from_git(
        repo,
        base=base_sha,
        head=head_sha,
        repo="owner/local-demo",
        repository_url="https://github.com/owner/local-demo",
        license="MIT",
        language="python",
        title="Multi-file fix via git",
        pr_number=99,
    )
    assert candidate.history_authority == "git"
    assert candidate.base_commit == base_sha
    assert len(candidate.gold_files) >= 2
    assert "pkg/a.py" in candidate.gold_files or "pkg/a.py" in str(candidate.gold_files)
    assert candidate.test_files
    assert candidate.test_patch.strip()
    # Git-derived gold must not include held-out tests
    for path in count_files_in_patch(candidate.gold_patch):
        assert not is_test_path(path)
    # Round-trip artifacts
    case = write_candidate_artifacts(candidate, tmp_path / "artifacts")
    assert (case / "gold.patch").read_text(encoding="utf-8")
    assert (case / "test.patch").read_text(encoding="utf-8")


def test_local_git_rejects_under_multi_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base_files = {
        "pkg/a.py": "def a():\n    return 1\n",
        "tests/test_a.py": "def test_placeholder():\n    assert True\n",
    }
    head_files = {
        "pkg/a.py": "def a():\n    return 2\n",
        "tests/test_a.py": "from pkg.a import a\n\ndef test_a():\n    assert a() == 2\n",
    }
    base_sha, head_sha = build_local_git_repo_with_range(
        repo, base_files=base_files, head_files=head_files
    )
    with pytest.raises(DiscoverError, match="multi-file"):
        discover_merge_range_from_git(
            repo,
            base=base_sha,
            head=head_sha,
            repo="owner/local-demo",
            repository_url="https://github.com/owner/local-demo",
            license="MIT",
        )


def test_extract_test_patch_only_tests() -> None:
    patch = extract_test_patch_from_files(
        [
            _source("a.py"),
            _test("tests/test_x.py", "@@ -0,0 +1,2 @@\n+def test_x():\n+    assert 1\n"),
        ]
    )
    assert "tests/test_x.py" in patch
    assert "a.py" not in count_files_in_patch(patch)


def test_report_json_funnel_shape(tmp_path: Path) -> None:
    report = discover_offline_fixture(work_root=tmp_path)
    payload = report.to_dict()
    assert payload["ok"] is True
    assert payload["provider_calls"] == 0
    assert payload["offline"] is True
    assert payload["keep_count"] == 1
    assert re.fullmatch(r".+", payload["kept"][0]["base_commit"])


def test_multi_file_filter_still_shared_with_pr_miner() -> None:
    assert multi_file_source_filter([_source("a.py"), _source("b.py"), _test("tests/t.py")])
    assert not multi_file_source_filter([_source("a.py"), _test("tests/t.py")])
