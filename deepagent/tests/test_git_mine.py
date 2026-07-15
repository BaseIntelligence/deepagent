"""Unit tests for git-clone-only allowlist miner + VAL-OXY-005 probe honesty."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_factory.sources.allowlist import REMOTE_SEEDS, SCALE_LANGUAGES, SeedRepo
from swe_factory.sources.discover import (
    build_local_git_repo_with_range,
    is_full_sha,
)
from swe_factory.sources.git_mine import (
    list_merge_ranges,
    list_non_merge_ranges,
    mine_allowlist_git_only,
    mine_seed_merges,
    probe_oxylabs_live,
)


def _seed_for_local(tmp_path: Path) -> tuple[SeedRepo, Path, str, str]:
    """Build an ephemeral git repo with multi-file source+test change and a SeedRepo."""
    repo = tmp_path / "ephemeral_repo"
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
    # Create a second parent merge-ish commit? list_non_merge works with consecutive.
    seed = SeedRepo(
        seed_id="local_py_demo",
        language="python",
        repo="owner/local-demo",
        base_commit=base_sha,
        license="MIT",
        description="ephemeral git mine seed",
        modular=True,
        mine_priority=1,
    )
    return seed, repo, base_sha, head_sha


def test_list_non_merge_ranges_from_ephemeral(tmp_path: Path) -> None:
    _, repo, base_sha, head_sha = _seed_for_local(tmp_path)
    ranges = list_non_merge_ranges(repo, max_commits=10)
    assert ranges
    assert any(r.merge_sha == head_sha and r.base_sha == base_sha for r in ranges)


def test_mine_seed_merges_on_local_clone_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, repo, _base, _head = _seed_for_local(tmp_path)

    # Point clone_seed_repo at the prebuilt repo path
    def _fake_clone(seed_obj, *, dest_root, depth=80, reuse=True, **_kwargs):  # noqa: ANN001
        del seed_obj, dest_root, depth, reuse
        return repo

    monkeypatch.setattr(
        "swe_factory.sources.git_mine.clone_seed_repo",
        _fake_clone,
    )
    keeps, rejects, stats = mine_seed_merges(
        seed,
        dest_root=tmp_path / "clones",
        max_merges=20,
        max_keeps=3,
        allow_non_merge_fallback=True,
    )
    assert stats.clone_ok is True
    assert stats.kept >= 1
    assert keeps
    cand = keeps[0]
    assert cand.history_authority == "git"
    assert cand.http_metadata_source == "none"
    assert is_full_sha(cand.base_commit)
    assert len(cand.gold_files) >= 2
    assert cand.test_files
    assert cand.repository_url.startswith("https://")
    assert cand.meta.get("mine_mode") == "git_clone_only"
    # rejects may be empty or contain under-filter ranges — fine either way
    assert isinstance(rejects, list)


def test_mine_allowlist_git_only_offline_with_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, repo, _base, _head = _seed_for_local(tmp_path)

    def _fake_clone(seed_obj, *, dest_root, depth=80, reuse=True, **_kwargs):  # noqa: ANN001
        del seed_obj, dest_root, depth, reuse
        return repo

    monkeypatch.setattr(
        "swe_factory.sources.git_mine.clone_seed_repo",
        _fake_clone,
    )
    # Only mine our stub seed
    monkeypatch.setattr(
        "swe_factory.sources.git_mine.remote_mine_seeds",
        lambda language=None: [seed] if language in (None, "python") else [],
    )
    report = mine_allowlist_git_only(
        work_root=tmp_path / "mine_out",
        target_candidates=5,
        max_merges_per_seed=20,
        max_keeps_per_seed=3,
        write_artifacts=True,
    )
    assert report.mode == "git_clone_only"
    assert report.provider_calls == 0
    assert report.history_authority == "git"
    assert report.oxylabs_status == "not_used"
    assert report.keep_count >= 1
    assert (tmp_path / "mine_out" / "git_mine_report.json").is_file()
    assert (tmp_path / "mine_out" / "candidates.jsonl").is_file()
    assert (tmp_path / "mine_out" / "language_stats.json").is_file()
    stats = json.loads((tmp_path / "mine_out" / "language_stats.json").read_text(encoding="utf-8"))
    assert "language_kept" in stats
    assert "under_supply" in stats
    # language kept should include python count
    assert stats["language_kept"].get("python", 0) >= 1
    # ensure every scale language appears in under_supply OR kept histogram keys
    payload = report.to_dict()
    for lang in SCALE_LANGUAGES:
        assert lang in payload["language_kept"] or any(lang in u for u in payload["under_supply"])


def test_probe_oxylabs_blocked_without_creds() -> None:
    evidence = probe_oxylabs_live(env={})
    assert evidence["assertion"] == "VAL-OXY-005"
    assert evidence["credentials_present"] is False
    assert evidence["status"] == "blocked"
    assert evidence["ok"] is False
    assert "OXYLABS" in evidence["reason"]
    # never fake pass
    assert evidence["ok"] is not True or evidence["status"] != "passed"


def test_probe_oxylabs_with_mock_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from swe_factory.sources.oxylabs import (
        DictOxylabsTransport,
        OxylabsClient,
        OxylabsCredentials,
        OxylabsFetchResult,
    )

    transport = DictOxylabsTransport(
        default={
            "results": [
                {
                    "content": "<html>repo</html>",
                    "status_code": 200,
                    "url": "https://github.com/psf/requests",
                }
            ],
            "job": {"id": "j-1"},
        }
    )
    client = OxylabsClient(
        credentials=OxylabsCredentials("u", "p"),
        transport=transport,
    )

    class _CM:
        def __enter__(self) -> OxylabsClient:
            return client

        def __exit__(self, *a: object) -> None:
            return None

    import swe_factory.sources.oxylabs as oxy

    monkeypatch.setattr(oxy, "has_oxylabs_credentials", lambda env=None: True)
    monkeypatch.setattr(
        oxy.OxylabsClient,
        "from_env",
        classmethod(lambda cls, **kw: _CM()),
    )

    evidence = probe_oxylabs_live(env={"OXYLABS_USERNAME": "u", "OXYLABS_PASSWORD": "p"})
    assert evidence["status"] == "passed"
    assert evidence["ok"] is True
    assert evidence["content_bytes"] > 0
    assert isinstance(
        OxylabsFetchResult(
            url="https://github.com/psf/requests",
            status_code=200,
            content="x",
            job_id=None,
        ),
        OxylabsFetchResult,
    )


def test_list_merge_ranges_empty_repo_without_merges(tmp_path: Path) -> None:
    _, repo, _, _ = _seed_for_local(tmp_path)
    # linear history only → list_merge_ranges returns empty; non-merge fallback used by miner
    merges = list_merge_ranges(repo, max_merges=10)
    assert merges == [] or all(is_full_sha(m.merge_sha) for m in merges)


def test_remote_seed_count_feeds_m8_target() -> None:
    """Inventory must be large enough that 6 keeps/seed can reach ≥30."""
    assert len(REMOTE_SEEDS) * 6 >= 30
