"""Unit tests for the Stage-0 source registry (offline, no network).

Covers the m2-sources contract assertions:
- VAL-ENV-015: each RepoSpec records non-empty commit/commit_date/language/
  license and the curated registry covers Python, JS/TS, and Go.
- VAL-ENV-016: the recorded commit is a pinned, full 40-hex SHA (never a branch
  tip / short ref) and the checkout commands target exactly that SHA.
- VAL-ENV-017: the per-repo instance cap is enforced and tracked - N accepted,
  the (N+1)th rejected with a cap reason, usage never exceeding the cap.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from swe_forge.forge.cli import app as forge_app
from swe_forge.forge.models import (
    SUPPORTED_LANGUAGES,
    InstanceGrant,
    ModelError,
    RepoSpec,
)
from swe_forge.forge.sources import (
    SourceError,
    SourceRegistry,
    UnknownRepoError,
    build_source_registry,
)

runner = CliRunner()

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_GOOD_SHA = "77db208ab4ae0cd2061d909fe222a1db72867850"


def make_spec(**overrides: object) -> RepoSpec:
    """Build a valid RepoSpec, overriding individual fields for negative tests."""
    base: dict[str, object] = {
        "repo_id": "acme/widget",
        "url": "https://github.com/acme/widget.git",
        "commit": _GOOD_SHA,
        "commit_date": "2026-02-25T11:10:21Z",
        "language": "python",
        "license": "MIT",
        "instance_cap": 3,
    }
    base.update(overrides)
    return RepoSpec(**base)  # type: ignore[arg-type]


class TestRepoSpecValidation:
    def test_valid_spec_normalizes_commit_and_language(self) -> None:
        spec = make_spec(commit=_GOOD_SHA.upper(), language="Python")
        assert spec.commit == _GOOD_SHA
        assert spec.language == "python"

    @pytest.mark.parametrize(
        "field",
        ["repo_id", "url", "commit", "commit_date", "license"],
    )
    def test_empty_required_field_rejected(self, field: str) -> None:
        with pytest.raises(ModelError):
            make_spec(**{field: "   "})

    @pytest.mark.parametrize(
        "bad_commit",
        ["main", "v1.2.3", "77db208", "x" * 40, _GOOD_SHA + "ab", "HEAD"],
    )
    def test_non_pinned_commit_rejected(self, bad_commit: str) -> None:
        with pytest.raises(ModelError):
            make_spec(commit=bad_commit)

    def test_unknown_language_rejected(self) -> None:
        with pytest.raises(ModelError):
            make_spec(language="rust")

    def test_bad_commit_date_rejected(self) -> None:
        with pytest.raises(ModelError):
            make_spec(commit_date="not-a-date")

    @pytest.mark.parametrize("cap", [0, -1])
    def test_cap_below_one_rejected(self, cap: int) -> None:
        with pytest.raises(ModelError):
            make_spec(instance_cap=cap)

    def test_used_above_cap_rejected(self) -> None:
        with pytest.raises(ModelError):
            make_spec(instance_cap=2, used=3)

    def test_negative_used_rejected(self) -> None:
        with pytest.raises(ModelError):
            make_spec(used=-1)


class TestRepoSpecCap:
    def test_cap_enforced_and_usage_tracked(self) -> None:
        spec = make_spec(instance_cap=3)
        grants = [spec.acquire() for _ in range(3)]

        assert all(g.accepted for g in grants)
        assert [g.instance_index for g in grants] == [1, 2, 3]
        assert spec.used == 3
        assert spec.remaining == 0
        assert spec.at_cap is True

    def test_over_cap_request_rejected_with_reason(self) -> None:
        spec = make_spec(instance_cap=2)
        spec.acquire()
        spec.acquire()
        rejected = spec.acquire()

        assert rejected.accepted is False
        assert rejected.instance_index == 0
        assert "per-repo cap reached" in rejected.reason
        assert "acme/widget" in rejected.reason

    def test_usage_never_exceeds_cap_under_overrequest(self) -> None:
        spec = make_spec(instance_cap=2)
        accepted = sum(1 for _ in range(10) if spec.acquire().accepted)

        assert accepted == 2
        assert spec.used == 2
        assert spec.used <= spec.instance_cap
        assert spec.remaining == 0

    def test_reset_usage_restores_capacity(self) -> None:
        spec = make_spec(instance_cap=1)
        assert spec.acquire().accepted is True
        assert spec.acquire().accepted is False
        spec.reset_usage()
        assert spec.used == 0
        assert spec.acquire().accepted is True


class TestRepoSpecSerialization:
    def test_to_dict_contains_contamination_metadata(self) -> None:
        spec = make_spec()
        data = spec.to_dict()
        for key in ("commit", "commit_date", "language", "license"):
            assert data[key]
        assert data["instance_cap"] == 3
        assert data["used"] == 0
        assert data["remaining"] == 3

    def test_instance_grant_to_dict_shape(self) -> None:
        spec = make_spec(instance_cap=1)
        grant = spec.acquire()
        assert isinstance(grant, InstanceGrant)
        data = grant.to_dict()
        assert data["accepted"] is True
        assert data["instance_index"] == 1
        assert set(data) == {
            "repo_id",
            "accepted",
            "cap",
            "used",
            "remaining",
            "instance_index",
            "reason",
        }

    def test_checkout_commands_pin_the_exact_sha(self) -> None:
        spec = make_spec(default_branch="main")
        joined = "\n".join(spec.checkout_commands())
        # The checkout targets the pinned SHA, never the branch name.
        assert "git checkout -q " + spec.commit in joined
        assert "git fetch -q --depth 1 origin " + spec.commit in joined
        assert "checkout -q main" not in joined


class TestSourceRegistry:
    def test_duplicate_repo_id_rejected(self) -> None:
        with pytest.raises(SourceError):
            SourceRegistry([make_spec(), make_spec()])

    def test_get_unknown_repo_raises(self) -> None:
        registry = SourceRegistry([make_spec()])
        with pytest.raises(UnknownRepoError):
            registry.get("nope/missing")

    def test_by_language_and_languages(self) -> None:
        registry = SourceRegistry(
            [
                make_spec(repo_id="a/py", language="python"),
                make_spec(repo_id="b/js", language="javascript"),
            ]
        )
        assert registry.languages() == ("javascript", "python")
        assert [s.repo_id for s in registry.by_language("python")] == ["a/py"]
        assert registry.has_language("go") is False

    def test_acquire_delegates_to_spec(self) -> None:
        registry = SourceRegistry([make_spec(repo_id="a/py", instance_cap=1)])
        assert registry.acquire("a/py").accepted is True
        assert registry.acquire("a/py").accepted is False

    def test_to_list_serializes_every_entry(self) -> None:
        registry = SourceRegistry(
            [
                make_spec(repo_id="a/py"),
                make_spec(repo_id="b/js", language="javascript"),
            ]
        )
        records = registry.to_list()
        assert {r["repo_id"] for r in records} == {"a/py", "b/js"}


class TestCuratedRegistry:
    def test_covers_every_supported_language(self) -> None:
        registry = build_source_registry()
        for language in SUPPORTED_LANGUAGES:
            assert registry.has_language(language), language

    def test_every_entry_has_populated_metadata_and_pinned_sha(self) -> None:
        registry = build_source_registry()
        assert len(registry) >= 3
        for spec in registry:
            assert spec.repo_id and spec.url
            assert spec.commit and _FULL_SHA.match(spec.commit)
            assert spec.commit_date
            assert spec.language in SUPPORTED_LANGUAGES
            assert spec.license

    def test_fresh_registries_have_independent_usage(self) -> None:
        first = build_source_registry()
        second = build_source_registry()
        repo_id = first.repo_ids()[0]
        first.acquire(repo_id)
        assert first.get(repo_id).used == 1
        # A second registry must start clean (no shared mutable state).
        assert second.get(repo_id).used == 0


class TestSourcesCli:
    def test_sources_list_json_covers_languages(self) -> None:
        result = runner.invoke(forge_app, ["sources-list", "--json"])
        assert result.exit_code == 0
        records = json.loads(result.output)
        languages = {r["language"] for r in records}
        assert {"python", "javascript", "go"} <= languages
        for record in records:
            assert record["commit"] and _FULL_SHA.match(record["commit"])
            assert record["commit_date"] and record["license"]

    def test_sources_list_filter_by_language(self) -> None:
        result = runner.invoke(
            forge_app, ["sources-list", "--language", "go", "--json"]
        )
        assert result.exit_code == 0
        records = json.loads(result.output)
        assert records and all(r["language"] == "go" for r in records)

    def test_sources_list_unknown_language_fails(self) -> None:
        result = runner.invoke(
            forge_app, ["sources-list", "--language", "rust", "--json"]
        )
        assert result.exit_code == 1

    def test_sources_acquire_enforces_cap(self) -> None:
        registry = build_source_registry()
        repo_id = registry.repo_ids()[0]
        cap = registry.get(repo_id).instance_cap

        result = runner.invoke(
            forge_app,
            ["sources-acquire", "--repo", repo_id, "--count", str(cap + 1), "--json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["cap"] == cap
        assert payload["accepted"] == cap
        assert payload["rejected"] == 1
        assert payload["used"] == cap
        assert payload["remaining"] == 0
        # Exactly the (N+1)th request is rejected with a cap reason.
        last = payload["attempts"][-1]
        assert last["accepted"] is False
        assert "per-repo cap reached" in last["reason"]
        # Usage never exceeds the cap across all recorded attempts.
        assert max(a["used"] for a in payload["attempts"]) == cap

    def test_sources_acquire_unknown_repo_fails(self) -> None:
        result = runner.invoke(
            forge_app, ["sources-acquire", "--repo", "nope/x", "--json"]
        )
        assert result.exit_code == 1

    def test_sources_acquire_rejects_bad_count(self) -> None:
        registry = build_source_registry()
        repo_id = registry.repo_ids()[0]
        result = runner.invoke(
            forge_app, ["sources-acquire", "--repo", repo_id, "--count", "0"]
        )
        assert result.exit_code == 1
