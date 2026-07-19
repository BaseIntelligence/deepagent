"""Unit tests: max packs per upstream repo diversity (VAL-DCOV-003)."""

from __future__ import annotations

from swe_factory.pipeline.repo_diversity import (
    DEFAULT_MAX_PACKS_PER_REPO,
    apply_max_packs_per_repo,
    normalize_upstream_repo,
    packs_per_repo_histogram,
    select_diverse_pack_ids,
)


def test_default_max_packs_per_repo_is_two() -> None:
    assert DEFAULT_MAX_PACKS_PER_REPO == 2


def test_normalize_upstream_repo() -> None:
    assert normalize_upstream_repo("pallets/werkzeug") == "pallets/werkzeug"
    assert normalize_upstream_repo("https://github.com/pallets/werkzeug") == "pallets/werkzeug"
    assert normalize_upstream_repo("https://github.com/pallets/werkzeug.git") == "pallets/werkzeug"
    assert normalize_upstream_repo("github.com/Pallets/Werkzeug") == "pallets/werkzeug"
    assert normalize_upstream_repo("") == ""
    assert normalize_upstream_repo("not-a-repo") == ""


def test_apply_max_packs_per_repo_caps_at_two() -> None:
    items = [
        {"pack_id": "realpr-werkzeug-2608", "repo": "pallets/werkzeug", "score": 30},
        {"pack_id": "realpr-werkzeug-2637", "repo": "pallets/werkzeug", "score": 40},
        {"pack_id": "realpr-werkzeug-3116", "repo": "pallets/werkzeug", "score": 50},
        {"pack_id": "realpr-packaging-1120", "repo": "pypa/packaging", "score": 20},
        {"pack_id": "realpr-itemadapter-101", "repo": "scrapy/itemadapter", "score": 25},
    ]
    kept, dropped = apply_max_packs_per_repo(items, max_packs=2, score_key="score")
    kept_ids = {k["pack_id"] for k in kept}
    drop_ids = {d["pack_id"] for d in dropped}
    # Prefer higher score: keep 3116 + 2637; drop 2608
    assert "realpr-werkzeug-3116" in kept_ids
    assert "realpr-werkzeug-2637" in kept_ids
    assert "realpr-werkzeug-2608" in drop_ids
    assert "realpr-packaging-1120" in kept_ids
    assert "realpr-itemadapter-101" in kept_ids
    assert len(kept) == 4
    assert len(dropped) == 1


def test_select_diverse_pack_ids_stable_order() -> None:
    pairs = [
        ("realpr-werkzeug-a", "pallets/werkzeug"),
        ("realpr-werkzeug-b", "pallets/werkzeug"),
        ("realpr-werkzeug-c", "pallets/werkzeug"),
        ("realpr-click-1", "pallets/click"),
        ("realpr-click-2", "pallets/click"),
        ("realpr-click-3", "pallets/click"),
        ("realpr-flask-1", "pallets/flask"),
    ]
    selected = select_diverse_pack_ids(pairs, max_packs=2)
    assert selected == [
        "realpr-werkzeug-a",
        "realpr-werkzeug-b",
        "realpr-click-1",
        "realpr-click-2",
        "realpr-flask-1",
    ]
    # werkzeug-c and click-3 capped out
    assert "realpr-werkzeug-c" not in selected
    assert "realpr-click-3" not in selected


def test_apply_with_url_field() -> None:
    items = [
        {
            "pack_id": "p1",
            "repository_url": "https://github.com/encode/httpx",
        },
        {
            "pack_id": "p2",
            "repository_url": "https://github.com/encode/httpx.git",
        },
        {
            "pack_id": "p3",
            "repository_url": "https://github.com/encode/httpx",
        },
        {
            "pack_id": "p4",
            "repository_url": "https://github.com/psf/requests",
        },
    ]
    kept, dropped = apply_max_packs_per_repo(items, max_packs=2)
    assert len(kept) == 3
    assert len(dropped) == 1
    assert {k["pack_id"] for k in kept} == {"p1", "p2", "p4"}
    assert dropped[0]["pack_id"] == "p3"


def test_histogram() -> None:
    items = [
        {"repo": "a/b"},
        {"repo": "a/b"},
        {"repo": "c/d"},
    ]
    hist = packs_per_repo_histogram(items)
    assert hist == {"a/b": 2, "c/d": 1}


def test_empty_input() -> None:
    kept, dropped = apply_max_packs_per_repo([], max_packs=2)
    assert kept == []
    assert dropped == []
    assert select_diverse_pack_ids([], max_packs=2) == []
