"""M8 allowlist expansion: multi-lang permissive inventory + under-supply honesty."""

from __future__ import annotations

import re

from swe_factory.sources.allowlist import (
    ALLOWLIST,
    REMOTE_SEEDS,
    SCALE_LANGUAGES,
    TINY_GREEN,
    allowlist_summary,
    get_seed,
    language_histogram,
    local_offline_seeds,
    normalize_language,
    remote_mine_seeds,
    scale_inventory_report,
    seeds_for_language,
    under_supply_reasons,
)
from swe_factory.sources.license_gate import is_permissive

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_remote_allowlist_covers_scale_languages() -> None:
    hist = language_histogram(REMOTE_SEEDS)
    for lang in SCALE_LANGUAGES:
        assert lang in hist
        # Py/TS/Go/JS must be seeded; Rust best-effort but inventory >0 after M8 expand
        assert hist[lang] >= 1, f"expected inventory for {lang}, got {hist}"
    assert hist["python"] >= 4
    assert hist["go"] >= 4
    assert hist["javascript"] >= 4
    assert hist["typescript"] >= 2
    assert hist["rust"] >= 2


def test_remote_seeds_are_permissive_with_full_sha() -> None:
    assert len(REMOTE_SEEDS) >= 20
    for seed in REMOTE_SEEDS:
        assert is_permissive(seed.license), seed.seed_id
        assert _FULL_SHA.fullmatch(seed.base_commit), seed.seed_id
        assert "/" in seed.repo
        assert seed.repository_url.startswith("https://github.com/")
        assert seed.modular is True


def test_remote_mine_seeds_ordered_and_filterable() -> None:
    all_remote = remote_mine_seeds()
    assert [s.seed_id for s in all_remote] == [
        s.seed_id for s in sorted(REMOTE_SEEDS, key=lambda x: (x.mine_priority, x.seed_id))
    ]
    py = remote_mine_seeds(language="python")
    assert py
    assert all(s.language == "python" for s in py)
    assert remote_mine_seeds(language="py")
    assert remote_mine_seeds(language="go")
    assert remote_mine_seeds(language="golang")


def test_normalize_language_aliases() -> None:
    assert normalize_language("py") == "python"
    assert normalize_language("js") == "javascript"
    assert normalize_language("ts") == "typescript"
    assert normalize_language("golang") == "go"
    assert normalize_language("rs") == "rust"


def test_scale_inventory_report_honest_zeros() -> None:
    report = scale_inventory_report()
    assert report["remote_seed_count"] == len(REMOTE_SEEDS)
    assert report["oxylabs_required_for_discover"] is False
    assert report["history_authority"] == "git"
    hist = report["remote_language_histogram"]
    assert isinstance(hist, dict)
    assert set(SCALE_LANGUAGES).issubset(set(hist))
    # empty histogram under_supply helper
    reasons = under_supply_reasons(
        {"python": 0, "go": 0, "typescript": 0, "javascript": 0, "rust": 0}
    )
    assert any("python" in r for r in reasons)
    assert any("rust" in r for r in reasons)


def test_get_seed_and_offline_still_present() -> None:
    assert get_seed(TINY_GREEN.seed_id).seed_id == TINY_GREEN.seed_id
    assert get_seed("python_boltons").language == "python"
    assert get_seed("rust_log").language == "rust"
    local = local_offline_seeds()
    assert any(s.seed_id == TINY_GREEN.seed_id for s in local)
    rows = allowlist_summary()
    assert any(r["remote"] is True for r in rows)
    assert any(r["language"] == "rust" for r in rows)
    # seeds_for_language still works for single-lang
    assert seeds_for_language("go")
    assert all(s.modular for s in ALLOWLIST)


def test_no_copyleft_on_remote_inventory() -> None:
    for seed in REMOTE_SEEDS:
        assert "gpl" not in seed.license.lower()
        assert "agpl" not in seed.license.lower()
