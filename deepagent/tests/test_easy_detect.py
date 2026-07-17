"""M24 auto too-easy detector (VAL-DEASY-001/002/005).

Synthetic scoreboard cases:
- dual-model solve-all → EASY_SOLVE_ALL / should_drop
- one-sided discrimination → keep
- thin F2P structural → drop via hardness_floors reason reuse
- scoreboard-driven force_drop (no hardcoded pack names)
"""

from __future__ import annotations

import json
from pathlib import Path

from swe_factory.pipeline.curate_prod_hard import (
    force_drop_from_scoreboard,
    merge_force_drops,
)
from swe_factory.pipeline.easy_detect import (
    EASY_SOLVE_ALL,
    REASON_NOT_EASY,
    REASON_ONE_SIDED_DISCRIM,
    classify_pack_from_panel_row,
    classify_scoreboard,
    force_drop_from_easy_report,
)
from swe_factory.pipeline.hardness_floors import (
    REASON_F2P_BELOW_FLOOR,
    REASON_SOLVE_ALL_EASY,
    REASON_THIN_F2P_EASY,
)


def _scoreboard_dual_solve_all() -> dict:
    """Mirror panel_prod_hard_bench10_n5 scoreboard shape (synthetic)."""
    return {
        "models": ["x-ai/grok-4.5", "moonshotai/kimi-k2.6"],
        "k": 1,
        "per_pack": [
            {
                "pack_id": "realpr-werkzeug-fake-1",
                "complete": True,
                "decision": "drop",
                "frontier": 1.0,
                "grok-4.5": 1.0,
                "kimi-k2.6": 1.0,
            },
            {
                "pack_id": "realpr-discrim-oneside",
                "complete": True,
                "decision": "keep",
                "frontier": 0.5,
                "grok-4.5": 1.0,
                "kimi-k2.6": 0.0,
            },
            {
                "pack_id": "realpr-solve-none-hard",
                "complete": True,
                "decision": "drop",
                "frontier": 0.0,
                "grok-4.5": 0.0,
                "kimi-k2.6": 0.0,
            },
            {
                "pack_id": "realpr-werkzeug-fake-2",
                "complete": True,
                "decision": "drop",
                "frontier": 1.0,
                "grok-4.5": 1.0,
                "kimi-k2.6": 1.0,
            },
        ],
    }


def test_dual_solve_all_flags_easy_without_name_hardcode() -> None:
    row = {
        "pack_id": "any-solve-all-id",
        "frontier": 1.0,
        "grok-4.5": 1.0,
        "kimi-k2.6": 1.0,
    }
    r = classify_pack_from_panel_row(
        row,
        models=["x-ai/grok-4.5", "moonshotai/kimi-k2.6"],
    )
    assert r.should_drop_hardness is True
    assert r.reason_code == REASON_SOLVE_ALL_EASY
    assert r.label == EASY_SOLVE_ALL
    assert r.all_models_solved is True
    # Classification does not require name-based match (any-solve-all-id works).
    assert r.pack_id == "any-solve-all-id"


def test_one_sided_discrimination_stays() -> None:
    row = {
        "pack_id": "realpr-itemadapter-101",
        "frontier": 0.5,
        "grok-4.5": 1.0,
        "kimi-k2.6": 0.0,
    }
    r = classify_pack_from_panel_row(
        row,
        models=["x-ai/grok-4.5", "moonshotai/kimi-k2.6"],
    )
    assert r.should_drop_hardness is False
    assert r.reason_code == REASON_ONE_SIDED_DISCRIM
    assert r.label is None


def test_flip_side_discrimination_stays() -> None:
    row = {
        "pack_id": "realpr-attrs-1323",
        "frontier": 0.5,
        "grok-4.5": 0.0,
        "kimi-k2.6": 1.0,
    }
    r = classify_pack_from_panel_row(row)
    assert r.should_drop_hardness is False
    assert r.reason_code == REASON_ONE_SIDED_DISCRIM


def test_solve_none_not_easy_drop() -> None:
    """Solve-none is *hard*, not easy — detector must not drop as EASY_SOLVE_ALL."""
    row = {
        "pack_id": "realpr-httpx-3672",
        "frontier": 0.0,
        "grok-4.5": 0.0,
        "kimi-k2.6": 0.0,
    }
    r = classify_pack_from_panel_row(row)
    assert r.should_drop_hardness is False
    assert r.reason_code == REASON_NOT_EASY
    assert r.label != EASY_SOLVE_ALL


def test_thin_f2p_reuses_hardness_floor_reason() -> None:
    row = {
        "pack_id": "thin-pack",
        "frontier": 0.5,
        "grok-4.5": 1.0,
        "kimi-k2.6": 0.0,
    }
    r = classify_pack_from_panel_row(
        row,
        f2p_node_ids=["only_one"],
        min_f2p_nodes=3,
    )
    assert r.should_drop_hardness is True
    assert r.reason_code in {REASON_THIN_F2P_EASY, REASON_F2P_BELOW_FLOOR}
    assert r.f2p_count == 1
    assert r.label == "THIN_F2P_EASY"


def test_classify_scoreboard_batch(tmp_path: Path) -> None:
    sb = _scoreboard_dual_solve_all()
    path = tmp_path / "scoreboard.json"
    path.write_text(json.dumps(sb), encoding="utf-8")
    report = classify_scoreboard(path)
    assert report.ok
    by = report.by_pack()
    assert by["realpr-werkzeug-fake-1"].should_drop_hardness is True
    assert by["realpr-werkzeug-fake-1"].reason_code == REASON_SOLVE_ALL_EASY
    assert by["realpr-werkzeug-fake-2"].should_drop_hardness is True
    assert by["realpr-discrim-oneside"].should_drop_hardness is False
    assert by["realpr-solve-none-hard"].should_drop_hardness is False
    assert "realpr-werkzeug-fake-1" in report.drop_ids
    assert "realpr-discrim-oneside" in report.keep_ids
    # force_drop table is scoreboard-derived, no hardcoded keys required
    drops = force_drop_from_easy_report(report)
    assert set(drops) == {"realpr-werkzeug-fake-1", "realpr-werkzeug-fake-2"}
    assert drops["realpr-werkzeug-fake-1"]["reason_code"] == REASON_SOLVE_ALL_EASY


def test_classify_from_report_pack_results_shape() -> None:
    report_doc = {
        "models": ["x-ai/grok-4.5", "moonshotai/kimi-k2.6"],
        "pack_results": [
            {
                "pack_id": "dual-all",
                "decision": {
                    "verdict": "drop",
                    "rule": "solve-all",
                    "frontier_pass_at_k": 1.0,
                    "per_model_pass_at_k": {
                        "x-ai/grok-4.5": 1.0,
                        "moonshotai/kimi-k2.6": 1.0,
                    },
                },
            },
            {
                "pack_id": "oneside",
                "decision": {
                    "verdict": "keep",
                    "rule": "in-band-high-discrimination",
                    "frontier_pass_at_k": 0.5,
                    "per_model_pass_at_k": {
                        "x-ai/grok-4.5": 1.0,
                        "moonshotai/kimi-k2.6": 0.0,
                    },
                },
            },
        ],
    }
    report = classify_scoreboard(report_doc)
    by = report.by_pack()
    assert by["dual-all"].should_drop_hardness
    assert not by["oneside"].should_drop_hardness


def test_force_drop_from_scoreboard_api(tmp_path: Path) -> None:
    path = tmp_path / "sb.json"
    path.write_text(json.dumps(_scoreboard_dual_solve_all()), encoding="utf-8")
    drops, easy = force_drop_from_scoreboard(path)
    assert "realpr-werkzeug-fake-1" in drops
    assert "realpr-discrim-oneside" not in drops
    assert easy.ok
    # merge with empty explicit table must not require name hardcodes
    merged = merge_force_drops(None, drops)
    assert set(merged) == {
        "realpr-werkzeug-fake-1",
        "realpr-werkzeug-fake-2",
    }


def test_real_prod_hard_scoreboard_flags_three_werkzeug_class() -> None:
    """Live M23 scoreboard artifact: auto-flag 3 dual solve-alls without name filter."""
    sb_path = Path("datasets/panel_prod_hard_bench10_n5/scoreboard.json")
    if not sb_path.is_file():
        # Offline-only environments without the panel artifact
        return
    report = classify_scoreboard(sb_path)
    solve_alls = [
        r.pack_id
        for r in report.results
        if r.should_drop_hardness and r.reason_code == REASON_SOLVE_ALL_EASY
    ]
    # Scoreboard-driven detection (not name hardcode in detector)
    assert len(solve_alls) >= 3
    # Known M23 cells happen to be werkzeug — assert by matrix, not exclusive namesett
    for r in report.results:
        rates = r.per_model_pass_at_k
        if rates and all(v >= 1.0 for v in rates.values()):
            assert r.should_drop_hardness
            assert r.pack_id in solve_alls
    # One-sided stays
    by = report.by_pack()
    if "realpr-itemadapter-101" in by:
        assert by["realpr-itemadapter-101"].should_drop_hardness is False
    if "realpr-attrs-1323" in by:
        assert by["realpr-attrs-1323"].should_drop_hardness is False
