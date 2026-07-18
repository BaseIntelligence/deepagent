"""Product hardness floors + anti-easy policy (M27 DeepSWE-median / VAL-DMED-001).

Covers:
- MIN_F2P_NODES default **5** refuse thin F2P on product/live_generate
- PRODUCT floors files≥4, hunks≥14, gold added_lines≥400
- engineering_opt_out skips floors (offline fixtures only)
- multi-file + source hunk + added-line floors retained
- solve-all class reason code when panel frontier=1.0
- ship_real_pr ProductHardnessFloorsRejected wiring smoke
- qs-class thin gold refuse; DeepSWE-median gold accept
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swe_factory.pipeline.hardness_floors import (
    DEFAULT_MIN_F2P_NODES,
    PRODUCT_MIN_ADDED_LINES,
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
    REASON_ADDED_LINES_BELOW_FLOOR,
    REASON_F2P_BELOW_FLOOR,
    REASON_HARDNESS_OK,
    REASON_HARDNESS_SKIPPED,
    REASON_MULTI_FILE_FLOOR,
    REASON_SOLVE_ALL_EASY,
    REASON_SOURCE_HUNKS_BELOW_FLOOR,
    REASON_THIN_F2P_EASY,
    ProductHardnessFloorRejected,
    anti_easy_policy_summary,
    check_product_hardness_floors,
    count_gold_added_lines,
    hardness_result_from_pack_dir,
    is_hardness_enforced_dest,
    refuse_product_hardness_floors,
    resolve_min_f2p_nodes,
)
from swe_factory.pipeline.ship_real_pr import (
    ProductHardnessFloorsRejected,
    is_live_generate_dest,
    is_product_deepagent_dest,
)


def _patch_lines(n_files: int, hunks_per_file: int, plus_per_hunk: int) -> str:
    """Synthetic gold solution.patch with controlled files/hunks/added."""
    parts: list[str] = []
    for fi in range(n_files):
        path = f"pkg/mod_{fi}.py"
        hunks = []
        for h in range(hunks_per_file):
            plus = "\n".join(f"+added_{fi}_{h}_{k}" for k in range(plus_per_hunk))
            hunks.append(
                f"@@ -{1 + h * 5},1 +{1 + h * 5},{plus_per_hunk + 1} @@\n"
                f" context\n"
                f"-old_{fi}_{h}\n"
                f"{plus}\n"
            )
        body = "".join(hunks)
        parts.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{body}")
    return "".join(parts)


def test_default_min_f2p_is_five_m27() -> None:
    """VAL-DMED-001: product defaults match DeepSWE-median band."""
    assert DEFAULT_MIN_F2P_NODES == 5
    assert resolve_min_f2p_nodes() == 5
    assert PRODUCT_SOURCE_HUNK_FLOOR == 14
    assert PRODUCT_MULTI_FILE_FLOOR == 4
    assert PRODUCT_MIN_ADDED_LINES == 400


def test_resolve_min_f2p_from_env() -> None:
    assert resolve_min_f2p_nodes(env={"MIN_F2P_NODES": "7"}) == 7
    assert resolve_min_f2p_nodes(env={"DEEPAGENT_MIN_F2P_NODES": "6"}) == 6
    assert resolve_min_f2p_nodes(env={"MIN_F2P_NODES": "0"}) == DEFAULT_MIN_F2P_NODES
    assert resolve_min_f2p_nodes(env={"MIN_F2P_NODES": "bogus"}) == DEFAULT_MIN_F2P_NODES
    assert resolve_min_f2p_nodes(override=9, env={"MIN_F2P_NODES": "2"}) == 9


def test_is_hardness_enforced_dest_markers() -> None:
    assert is_hardness_enforced_dest("datasets/deepagent_v1") is True
    assert is_hardness_enforced_dest("datasets/test_n10") is True
    assert is_hardness_enforced_dest("datasets/prod_hard_keep") is True
    assert is_hardness_enforced_dest("datasets/prod_hard_deepswe_med") is True
    assert is_hardness_enforced_dest("tmp/offline_unit", offline_only=True) is False
    assert is_hardness_enforced_dest("datasets/deepagent_v1", engineering_opt_out=True) is False
    assert is_hardness_enforced_dest("tmp/scratch", live_mine=True) is True
    assert is_product_deepagent_dest("datasets/deepagent_v1")
    assert is_live_generate_dest("datasets/test_n10")
    assert is_live_generate_dest("datasets/prod_hard_keep")
    assert is_live_generate_dest("datasets/prod_hard_deepswe_med")


def test_count_gold_added_lines_ignores_headers() -> None:
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n-old\n+new1\n+new2\n"
    assert count_gold_added_lines(patch) == 2
    assert count_gold_added_lines("") == 0
    assert count_gold_added_lines(None) == 0


def test_check_floors_accepts_at_m27_median_band() -> None:
    """At-floor DeepSWE-median gold: files=4, hunks=14, f2p=5, added≥400."""
    # 4 files × 4 hunks = 16 hunks; 4×4×26 plus ≈ 416 added
    patch = _patch_lines(n_files=4, hunks_per_file=4, plus_per_hunk=26)
    added = count_gold_added_lines(patch)
    assert added >= PRODUCT_MIN_ADDED_LINES
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=16,
        solution_patch=patch,
    )
    assert result.ok
    assert result.reason_code == REASON_HARDNESS_OK
    assert result.f2p_count == 5
    assert result.min_f2p_nodes == 5
    assert result.min_source_hunks == 14
    assert result.min_source_files == 4
    assert result.min_added_lines == 400
    assert (result.added_lines or 0) >= 400


def test_check_floors_refuses_thin_f2p_one() -> None:
    patch = _patch_lines(4, 4, 26)
    result = check_product_hardness_floors(
        f2p_node_ids=["only_one"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=16,
        solution_patch=patch,
    )
    assert not result.ok
    assert result.reason_code == REASON_F2P_BELOW_FLOOR
    assert REASON_F2P_BELOW_FLOOR in result.reasons
    assert REASON_THIN_F2P_EASY in result.reasons
    assert result.f2p_count == 1


def test_check_floors_refuses_f2p_three_below_default_five() -> None:
    """qs-class F2P=3 fails new default floor 5 (VAL-DMED-001)."""
    patch = _patch_lines(4, 4, 26)
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=16,
        solution_patch=patch,
    )
    assert not result.ok
    assert result.f2p_count == 3
    assert REASON_F2P_BELOW_FLOOR in result.reasons


def test_check_floors_refuses_qs_class_thin_gold() -> None:
    """qs-487 class: 2 files, ~21 added, f2p=3, hunks~11 → all floors refuse."""
    # 2 files × 6 hunks = 12; 2 plus each → 24 added
    patch = _patch_lines(n_files=2, hunks_per_file=6, plus_per_hunk=2)
    assert count_gold_added_lines(patch) < PRODUCT_MIN_ADDED_LINES
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c"],
        source_files=["lib/parse.js", "lib/stringify.js"],
        source_hunk_count=11,
        solution_patch=patch,
    )
    assert not result.ok
    assert REASON_MULTI_FILE_FLOOR in result.reasons or result.source_file_count < 4
    assert REASON_SOURCE_HUNKS_BELOW_FLOOR in result.reasons
    assert REASON_F2P_BELOW_FLOOR in result.reasons
    assert REASON_ADDED_LINES_BELOW_FLOOR in result.reasons


def test_check_floors_refuses_low_hunks_and_single_file() -> None:
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=["pkg/only.py"],
        source_hunk_count=9,
        added_lines=500,
    )
    assert not result.ok
    assert REASON_SOURCE_HUNKS_BELOW_FLOOR in result.reasons
    assert REASON_MULTI_FILE_FLOOR in result.reasons


def test_check_floors_refuses_added_below_400() -> None:
    # Enough files/hunks/f2p but gold too small.
    patch = _patch_lines(n_files=4, hunks_per_file=4, plus_per_hunk=2)
    assert count_gold_added_lines(patch) < 400
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=16,
        solution_patch=patch,
    )
    assert not result.ok
    assert REASON_ADDED_LINES_BELOW_FLOOR in result.reasons


def test_check_floors_boundary_hunks_13_reject_14_ok() -> None:
    patch_ok = _patch_lines(4, 4, 26)
    bad = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=13,
        solution_patch=patch_ok,
    )
    assert not bad.ok
    assert REASON_SOURCE_HUNKS_BELOW_FLOOR in bad.reasons

    good = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=14,
        solution_patch=patch_ok,
    )
    assert good.ok


def test_check_floors_solve_all_policy() -> None:
    patch = _patch_lines(4, 4, 26)
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=16,
        solution_patch=patch,
        panel_frontier_pass_at_k=1.0,
    )
    assert not result.ok
    assert result.reason_code == REASON_SOLVE_ALL_EASY
    assert REASON_SOLVE_ALL_EASY in result.reasons


def test_refuse_product_dest_raises_on_thin_f2p() -> None:
    with pytest.raises(ProductHardnessFloorRejected) as ei:
        refuse_product_hardness_floors(
            f2p_node_ids=["thin"],
            source_files=[f"pkg/mod_{i}.py" for i in range(4)],
            source_hunk_count=16,
            added_lines=500,
            dest="datasets/deepagent_v1",
        )
    assert ei.value.reason_code == REASON_F2P_BELOW_FLOOR
    assert ei.value.result is not None
    assert ei.value.result.f2p_count == 1


def test_refuse_live_generate_dest_raises() -> None:
    with pytest.raises(ProductHardnessFloorRejected) as ei:
        refuse_product_hardness_floors(
            f2p_node_ids=["x"],
            source_files=[f"pkg/mod_{i}.py" for i in range(4)],
            source_hunk_count=16,
            added_lines=500,
            dest="datasets/test_n10",
            task_id="realpr-charset-normalizer-715",
        )
    assert "f2p" in str(ei.value).lower() or "floor" in str(ei.value).lower()
    assert "VAL-DHARD" in str(ei.value) or "VAL-DMED" in str(ei.value)


def test_refuse_deepswe_med_dest_raises_on_added() -> None:
    with pytest.raises(ProductHardnessFloorRejected) as ei:
        refuse_product_hardness_floors(
            f2p_node_ids=["a", "b", "c", "d", "e"],
            source_files=[f"pkg/mod_{i}.py" for i in range(4)],
            source_hunk_count=16,
            added_lines=50,
            dest="datasets/prod_hard_deepswe_med",
            task_id="thin-gold",
        )
    assert ei.value.reason_code == REASON_ADDED_LINES_BELOW_FLOOR


def test_engineering_opt_out_skips_refuse() -> None:
    result = refuse_product_hardness_floors(
        f2p_node_ids=["thin"],
        source_files=["a.py"],
        source_hunk_count=1,
        dest="datasets/deepagent_v1",
        engineering_opt_out=True,
    )
    assert result.ok
    assert result.reason_code == REASON_HARDNESS_SKIPPED


def test_offline_only_skips_refuse() -> None:
    result = refuse_product_hardness_floors(
        f2p_node_ids=["thin"],
        source_files=["a.py", "b.py"],
        source_hunk_count=12,
        dest="tmp/offline_sandbox",
        offline_only=True,
    )
    assert result.ok
    assert result.reason_code == REASON_HARDNESS_SKIPPED


def test_force_enforces_even_offline() -> None:
    with pytest.raises(ProductHardnessFloorRejected):
        refuse_product_hardness_floors(
            f2p_node_ids=["thin"],
            source_files=["a.py", "b.py"],
            source_hunk_count=12,
            dest="tmp/offline_sandbox",
            offline_only=True,
            force=True,
        )


def test_refuse_passes_when_floors_met() -> None:
    patch = _patch_lines(4, 4, 26)
    result = refuse_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=16,
        solution_patch=patch,
        dest="datasets/prod_hard_deepswe_med",
    )
    assert result.ok
    assert result.reason_code == REASON_HARDNESS_OK


def test_ship_exception_wraps_hardness_floor() -> None:
    """ship_real_pr typing: ProductHardnessFloorsRejected is a distinct refuse class."""
    assert issubclass(ProductHardnessFloorsRejected, Exception)
    exc = ProductHardnessFloorsRejected("thin f2p", reason_code=REASON_F2P_BELOW_FLOOR, result=None)
    assert exc.reason_code == REASON_F2P_BELOW_FLOOR


def test_hardness_result_from_pack_dir(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    (pack / "tests").mkdir(parents=True)
    (pack / "solution").mkdir(parents=True)
    (pack / "tests" / "config.json").write_text(
        '{"f2p_node_ids": ["t1", "t2", "t3", "t4", "t5"], "p2p_node_ids": ["p1"]}\n',
        encoding="utf-8",
    )
    patch = _patch_lines(4, 4, 26)
    (pack / "solution" / "solution.patch").write_text(patch, encoding="utf-8")
    ok = hardness_result_from_pack_dir(pack, source_hunk_count=16)
    assert ok.ok
    assert ok.f2p_count == 5
    assert (ok.added_lines or 0) >= 400

    thin = tmp_path / "thin"
    (thin / "tests").mkdir(parents=True)
    (thin / "solution").mkdir(parents=True)
    (thin / "tests" / "config.json").write_text(
        '{"f2p_node_ids": ["only"], "p2p_node_ids": []}\n',
        encoding="utf-8",
    )
    (thin / "solution" / "solution.patch").write_text(
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
        encoding="utf-8",
    )
    bad = hardness_result_from_pack_dir(thin, source_hunk_count=12)
    assert not bad.ok
    assert bad.reason_code in {
        REASON_F2P_BELOW_FLOOR,
        REASON_MULTI_FILE_FLOOR,
        REASON_ADDED_LINES_BELOW_FLOOR,
        REASON_SOURCE_HUNKS_BELOW_FLOOR,
    }


def test_anti_easy_policy_summary_documents_m27_codes() -> None:
    summary = anti_easy_policy_summary()
    assert summary["floors"]["min_f2p_nodes"] == 5
    assert summary["floors"]["source_hunk_floor"] == 14
    assert summary["floors"]["multi_file_floor"] == 4
    assert summary["floors"]["min_added_lines"] == 400
    codes = summary["refuse_reason_codes"]
    assert REASON_F2P_BELOW_FLOOR in codes
    assert REASON_SOLVE_ALL_EASY in codes
    assert REASON_THIN_F2P_EASY in codes
    assert REASON_ADDED_LINES_BELOW_FLOOR in codes
    assert "VAL-DMED-001" in summary["assertions"] or "VAL-DHARD-002" in summary["assertions"]
    assert "engineering" in summary["engineering_opt_out"].lower()


def test_live_pack_smoke_qs487_when_present() -> None:
    """If prod_hard_keep still holds qs-487, M27 floors refuse it."""
    thin = Path("datasets/prod_hard_keep/tasks/realpr-qs-487")
    if not thin.is_dir():
        pytest.skip("qs-487 sample pack not present")
    result = hardness_result_from_pack_dir(thin)
    assert not result.ok
    # At least one of the median floors must bite.
    assert result.reasons
    assert any(
        r
        in {
            REASON_F2P_BELOW_FLOOR,
            REASON_MULTI_FILE_FLOOR,
            REASON_SOURCE_HUNKS_BELOW_FLOOR,
            REASON_ADDED_LINES_BELOW_FLOOR,
        }
        for r in result.reasons
    )
