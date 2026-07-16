"""Product hardness floors + anti-easy policy (VAL-DHARD-002/003/005).

Covers:
- MIN_F2P_NODES default 3 refuse thin F2P=1 on product/live_generate
- engineering_opt_out skips floors (offline fixtures only)
- multi-file + source hunk floors retained
- solve-all class reason code when panel frontier=1.0
- ship_real_pr ProductHardnessFloorsRejected wiring smoke
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swe_factory.pipeline.hardness_floors import (
    DEFAULT_MIN_F2P_NODES,
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
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


def test_default_min_f2p_is_three() -> None:
    assert DEFAULT_MIN_F2P_NODES == 3
    assert resolve_min_f2p_nodes() == 3
    assert PRODUCT_SOURCE_HUNK_FLOOR == 10
    assert PRODUCT_MULTI_FILE_FLOOR >= 2


def test_resolve_min_f2p_from_env() -> None:
    assert resolve_min_f2p_nodes(env={"MIN_F2P_NODES": "5"}) == 5
    assert resolve_min_f2p_nodes(env={"DEEPAGENT_MIN_F2P_NODES": "4"}) == 4
    assert resolve_min_f2p_nodes(env={"MIN_F2P_NODES": "0"}) == DEFAULT_MIN_F2P_NODES
    assert resolve_min_f2p_nodes(env={"MIN_F2P_NODES": "bogus"}) == DEFAULT_MIN_F2P_NODES
    assert resolve_min_f2p_nodes(override=7, env={"MIN_F2P_NODES": "2"}) == 7


def test_is_hardness_enforced_dest_markers() -> None:
    assert is_hardness_enforced_dest("datasets/deepagent_v1") is True
    assert is_hardness_enforced_dest("datasets/test_n10") is True
    assert is_hardness_enforced_dest("datasets/prod_hard_keep") is True
    assert is_hardness_enforced_dest("tmp/offline_unit", offline_only=True) is False
    assert is_hardness_enforced_dest("datasets/deepagent_v1", engineering_opt_out=True) is False
    assert is_hardness_enforced_dest("tmp/scratch", live_mine=True) is True
    assert is_product_deepagent_dest("datasets/deepagent_v1")
    assert is_live_generate_dest("datasets/test_n10")
    assert is_live_generate_dest("datasets/prod_hard_keep")


def test_check_floors_accepts_at_min_f2p() -> None:
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c"],
        source_files=["pkg/a.py", "pkg/b.py"],
        source_hunk_count=10,
    )
    assert result.ok
    assert result.reason_code == REASON_HARDNESS_OK
    assert result.f2p_count == 3
    assert result.min_f2p_nodes == 3


def test_check_floors_refuses_thin_f2p_one() -> None:
    result = check_product_hardness_floors(
        f2p_node_ids=["only_one"],
        source_files=["pkg/a.py", "pkg/b.py"],
        source_hunk_count=12,
    )
    assert not result.ok
    assert result.reason_code == REASON_F2P_BELOW_FLOOR
    assert REASON_F2P_BELOW_FLOOR in result.reasons
    assert REASON_THIN_F2P_EASY in result.reasons
    assert result.f2p_count == 1


def test_check_floors_refuses_f2p_two_below_default_three() -> None:
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b"],
        source_files=["pkg/a.py", "pkg/b.py"],
        source_hunk_count=12,
    )
    assert not result.ok
    assert result.f2p_count == 2
    assert REASON_F2P_BELOW_FLOOR in result.reasons
    assert REASON_THIN_F2P_EASY not in result.reasons  # thin fingerprint is ≈1


def test_check_floors_refuses_low_hunks_and_single_file() -> None:
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c"],
        source_files=["pkg/only.py"],
        source_hunk_count=9,
    )
    assert not result.ok
    assert REASON_SOURCE_HUNKS_BELOW_FLOOR in result.reasons
    assert REASON_MULTI_FILE_FLOOR in result.reasons


def test_check_floors_solve_all_policy() -> None:
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c"],
        source_files=["pkg/a.py", "pkg/b.py"],
        source_hunk_count=12,
        panel_frontier_pass_at_k=1.0,
    )
    assert not result.ok
    assert result.reason_code == REASON_SOLVE_ALL_EASY
    assert REASON_SOLVE_ALL_EASY in result.reasons


def test_refuse_product_dest_raises_on_thin_f2p() -> None:
    with pytest.raises(ProductHardnessFloorRejected) as ei:
        refuse_product_hardness_floors(
            f2p_node_ids=["thin"],
            source_files=["a.py", "b.py"],
            source_hunk_count=12,
            dest="datasets/deepagent_v1",
        )
    assert ei.value.reason_code == REASON_F2P_BELOW_FLOOR
    assert ei.value.result is not None
    assert ei.value.result.f2p_count == 1


def test_refuse_live_generate_dest_raises() -> None:
    with pytest.raises(ProductHardnessFloorRejected) as ei:
        refuse_product_hardness_floors(
            f2p_node_ids=["x"],
            source_files=["a.py", "b.py"],
            source_hunk_count=12,
            dest="datasets/test_n10",
            task_id="realpr-charset-normalizer-715",
        )
    assert "f2p" in str(ei.value).lower() or "floor" in str(ei.value).lower()
    assert "VAL-DHARD" in str(ei.value)


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
    result = refuse_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d"],
        source_files=["pkg/a.py", "pkg/b.py", "pkg/c.py"],
        source_hunk_count=15,
        dest="datasets/test_n10",
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
        '{"f2p_node_ids": ["t1", "t2", "t3"], "p2p_node_ids": ["p1"]}\n',
        encoding="utf-8",
    )
    (pack / "solution" / "solution.patch").write_text(
        "diff --git a/pkg/a.py b/pkg/a.py\n"
        "--- a/pkg/a.py\n"
        "+++ b/pkg/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/pkg/b.py b/pkg/b.py\n"
        "--- a/pkg/b.py\n"
        "+++ b/pkg/b.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n",
        encoding="utf-8",
    )
    ok = hardness_result_from_pack_dir(pack, source_hunk_count=12)
    assert ok.ok
    assert ok.f2p_count == 3

    thin = tmp_path / "thin"
    (thin / "tests").mkdir(parents=True)
    (thin / "tests" / "config.json").write_text(
        '{"f2p_node_ids": ["only"], "p2p_node_ids": []}\n',
        encoding="utf-8",
    )
    bad = hardness_result_from_pack_dir(thin, source_hunk_count=12)
    assert not bad.ok
    assert bad.reason_code == REASON_F2P_BELOW_FLOOR


def test_anti_easy_policy_summary_documents_codes() -> None:
    summary = anti_easy_policy_summary()
    assert summary["floors"]["min_f2p_nodes"] == 3
    codes = summary["refuse_reason_codes"]
    assert REASON_F2P_BELOW_FLOOR in codes
    assert REASON_SOLVE_ALL_EASY in codes
    assert REASON_THIN_F2P_EASY in codes
    assert "VAL-DHARD-002" in summary["assertions"]
    assert "engineering" in summary["engineering_opt_out"].lower()


def test_live_pack_smoke_thin_f2p_when_present() -> None:
    """If test_n10 still holds known thin F2P packs, hardness check flags them."""
    root = Path("datasets/test_n10/tasks")
    if not root.is_dir():
        pytest.skip("test_n10 not present")
    # realpr-charset-normalizer-715 historically F2P=1 (easy drop for m21c)
    thin = root / "realpr-charset-normalizer-715"
    if not thin.is_dir():
        pytest.skip("known thin sample pack not present")
    result = hardness_result_from_pack_dir(thin, source_hunk_count=12)
    # Whether or not config missing, if f2p_count is 1 → refuse.
    if result.f2p_count == 1:
        assert not result.ok
        assert REASON_F2P_BELOW_FLOOR in result.reasons
