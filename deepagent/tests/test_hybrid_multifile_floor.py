"""Hybrid multi-file DeepSWE-min floor (VAL-DMED-012).

Admit when:
  source_files >= 4
  OR (source_files >= 3 AND gold_added_lines >= 500 AND hunks >= 14)

Thin 2-file qs-class still refuse.
packaging-1120 class (3 files, added=882, hunks=24) must pass structural.
"""

from __future__ import annotations

from swe_factory.pipeline.hardness_floors import (
    PRODUCT_MIN_ADDED_LINES,
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
    REASON_HARDNESS_OK,
    REASON_MULTI_FILE_FLOOR,
    check_product_hardness_floors,
    count_gold_added_lines,
    multi_file_floor_ok,
)
from swe_factory.producers.hard_filter import (
    PRODUCT_MULTI_FILE_FLOOR as HF_MULTI,
)
from swe_factory.producers.hard_filter import (
    PRODUCT_SOURCE_HUNK_FLOOR as HF_HUNKS,
)
from swe_factory.producers.hard_filter import (
    evaluate_product_hard_filter,
)


def _patch_lines(n_files: int, hunks_per_file: int, plus_per_hunk: int) -> str:
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


def test_multi_file_floor_helper_hybrid_rule() -> None:
    # files>=4 always ok (even with small added — other floors catch size)
    assert multi_file_floor_ok(source_files=4, added_lines=10, hunks=14) is True
    assert multi_file_floor_ok(source_files=5, added_lines=0, hunks=0) is True
    # packaging-class hybrid: 3 files + added>=500 + hunks>=14
    assert multi_file_floor_ok(source_files=3, added_lines=500, hunks=14) is True
    assert multi_file_floor_ok(source_files=3, added_lines=882, hunks=24) is True
    # thin hybrid fails
    assert multi_file_floor_ok(source_files=3, added_lines=499, hunks=14) is False
    assert multi_file_floor_ok(source_files=3, added_lines=500, hunks=13) is False
    assert multi_file_floor_ok(source_files=2, added_lines=900, hunks=20) is False
    assert multi_file_floor_ok(source_files=1, added_lines=1000, hunks=30) is False


def test_packaging_1120_class_passes_structural() -> None:
    """3 files, added=882, hunks=24, f2p>=5 → hardness floors ok (multi-file hybrid)."""
    # Build ≈882 plus-lines across 3 files / 24 hunks: 3 * ８ * 37 = 888
    patch = _patch_lines(n_files=3, hunks_per_file=8, plus_per_hunk=37)
    added = count_gold_added_lines(patch)
    assert added >= 500
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        source_files=[
            "src/packaging/tags.py",
            "src/packaging/utils.py",
            "src/packaging/version.py",
        ],
        source_hunk_count=24,
        solution_patch=patch,
    )
    assert result.ok, result.to_dict()
    assert result.reason_code == REASON_HARDNESS_OK
    assert result.source_file_count == 3
    assert (result.added_lines or 0) >= 500
    assert REASON_MULTI_FILE_FLOOR not in result.reasons
    # meta stamps hybrid admit
    assert result.meta.get("multi_file_hybrid_admit") is True or result.meta.get(
        "multi_file_rule"
    ) in {"hybrid_deepswe_min", "files_ge_4_or_hybrid_3"}


def test_qs_487_class_still_refuses() -> None:
    """2 files, ~21 added, f2p=3, hunks~11 → refuse (no hybrid admit)."""
    patch = _patch_lines(n_files=2, hunks_per_file=6, plus_per_hunk=2)
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c"],
        source_files=["lib/parse.js", "lib/stringify.js"],
        source_hunk_count=11,
        solution_patch=patch,
    )
    assert not result.ok
    assert REASON_MULTI_FILE_FLOOR in result.reasons
    assert result.source_file_count == 2


def test_three_file_thin_added_still_refuses() -> None:
    """3 files but added < 500 → multi-file hybrid does not admit."""
    patch = _patch_lines(n_files=3, hunks_per_file=5, plus_per_hunk=5)  # 75 added
    assert count_gold_added_lines(patch) < 500
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=[f"pkg/m{i}.py" for i in range(3)],
        source_hunk_count=15,
        solution_patch=patch,
    )
    assert not result.ok
    assert REASON_MULTI_FILE_FLOOR in result.reasons


def test_four_file_band_still_ok_defaults() -> None:
    assert PRODUCT_MULTI_FILE_FLOOR == 4
    assert PRODUCT_SOURCE_HUNK_FLOOR == 14
    assert PRODUCT_MIN_ADDED_LINES == 400
    patch = _patch_lines(n_files=4, hunks_per_file=4, plus_per_hunk=26)
    result = check_product_hardness_floors(
        f2p_node_ids=["a", "b", "c", "d", "e"],
        source_files=[f"pkg/mod_{i}.py" for i in range(4)],
        source_hunk_count=16,
        solution_patch=patch,
    )
    assert result.ok


def test_hard_filter_hybrid_admits_packaging_class_files() -> None:
    """Mine-time hard filter also uses hybrid multi-file (VAL-DMED-012)."""
    assert HF_MULTI == 4
    assert HF_HUNKS == 14
    # Simulate packaging-class PR with 3 product sources + hunks + additions.
    files = []
    for name in (
        "src/packaging/tags.py",
        "src/packaging/utils.py",
        "src/packaging/version.py",
    ):
        hunk_body = "\n".join(
            f"@@ -{1 + h},1 +{1 + h},10 @@\n context\n-old\n"
            + "\n".join(f"+line{h}_{k}" for k in range(9))
            for h in range(8)
        )
        files.append(
            {
                "filename": name,
                "status": "modified",
                "patch": f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n{hunk_body}\n",
                "additions": 300,
            }
        )
    files.append(
        {
            "filename": "tests/test_tags.py",
            "status": "modified",
            "patch": "@@ -1 +1 @@\n-x\n+y\n",
            "additions": 1,
        }
    )
    result = evaluate_product_hard_filter(
        files=files,
        base_commit="a" * 40,
        merged_at="2024-01-01T00:00:00Z",
        language="python",
        license="Apache-2.0",
        repo="pypa/packaging",
        source_track="real_pr",
    )
    assert result.accepted, result.to_dict() if hasattr(result, "to_dict") else result
    assert result.stats.source_file_count == 3
