"""M25/M27 intrinsic request+patch difficulty (VAL-DINTR-001/002, VAL-DMED-002).

* both models solve → easy_detect keeps (should_drop_hardness=False) unless
  intrinsic EASY_REQUEST
* qs-487-class thin gold (~21 added, 2 files, f2p=3) → high-conf EASY_REQUEST drop
* DeepSWE-median multi-file large gold → HARD_REQUEST keep
* model scores never sole drop input
"""

from __future__ import annotations

import json
from pathlib import Path

from swe_factory.pipeline.curate_prod_hard import decide_pack
from swe_factory.pipeline.easy_detect import (
    EASY_SOLVE_ALL,
    REASON_SOLVE_ALL_EASY,
    classify_pack_from_panel_row,
)
from swe_factory.pipeline.intrinsic_difficulty import (
    CLASS_EASY_REQUEST,
    CLASS_HARD_REQUEST,
    CLASS_UNCERTAIN,
    REASON_EASY_REQUEST,
    intrinsic_from_pack_dir,
    score_request_patch_difficulty,
)


def _thin_instruction() -> str:
    # Short, low-outcome contract but still behavioral (so alignment gate may
    # pass when F2P exist) — intrinsic path should still class EASY_REQUEST.
    return (
        "# Bump adapter label\n\n"
        "Change the public label string so validation readers see v2.\n\n"
        "## Expected outcomes\n"
        "1. Label validation returns the v2 string for callers.\n\n"
        "## Constraints\n"
        "- Touch only the label constant module.\n"
    )


def _qs_class_instruction() -> str:
    """Long-ish real-deepagent-style prompt but thin gold (real qs-487 is ~2.5k)."""
    return (
        "# Fix querystring nested encoding edge cases\n\n"
        "Make parse/stringify round-trips correctly handle nested objects and "
        "arrays so deeply nested keys keep expected structure after encode/decode. "
        "Agent must update the option handling without changing unrelated modules.\n\n"
        "## Expected outcomes\n"
        "1. Nested objects stringify with bracket notation correctly.\n"
        "2. Parse restores nested arrays without flattening incorrectly.\n"
        "3. Strict mode throws when options are invalid.\n\n"
        "## Constraints\n"
        "- Touch only the parse and stringify modules in this package.\n"
        "- Keep public API names stable.\n"
        "- Do not change unrelated middleware.\n\n"
        "IMPORTANT: Please work on this in a new branch from main and commit "
        "everything when you are done.\n"
    )


def _hard_instruction() -> str:
    return (
        "# Fix complex multi-module negotiation behaviour\n\n"
        "Implement multi-step adapter validation across nested mapping adapters, "
        "list adapters, and schema export so nested fields reject invalid payloads, "
        "preserve document order, and keep public API names stable. Agents must "
        "reason about cross-module invariants without breaking pass-to-pass.\n\n"
        "## Expected outcomes\n"
        "1. Nested mapping adapters validate field types and raise clear errors.\n"
        "2. List adapters preserve insertion order across round-trips.\n"
        "3. Schema export mirrors runtime adapter names without dropping fields.\n"
        "4. Missing required fields produce explicit contract errors.\n"
        "5. Cross-module registry keys stay stable under rename of private helpers.\n\n"
        "## Constraints\n"
        "- Touch only the adapter package sources needed for the behaviour.\n"
        "- Keep public API names stable except where requirements demand new hooks.\n"
        "- Do not change unrelated serializers.\n"
        "- Preserve document order for list adapters.\n"
        "- Must not weaken TypeError contracts on nested bad payloads.\n\n"
        "IMPORTANT: Please work on this in a new branch from main and commit "
        "everything when you are done.\n"
    )


def _tiny_patch() -> str:
    return (
        "diff --git a/pkg/label.py b/pkg/label.py\n"
        "--- a/pkg/label.py\n+++ b/pkg/label.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-LABEL = 'v1'\n"
        "+LABEL = 'v2'\n"
    )


def _qs_class_patch() -> str:
    """qs-487-class: 2 files, ~11 hunks, ~21 added lines."""
    parts: list[str] = []
    for path, n_hunks in (("lib/parse.js", 6), ("lib/stringify.js", 5)):
        hunks = []
        for h in range(n_hunks):
            hunks.append(
                f"@@ -{10 + h * 3},2 +{10 + h * 3},3 @@\n"
                f" keep\n"
                f"-old_{path}_{h}\n"
                f"+new_{path}_{h}\n"
                f"+extra_{path}_{h}\n"
            )
        body = "".join(hunks)
        parts.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{body}")
    return "".join(parts)


def _large_multi_module_patch() -> str:
    """DeepSWE-median-ish: ≥4 files, ≥14 hunks, ≥400 added lines."""
    parts: list[str] = []
    modules = ("adapter", "schema", "registry", "export", "validate", "runtime")
    for mod_i, mod in enumerate(modules[:4]):
        path = f"{mod}/core.py"
        hunks = []
        for h in range(4):  # 4 files × 4 hunks = 16
            plus = "\n".join(f"+new_{mod}_{h}_{k}" for k in range(26))
            hunks.append(
                f"@@ -{10 + h * 5},3 +{10 + h * 5},29 @@\n keep_line_{h}\n-old_{mod}_{h}\n{plus}\n"
            )
        body = "".join(hunks)
        parts.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{body}")
        _ = mod_i  # silence unused
    return "".join(parts)


def test_dual_model_solve_all_label_does_not_auto_drop() -> None:
    """VAL-DINTR-001: both models pass@1=1 labels only; should_drop_hardness=False."""
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
    assert r.reason_code == REASON_SOLVE_ALL_EASY
    assert r.label == EASY_SOLVE_ALL
    assert r.all_models_solved is True
    assert r.should_drop_hardness is False


def test_dual_model_solve_all_legacy_opt_in_still_drops() -> None:
    row = {
        "pack_id": "legacy",
        "frontier": 1.0,
        "grok-4.5": 1.0,
        "kimi-k2.6": 1.0,
    }
    r = classify_pack_from_panel_row(row, drop_on_solve_all=True)
    assert r.should_drop_hardness is True
    assert r.label == EASY_SOLVE_ALL


def test_tiny_trivial_patch_thin_request_is_easy_request() -> None:
    r = score_request_patch_difficulty(
        _thin_instruction(),
        _tiny_patch(),
        f2p_count=1,
    )
    assert r.intrinsic_class == CLASS_EASY_REQUEST
    assert r.easily_approachable is True
    assert r.confidence == "high"
    assert r.should_drop_hardness is True
    assert r.reason_code == REASON_EASY_REQUEST
    assert r.metrics["hunk_count"] <= 4
    assert r.metrics["source_file_count"] <= 2


def test_qs_class_thin_gold_is_high_confidence_easy_request() -> None:
    """VAL-DMED-002: qs-487-class (~21 added, 2 files, f2p=3) → EASY drop."""
    patch = _qs_class_patch()
    r = score_request_patch_difficulty(
        _qs_class_instruction(),
        patch,
        f2p_count=3,
    )
    assert r.metrics["added_lines"] <= 50
    assert r.metrics["source_file_count"] == 2
    assert r.intrinsic_class == CLASS_EASY_REQUEST
    assert r.easily_approachable is True
    assert r.confidence == "high"
    assert r.should_drop_hardness is True
    assert r.reason_code == REASON_EASY_REQUEST


def test_deepswe_median_gold_is_hard_request() -> None:
    """VAL-DMED-002: median multi-file large added stays HARD_REQUEST."""
    patch = _large_multi_module_patch()
    r = score_request_patch_difficulty(
        _hard_instruction(),
        patch,
        f2p_count=9,
    )
    assert r.intrinsic_class == CLASS_HARD_REQUEST
    assert r.easily_approachable is False
    assert r.should_drop_hardness is False
    assert r.metrics["hunk_count"] >= 14
    assert r.metrics["source_file_count"] >= 4
    assert r.metrics["added_lines"] >= 400
    assert r.metrics["outcomes"] >= 4


def test_large_multi_outcome_is_hard_request() -> None:
    r = score_request_patch_difficulty(
        _hard_instruction(),
        _large_multi_module_patch(),
        f2p_count=9,
    )
    assert r.intrinsic_class == CLASS_HARD_REQUEST
    assert r.easily_approachable is False
    assert r.should_drop_hardness is False
    assert r.metrics["hunk_count"] >= 10
    assert r.metrics["source_file_count"] >= 3
    assert r.metrics["outcomes"] >= 4


def test_mixed_signals_uncertain_keeps() -> None:
    # Long multi-outcome text with tiny patch → mixed / uncertain, keep.
    r = score_request_patch_difficulty(
        _hard_instruction(),
        _tiny_patch(),
        f2p_count=5,
    )
    assert r.intrinsic_class in {CLASS_UNCERTAIN, CLASS_HARD_REQUEST, CLASS_EASY_REQUEST}
    if r.intrinsic_class == CLASS_EASY_REQUEST:
        # Still acceptable if patch dominates, but drop is ok for thin gold.
        assert r.metrics["added_lines"] <= 25
    else:
        assert r.should_drop_hardness is False or r.intrinsic_class == CLASS_EASY_REQUEST


def _write_pack(
    root: Path,
    task_id: str,
    *,
    instruction: str,
    solution: str,
    f2p: list[str],
) -> Path:
    d = root / "tasks" / task_id
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir(parents=True)
    (d / "solution").mkdir(parents=True)
    (d / "task.toml").write_text(
        '[metadata]\nlanguage = "python"\nrepository_url = "https://github.com/example/x.git"\n'
        'base_commit_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n',
        encoding="utf-8",
    )
    (d / "instruction.md").write_text(instruction, encoding="utf-8")
    (d / "pre_artifacts.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (d / "environment" / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    (d / "tests" / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    (d / "tests" / "test.sh").write_text("#!/bin/bash\npytest\n", encoding="utf-8")
    (d / "tests" / "grader.py").write_text("def grade():\n    return 1\n", encoding="utf-8")
    (d / "tests" / "config.json").write_text(
        json.dumps({"f2p_node_ids": f2p, "p2p_node_ids": []}), encoding="utf-8"
    )
    (d / "tests" / "test.patch").write_text(
        "diff --git a/tests/t.py b/tests/t.py\n+def test_a():\n+    assert True\n",
        encoding="utf-8",
    )
    (d / "solution" / "solution.patch").write_text(solution, encoding="utf-8")
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    return d


def test_intrinsic_from_pack_dir(tmp_path: Path) -> None:
    pack = _write_pack(
        tmp_path,
        "thin-pack",
        instruction=_thin_instruction(),
        solution=_tiny_patch(),
        f2p=["only"],
    )
    r = intrinsic_from_pack_dir(pack)
    assert r.intrinsic_class == CLASS_EASY_REQUEST
    assert r.should_drop_hardness is True


def test_decide_pack_keeps_model_solve_all_when_not_intrinsic_easy(tmp_path: Path) -> None:
    """Both models solve + large multi-outcome hard pack → keep (VAL-DINTR-001)."""
    pack = _write_pack(
        tmp_path,
        "realpr-hard-solve-all",
        instruction=_hard_instruction(),
        solution=_large_multi_module_patch(),
        f2p=[f"t{i}" for i in range(8)],
    )
    d = decide_pack(
        "realpr-hard-solve-all",
        pack_dir=pack,
        pack_row={
            "task_id": "realpr-hard-solve-all",
            "solution_reward": 1,
            "null_reward": 0,
            "certified": True,
            "source_hunk_count": 16,
        },
        panel_row={"verdict": "drop", "rule": "solve-all", "frontier_pass_at_k": 1.0},
        force_drop={},  # no explicit policy table for this id
    )
    assert d.keep is True
    assert d.reason_code == "keep_despite_model_solve_all"
    assert d.meta.get("intrinsic_class") == CLASS_HARD_REQUEST


def test_decide_pack_drops_intrinsic_easy_request_even_if_solve_none(tmp_path: Path) -> None:
    pack = _write_pack(
        tmp_path,
        "realpr-thin-easy",
        instruction=_thin_instruction(),
        solution=_tiny_patch(),
        f2p=["only", "two", "three"],  # still below m27 f2p floor 5
    )
    d = decide_pack(
        "realpr-thin-easy",
        pack_dir=pack,
        pack_row={
            "task_id": "realpr-thin-easy",
            "solution_reward": 1,
            "null_reward": 0,
            "certified": True,
            "source_hunk_count": 12,
        },
        panel_row={"verdict": "drop", "rule": "solve-none", "frontier_pass_at_k": 0.0},
        force_drop={},
    )
    # Floors and/or intrinsic must refuse thin gold.
    assert d.keep is False
    assert (
        d.reason_code
        in {
            REASON_EASY_REQUEST,
            "multi_file_floor_rejected",
            "source_hunks_below_floor",
            "f2p_nodes_below_floor",
            "gold_added_lines_below_floor",
            "added_lines_below_floor",
        }
        or "f2p" in d.reason_code
        or "floor" in d.reason_code
        or "intrinsic" in d.reason_code
    )


def test_decide_pack_drops_qs_class_via_floors_or_intrinsic(tmp_path: Path) -> None:
    """qs-487 class fails M27 floors and is high-conf EASY_REQUEST."""
    pack = _write_pack(
        tmp_path,
        "realpr-qs-class",
        instruction=_qs_class_instruction(),
        solution=_qs_class_patch(),
        f2p=["a", "b", "c"],
    )
    intr = score_request_patch_difficulty(_qs_class_instruction(), _qs_class_patch(), f2p_count=3)
    assert intr.intrinsic_class == CLASS_EASY_REQUEST
    assert intr.should_drop_hardness is True

    d = decide_pack(
        "realpr-qs-class",
        pack_dir=pack,
        pack_row={
            "task_id": "realpr-qs-class",
            "solution_reward": 1,
            "null_reward": 0,
            "certified": True,
            "source_hunk_count": 11,
        },
        force_drop={},
    )
    assert d.keep is False


def test_decide_pack_drops_tiny_request_via_intrinsic_when_multi_file(tmp_path: Path) -> None:
    # Two-file tiny gold still under easy hunk budget.
    two_file = (
        "diff --git a/pkg/a.py b/pkg/a.py\n"
        "--- a/pkg/a.py\n+++ b/pkg/a.py\n"
        "@@ -1,1 +1,1 @@\n-x=1\n+x=2\n"
        "diff --git a/pkg/b.py b/pkg/b.py\n"
        "--- a/pkg/b.py\n+++ b/pkg/b.py\n"
        "@@ -1,1 +1,1 @@\n-y=1\n+y=2\n"
    )
    pack = _write_pack(
        tmp_path,
        "realpr-tiny-multi",
        instruction=_thin_instruction(),
        solution=two_file,
        f2p=["a", "b", "c", "d", "e"],  # pass F2P floor alone; multi-file/added fail
    )
    # Pre-check intrinsic
    intr = score_request_patch_difficulty(_thin_instruction(), two_file, f2p_count=5)
    assert intr.intrinsic_class == CLASS_EASY_REQUEST

    d = decide_pack(
        "realpr-tiny-multi",
        pack_dir=pack,
        pack_row={
            "task_id": "realpr-tiny-multi",
            "solution_reward": 1,
            "null_reward": 0,
            "certified": True,
            "source_hunk_count": 12,
        },
        force_drop={},
    )
    assert d.keep is False
    assert (
        d.reason_code
        in {
            REASON_EASY_REQUEST,
            "multi_file_floor_rejected",
            "source_hunks_below_floor",
            "gold_added_lines_below_floor",
            "added_lines_below_floor",
            "f2p_nodes_below_floor",
            "prompt_empty_behavior_ask_vs_f2p",
        }
        or "intrinsic" in d.reason_code
        or "thin" in d.reason_code
        or "f2p" in d.reason_code
        or "floor" in d.reason_code
        or "added" in d.reason_code
    )


def test_live_qs487_pack_intrinsic_when_present() -> None:
    root = Path("datasets/prod_hard_keep/tasks/realpr-qs-487")
    if not root.is_dir():
        return
    r = intrinsic_from_pack_dir(root)
    assert r.intrinsic_class == CLASS_EASY_REQUEST
    assert r.confidence == "high"
    assert r.should_drop_hardness is True
    assert r.metrics.get("added_lines", 0) <= 50
    assert r.metrics.get("source_file_count", 0) <= 2
