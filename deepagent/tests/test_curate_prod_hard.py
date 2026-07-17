"""M21c curation: prod hardness panel from test_n10 (VAL-DHARD-004)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_factory.pipeline.curate_prod_hard import (
    DEFAULT_OUT,
    EXPLICIT_DROP,
    MIN_HARD_KEEP,
    NOMINAL_KEEP_CANDIDATES,
    ProdHardCurationError,
    curate_dispositions,
    curate_hardness_from_scoreboard,
    decide_pack,
    materialize_prod_hard_keep,
    recover_solve_all_only_drops,
)


def _write_minimal_pack(
    root: Path,
    task_id: str,
    *,
    instruction: str,
    f2p: list[str],
    solution_diff: str,
    test_patch: str = "diff --git a/tests/t.py b/tests/t.py\n+def test_a():\n+    assert True\n",
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
    (d / "tests" / "test.patch").write_text(test_patch, encoding="utf-8")
    (d / "solution" / "solution.patch").write_text(solution_diff, encoding="utf-8")
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    return d


def _aligned_instruction() -> str:
    return (
        "# Fix complex negotiation behaviour\n\n"
        "Implement multi-step adapter validation so nested fields reject "
        "invalid payloads and preserve document order.\n\n"
        "## Expected outcomes\n"
        "1. Nested mapping adapters validate field types and raise clear errors.\n"
        "2. List adapters preserve insertion order across round-trips.\n"
        "3. Schema export mirrors runtime adapter names without dropping \n"
        "required fields.\n"
        "4. Missing required fields produce explicit contract errors.\n\n"
        "## Constraints\n"
        "- Touch only the adapter package sources needed for the behaviour.\n"
        "- Keep public API names stable except where requirements demand new hooks.\n\n"
        "IMPORTANT: Please work on this in a new branch from main and commit "
        "everything when you are done.\n"
    )


def _misaligned_instruction() -> str:
    return (
        "Bump the package version and public API surface only.\n\n"
        "## Expected outcomes\n"
        "1. The package version string is updated to 11.0.0.\n"
        "2. __all__ lists public exports.\n\n"
        "## Constraints\n"
        "- Do not change the runtime behavior of existing iterators.\n"
        "- Keep version string centralised.\n"
    )


def _multi_file_gold() -> str:
    return (
        "diff --git a/pkg/a.py b/pkg/a.py\n"
        "--- a/pkg/a.py\n+++ b/pkg/a.py\n"
        "@@ -1,1 +1,3 @@\n"
        " x=1\n+def f():\n+    return 2\n"
        "diff --git a/pkg/b.py b/pkg/b.py\n"
        "--- a/pkg/b.py\n+++ b/pkg/b.py\n"
        "@@ -1,1 +1,2 @@\n"
        " y=1\n+z=3\n"
    )


def _behavioral_test_patch() -> str:
    return (
        "diff --git a/tests/test_adapter.py b/tests/test_adapter.py\n"
        "--- a/tests/test_adapter.py\n+++ b/tests/test_adapter.py\n"
        "@@ -0,0 +1,20 @@\n"
        "+def test_nested_rejects_bad_type():\n"
        "+    with pytest.raises(TypeError):\n"
        "+        adapt({'a': object()})\n"
        "+def test_list_preserves_order():\n"
        "+    assert adapt([1,2]) == [1,2]\n"
        "+def test_schema_required_fields():\n"
        "+    assert 'req' in schema()\n"
        "+def test_missing_field_error():\n"
        "+    with pytest.raises(KeyError):\n"
        "+        adapt({})\n"
    )


def _build_src_corpus(tmp: Path) -> Path:
    src = tmp / "test_n10"
    packs = {
        # hard keeps
        "realpr-itemadapter-101": {
            "instr": _aligned_instruction(),
            "f2p": [f"t{i}" for i in range(5)],
            "sol": 1,
            "null": 0,
            "hunks": 15,
        },
        "realpr-attrs-1323": {
            "instr": _aligned_instruction(),
            "f2p": ["a", "b", "c", "d"],
            "sol": 1,
            "null": 0,
            "hunks": 14,
        },
        "realpr-httpx-3672": {
            "instr": _aligned_instruction(),
            "f2p": ["h1", "h2", "h3", "h4", "h5"],
            "sol": 1,
            "null": 0,
            "hunks": 18,
        },
        "realpr-attrs-1457": {
            "instr": _aligned_instruction(),
            "f2p": [f"x{i}" for i in range(6)],
            "sol": 1,
            "null": 0,
            "hunks": 21,
        },
        "realpr-packaging-1120": {
            "instr": _aligned_instruction(),
            "f2p": [f"p{i}" for i in range(9)],
            "sol": 1,
            "null": 0,
            "hunks": 24,
        },
        # misalign drop
        "realpr-more-itertools-1136": {
            "instr": _misaligned_instruction(),
            "f2p": ["m1", "m2", "m3", "m4"],
            "sol": 1,
            "null": 0,
            "hunks": 15,
        },
        # solve-all easy (thin f2p)
        "realpr-charset-normalizer-715": {
            "instr": _aligned_instruction(),
            "f2p": ["only_one"],
            "sol": 1,
            "null": 0,
            "hunks": 62,
        },
        "realpr-rich-4070": {
            "instr": _aligned_instruction(),
            "f2p": ["only_one"],
            "sol": 1,
            "null": 0,
            "hunks": 33,
        },
        # thin keep-band that floors refuse
        "realpr-more-itertools-943": {
            "instr": _misaligned_instruction(),
            "f2p": ["only_one"],
            "sol": 1,
            "null": 0,
            "hunks": 16,
        },
        "realpr-rich-3486": {
            "instr": _aligned_instruction(),
            "f2p": ["only_one"],
            "sol": 1,
            "null": 0,
            "hunks": 12,
        },
    }
    pack_rows = []
    identity = {}
    for tid, meta in packs.items():
        _write_minimal_pack(
            src,
            tid,
            instruction=meta["instr"],
            f2p=meta["f2p"],
            solution_diff=_multi_file_gold(),
            test_patch=_behavioral_test_patch(),
        )
        pack_rows.append(
            {
                "task_id": tid,
                "certified": True,
                "solution_reward": meta["sol"],
                "null_reward": meta["null"],
                "source_hunk_count": meta["hunks"],
                "source_track": "real_pr",
                "language": "python",
                "backend": "docker",
                "label_method": "real_pr_dual_run_base_vs_gold",
                "live_mine": True,
            }
        )
        identity[tid] = {
            "base_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "language": "python",
            "license": "MIT",
            "repository_url": "https://github.com/example/x.git",
            "seed_id": "pr:1",
            "source_track": "real_pr",
        }
    manifest = {
        "count": len(pack_rows),
        "pack_count": len(pack_rows),
        "ok": True,
        "packs": pack_rows,
        "identity": identity,
        "product_surface": "datasets/test_n10",
        "live_generate_dest": True,
        "product_dest": False,
        "band": {"min": 5, "target": 10, "max": 10},
        "languages": {"python": len(pack_rows)},
        "mode": "ship_deepagent_real_pr_docker",
    }
    (src / "pack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (src / "oracle_evidence.json").write_text(
        json.dumps(
            {
                "backend": "docker",
                "certified_count": len(pack_rows),
                "records": [
                    {
                        "task_id": p["task_id"],
                        "solution_reward": 1,
                        "null_reward": 0,
                        "certified": True,
                    }
                    for p in pack_rows
                ],
            }
        ),
        encoding="utf-8",
    )
    return src


def test_explicit_drop_table() -> None:
    assert "realpr-more-itertools-1136" in EXPLICIT_DROP
    assert "realpr-charset-normalizer-715" in EXPLICIT_DROP
    assert "realpr-rich-4070" in EXPLICIT_DROP
    assert MIN_HARD_KEEP == 5
    assert "realpr-attrs-1457" in NOMINAL_KEEP_CANDIDATES
    assert "realpr-packaging-1120" in NOMINAL_KEEP_CANDIDATES


def test_decide_pack_drops_explicit_thin_f2p(tmp_path: Path) -> None:
    """charset thin F2P remains explicit structural drop (not model solve-all)."""
    src = _build_src_corpus(tmp_path)
    pack_row = {
        "task_id": "realpr-charset-normalizer-715",
        "solution_reward": 1,
        "null_reward": 0,
        "certified": True,
        "source_hunk_count": 62,
    }
    d = decide_pack(
        "realpr-charset-normalizer-715",
        pack_dir=src / "tasks" / "realpr-charset-normalizer-715",
        pack_row=pack_row,
        panel_row={"verdict": "drop", "rule": "solve-all", "frontier_pass_at_k": 1.0},
    )
    assert d.keep is False
    # M25: reason is structural thin F2P (or floor), not model solve-all alone.
    assert d.reason_code in {
        "thin_f2p_easy_class",
        "f2p_nodes_below_floor",
        "solve_all_easy_policy_drop",  # legacy name if table unmigrated
    }


def test_decide_pack_drops_misalign(tmp_path: Path) -> None:
    src = _build_src_corpus(tmp_path)
    d = decide_pack(
        "realpr-more-itertools-1136",
        pack_dir=src / "tasks" / "realpr-more-itertools-1136",
        pack_row={
            "task_id": "realpr-more-itertools-1136",
            "solution_reward": 1,
            "null_reward": 0,
            "certified": True,
            "source_hunk_count": 15,
        },
    )
    assert d.keep is False
    assert "misalign" in d.reason_code or "prompt" in d.reason_code


def test_decide_pack_keeps_legit_solve_none(tmp_path: Path) -> None:
    src = _build_src_corpus(tmp_path)
    d = decide_pack(
        "realpr-attrs-1457",
        pack_dir=src / "tasks" / "realpr-attrs-1457",
        pack_row={
            "task_id": "realpr-attrs-1457",
            "solution_reward": 1,
            "null_reward": 0,
            "certified": True,
            "source_hunk_count": 21,
        },
        panel_row={"verdict": "drop", "rule": "solve-none", "frontier_pass_at_k": 0.0},
    )
    assert d.keep is True
    assert d.reason_code == "keep_legit_hard_solve_none"
    assert d.dual_truth_ok is True


def test_curate_dispositions_counts(tmp_path: Path) -> None:
    src = _build_src_corpus(tmp_path)
    panel = {
        "pack_results": [
            {
                "pack_id": "realpr-charset-normalizer-715",
                "decision": {
                    "verdict": "drop",
                    "rule": "solve-all",
                    "frontier_pass_at_k": 1.0,
                },
            },
            {
                "pack_id": "realpr-attrs-1457",
                "decision": {
                    "verdict": "drop",
                    "rule": "solve-none",
                    "frontier_pass_at_k": 0.0,
                },
            },
            {
                "pack_id": "realpr-itemadapter-101",
                "decision": {
                    "verdict": "keep",
                    "rule": "in-band-high-discrimination",
                    "frontier_pass_at_k": 0.5,
                },
            },
        ]
    }
    disps = curate_dispositions(src, panel_report=panel)
    keep = {d.task_id for d in disps if d.keep}
    drop = {d.task_id for d in disps if not d.keep}
    assert "realpr-more-itertools-1136" in drop
    assert "realpr-charset-normalizer-715" in drop
    assert "realpr-rich-4070" in drop
    assert "realpr-itemadapter-101" in keep
    assert "realpr-attrs-1457" in keep
    assert "realpr-packaging-1120" in keep
    # thin F2P / misalign not keep
    assert "realpr-rich-3486" in drop
    assert "realpr-more-itertools-943" in drop
    assert len(keep) >= MIN_HARD_KEEP


def test_materialize_writes_drop_reasons_and_keeps(tmp_path: Path) -> None:
    src = _build_src_corpus(tmp_path)
    out = tmp_path / "prod_hard_keep"
    result = materialize_prod_hard_keep(src, out)
    assert result.ok
    assert result.pack_count >= MIN_HARD_KEEP
    assert (out / "pack_manifest.json").is_file()
    assert (out / "drop_reasons.json").is_file()
    assert (out / "PROVENANCE.md").is_file()
    assert (out / "report.md").is_file()
    man = json.loads((out / "pack_manifest.json").read_text(encoding="utf-8"))
    assert man["pack_count"] == result.pack_count
    assert "drop_reasons" in man
    assert "realpr-more-itertools-1136" in man["drop_reasons"]
    assert "realpr-charset-normalizer-715" in man["drop_reasons"]
    assert "realpr-rich-4070" in man["drop_reasons"]
    keep_ids = {p["task_id"] for p in man["packs"]}
    assert keep_ids == set(result.keep_ids)
    assert "realpr-more-itertools-1136" not in keep_ids
    for tid in result.keep_ids:
        assert (out / "tasks" / tid / "tests" / "config.json").is_file()
        assert (out / "tasks" / tid / "solution" / "solution.patch").is_file()
    prov = (out / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "Drop reasons" in prov
    assert "realpr-more-itertools-1136" in prov
    assert DEFAULT_OUT.as_posix() == "datasets/prod_hard_keep"


def test_materialize_under_yield_fail_closed(tmp_path: Path) -> None:
    src = tmp_path / "tiny"
    # Only two packs both solvable thin → residual 0 under floors
    _write_minimal_pack(
        src,
        "easy-1",
        instruction=_aligned_instruction(),
        f2p=["only"],
        solution_diff=_multi_file_gold(),
        test_patch=_behavioral_test_patch(),
    )
    _write_minimal_pack(
        src,
        "easy-2",
        instruction=_aligned_instruction(),
        f2p=["only"],
        solution_diff=_multi_file_gold(),
        test_patch=_behavioral_test_patch(),
    )
    (src / "pack_manifest.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "task_id": "easy-1",
                        "solution_reward": 1,
                        "null_reward": 0,
                        "certified": True,
                        "source_hunk_count": 12,
                    },
                    {
                        "task_id": "easy-2",
                        "solution_reward": 1,
                        "null_reward": 0,
                        "certified": True,
                        "source_hunk_count": 12,
                    },
                ],
                "identity": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProdHardCurationError, match="residual"):
        materialize_prod_hard_keep(src, tmp_path / "out")


def test_materialize_in_place_keeps_model_solve_all_when_hard(tmp_path: Path) -> None:
    """M25: dual-model solve-all alone does not drop (VAL-DINTR-001).

    In-place curate-hardness must also not wipe source when src==out.
    """
    src = _build_src_corpus(tmp_path)
    # Dual-truth hard pack with large multi-outcome instruction; scoreboard marks
    # both models solve — under M25 this remains a hardness keep.
    _write_minimal_pack(
        src,
        "realpr-solve-all-x",
        instruction=_aligned_instruction(),
        f2p=["s1", "s2", "s3", "s4", "s5"],
        solution_diff=_multi_file_gold()
        + (
            # pad hunks so source_hunk floor is honest from patch if counted
            "diff --git a/pkg/c.py b/pkg/c.py\n"
            "--- a/pkg/c.py\n+++ b/pkg/c.py\n"
            "@@ -1,3 +1,8 @@\n keep\n-old\n+new1\n+new2\n+new3\n+new4\n+new5\n"
            "@@ -10,2 +15,4 @@\n base\n-old2\n+n1\n+n2\n+n3\n"
            "diff --git a/pkg/d.py b/pkg/d.py\n"
            "--- a/pkg/d.py\n+++ b/pkg/d.py\n"
            "@@ -1,2 +1,5 @@\n a\n-b\n+c\n+d\n+e\n"
            "@@ -5,2 +8,4 @@\n x\n-y\n+z1\n+z2\n+z3\n"
            "diff --git a/pkg/e.py b/pkg/e.py\n"
            "--- a/pkg/e.py\n+++ b/pkg/e.py\n"
            "@@ -1,2 +1,5 @@\n p\n-q\n+r1\n+r2\n+r3\n"
            "@@ -8,1 +11,3 @@\n s\n+t1\n+t2\n"
        ),
        test_patch=_behavioral_test_patch(),
    )
    man = json.loads((src / "pack_manifest.json").read_text(encoding="utf-8"))
    man["packs"].append(
        {
            "task_id": "realpr-solve-all-x",
            "certified": True,
            "solution_reward": 1,
            "null_reward": 0,
            "source_hunk_count": 20,
            "source_track": "real_pr",
            "language": "python",
            "backend": "docker",
            "label_method": "real_pr_dual_run_base_vs_gold",
            "live_mine": True,
        }
    )
    man["identity"]["realpr-solve-all-x"] = {
        "base_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "language": "python",
        "license": "MIT",
        "repository_url": "https://github.com/example/x.git",
        "seed_id": "pr:99",
        "source_track": "real_pr",
    }
    man["pack_count"] = len(man["packs"])
    man["count"] = len(man["packs"])
    (src / "pack_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    scoreboard = {
        "models": ["x-ai/grok-4.5", "moonshotai/kimi-k2.6"],
        "per_pack": [
            {
                "pack_id": "realpr-itemadapter-101",
                "grok-4.5": 1.0,
                "kimi-k2.6": 0.0,
                "frontier": 0.5,
            },
            {
                "pack_id": "realpr-attrs-1323",
                "grok-4.5": 0.0,
                "kimi-k2.6": 1.0,
                "frontier": 0.5,
            },
            {
                "pack_id": "realpr-httpx-3672",
                "grok-4.5": 0.0,
                "kimi-k2.6": 0.0,
                "frontier": 0.0,
            },
            {
                "pack_id": "realpr-attrs-1457",
                "grok-4.5": 0.0,
                "kimi-k2.6": 0.0,
                "frontier": 0.0,
            },
            {
                "pack_id": "realpr-packaging-1120",
                "grok-4.5": 0.0,
                "kimi-k2.6": 0.0,
                "frontier": 0.0,
            },
            {
                "pack_id": "realpr-solve-all-x",
                "grok-4.5": 1.0,
                "kimi-k2.6": 1.0,
                "frontier": 1.0,
            },
        ],
    }
    sb_path = tmp_path / "scoreboard.json"
    sb_path.write_text(json.dumps(scoreboard), encoding="utf-8")

    result = curate_hardness_from_scoreboard(
        src,
        src,
        scoreboard=sb_path,
        min_keep=0,
        include_explicit_drops=False,
        clean_out=True,
        restore_solve_all=False,
    )
    assert result.ok
    # M25: solve-all-x stays (dual-truth + floors + not intrinsic easy).
    assert "realpr-solve-all-x" in result.keep_ids
    assert "realpr-solve-all-x" not in result.drop_ids
    assert (src / "tasks" / "realpr-itemadapter-101" / "solution" / "solution.patch").is_file()
    assert (src / "tasks" / "realpr-solve-all-x" / "solution" / "solution.patch").is_file()
    assert (src / "drop_reasons.json").is_file()
    assert (src / "curation_report.json").is_file()
    assert result.pack_count >= 5
    assert "realpr-itemadapter-101" in result.keep_ids


def _hard_instruction() -> str:
    return _aligned_instruction()


def _easy_instruction_tiny() -> str:
    return "Fix the typo in the docs title.\n"


def test_recover_solve_all_only_drops_restores_hard_not_intrinsic_easy(
    tmp_path: Path,
) -> None:
    """VAL-DINTR-003: re-admit solve-all-only drops that still pass gates."""
    archive = tmp_path / "archive"
    dest = tmp_path / "prod"
    hard_id = "realpr-archive-hard-1"
    easy_id = "realpr-archive-easy-1"
    _write_minimal_pack(
        archive,
        hard_id,
        instruction=_hard_instruction(),
        f2p=["a", "b", "c", "d", "e"],
        solution_diff=_multi_file_gold()
        + (
            "diff --git a/pkg/c.py b/pkg/c.py\n"
            "--- a/pkg/c.py\n+++ b/pkg/c.py\n"
            "@@ -1,3 +1,8 @@\n keep\n-old\n+new1\n+new2\n+new3\n+new4\n+new5\n"
            "@@ -10,2 +15,4 @@\n base\n-old2\n+n1\n+n2\n+n3\n"
            "diff --git a/pkg/d.py b/pkg/d.py\n"
            "--- a/pkg/d.py\n+++ b/pkg/d.py\n"
            "@@ -1,2 +1,5 @@\n a\n-b\n+c\n+d\n+e\n"
            "@@ -5,2 +8,4 @@\n x\n-y\n+z1\n+z2\n+z3\n"
        ),
        test_patch=_behavioral_test_patch(),
    )
    _write_minimal_pack(
        archive,
        easy_id,
        instruction=_easy_instruction_tiny(),
        f2p=["only1", "only2", "only3"],
        solution_diff=(
            "diff --git a/pkg/a.py b/pkg/a.py\n"
            "--- a/pkg/a.py\n+++ b/pkg/a.py\n"
            "@@ -1,1 +1,2 @@\n"
            " x=1\n+y=2\n"
        ),
        test_patch=_behavioral_test_patch(),
    )
    (archive / "pack_manifest.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "task_id": hard_id,
                        "certified": True,
                        "solution_reward": 1,
                        "null_reward": 0,
                        "source_hunk_count": 18,
                        "source_track": "real_pr",
                        "language": "python",
                        "backend": "docker",
                        "label_method": "real_pr_dual_run_base_vs_gold",
                    },
                    {
                        "task_id": easy_id,
                        "certified": True,
                        "solution_reward": 1,
                        "null_reward": 0,
                        "source_hunk_count": 11,
                        "source_track": "real_pr",
                        "language": "python",
                        "backend": "docker",
                        "label_method": "real_pr_dual_run_base_vs_gold",
                    },
                ],
                "identity": {
                    hard_id: {
                        "base_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "language": "python",
                        "license": "MIT",
                        "repository_url": "https://github.com/example/x.git",
                        "seed_id": "pr:1",
                        "source_track": "real_pr",
                    },
                    easy_id: {
                        "base_commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "language": "python",
                        "license": "MIT",
                        "repository_url": "https://github.com/example/x.git",
                        "seed_id": "pr:2",
                        "source_track": "real_pr",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    keep_id = "realpr-keep-train-1"
    _write_minimal_pack(
        dest,
        keep_id,
        instruction=_hard_instruction(),
        f2p=["k1", "k2", "k3", "k4"],
        solution_diff=_multi_file_gold(),
        test_patch=_behavioral_test_patch(),
    )
    man_packs = [
        {
            "task_id": keep_id,
            "certified": True,
            "solution_reward": 1,
            "null_reward": 0,
            "source_hunk_count": 15,
            "source_track": "real_pr",
        }
    ]
    for extra in ("realpr-extra-a", "realpr-extra-b", "realpr-extra-c", "realpr-extra-d"):
        _write_minimal_pack(
            dest,
            extra,
            instruction=_hard_instruction(),
            f2p=["e1", "e2", "e3", "e4"],
            solution_diff=_multi_file_gold(),
            test_patch=_behavioral_test_patch(),
        )
        man_packs.append(
            {
                "task_id": extra,
                "certified": True,
                "solution_reward": 1,
                "null_reward": 0,
                "source_hunk_count": 14,
                "source_track": "real_pr",
            }
        )
    (dest / "pack_manifest.json").write_text(
        json.dumps(
            {
                "packs": man_packs,
                "identity": {},
                "drop_reasons": {
                    hard_id: {
                        "reason_code": "solve_all_easy_policy_drop",
                        "detail": "EASY_SOLVE_ALL: both models pass@1=1.0",
                    },
                    easy_id: {
                        "reason_code": "solve_all_easy_policy_drop",
                        "detail": "EASY_SOLVE_ALL: both models pass@1=1.0",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (dest / "drop_reasons.json").write_text(
        json.dumps(
            {
                "drop_reasons": {
                    hard_id: {
                        "reason_code": "solve_all_easy_policy_drop",
                        "detail": "EASY_SOLVE_ALL: both models pass@1=1.0",
                    },
                    easy_id: {
                        "reason_code": "solve_all_easy_policy_drop",
                        "detail": "EASY_SOLVE_ALL: both models pass@1=1.0",
                    },
                },
                "drop_ids": [hard_id, easy_id],
                "keep_ids": [keep_id],
            }
        ),
        encoding="utf-8",
    )

    recovered = recover_solve_all_only_drops(
        dest,
        restore_roots=[archive],
        apply_intrinsic=True,
    )
    assert hard_id in recovered["recovered_ids"]
    assert easy_id not in recovered["recovered_ids"]
    assert (dest / "tasks" / hard_id / "solution" / "solution.patch").is_file()
    assert easy_id in recovered["skipped"]
    assert "re_eval_drop" in recovered["skipped"][easy_id]

    scoreboard = {
        "models": ["x-ai/grok-4.5", "moonshotai/kimi-k2.6"],
        "per_pack": [
            {
                "pack_id": keep_id,
                "grok-4.5": 0.0,
                "kimi-k2.6": 0.0,
                "frontier": 0.0,
            },
            {
                "pack_id": hard_id,
                "grok-4.5": 1.0,
                "kimi-k2.6": 1.0,
                "frontier": 1.0,
                "decision": "drop",
            },
        ],
    }
    sb = tmp_path / "sb.json"
    sb.write_text(json.dumps(scoreboard), encoding="utf-8")

    result = materialize_prod_hard_keep(
        dest,
        dest,
        scoreboard=sb,
        min_keep=5,
        include_explicit_drops=False,
        restore_solve_all=True,
        restore_roots=[archive],
        clean_out=True,
    )
    assert result.ok
    assert hard_id in result.keep_ids
    assert easy_id not in result.keep_ids
    assert hard_id not in result.drop_ids
    for code in (d.get("reason_code") for d in result.drop_reasons.values()):
        assert code != "solve_all_easy_policy_drop"
    keep_disp = {d.task_id: d for d in result.dispositions if d.keep}
    assert keep_disp[hard_id].reason_code == "keep_despite_model_solve_all"
    assert (dest / "tasks" / hard_id / "instruction.md").is_file()
    drop_blob = json.loads((dest / "drop_reasons.json").read_text(encoding="utf-8"))
    assert hard_id in (drop_blob.get("restored_solve_all_only") or [])
    assert hard_id not in (drop_blob.get("drop_ids") or [])


def test_recover_skips_materials_skeleton_not_harbor(tmp_path: Path) -> None:
    """Materials-only dirs (patch+meta) are not product harbor trees — skip."""
    materials = tmp_path / "live_materials" / "realpr-skel-1"
    materials.mkdir(parents=True)
    (materials / "meta.json").write_text("{}", encoding="utf-8")
    (materials / "solution.patch").write_text("diff\n", encoding="utf-8")
    dest = tmp_path / "prod"
    dest.mkdir()
    (dest / "drop_reasons.json").write_text(
        json.dumps(
            {
                "drop_reasons": {
                    "realpr-skel-1": {
                        "reason_code": "solve_all_easy_policy_drop",
                        "detail": "was solve-all",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out = recover_solve_all_only_drops(
        dest, restore_roots=[tmp_path / "live_materials"]
    )
    assert out["recovered_ids"] == []
    assert "realpr-skel-1" in out["skipped"]
