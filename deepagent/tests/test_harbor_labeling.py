"""Dual-run F2P/P2P labeling (VAL-LABEL-001..006).

Covers:
- VAL-LABEL-001: F2P fail@base pass@gold; empty F2P rejected
- VAL-LABEL-002: P2P pass-both; disjoint from F2P
- VAL-LABEL-003: node ids persist into tests/config.json
- VAL-LABEL-004: held-out test.patch verifier-only / agent isolation
- VAL-LABEL-005: deterministic recompute for fixed suite outcomes
- VAL-LABEL-006: multi-lang suite reporters feed dual-run + config
- Flake reject on dual-run signature disagreement
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_factory.producers.harbor_labeling import (
    LABEL_EMPTY_F2P,
    LABEL_FLAKE_REJECT,
    LABEL_G2_FLAKE,
    HarborLabelError,
    SuiteOutcome,
    assert_held_out_verifier_only,
    assert_no_dual_run_flake,
    compute_dual_run_labels,
    detect_dual_run_flake,
    label_cohorts_from_outcomes,
    labels_from_suite_outcomes,
    labels_to_tests_config,
    write_tests_config_json,
)
from swe_factory.producers.harbor_motors import (
    get_motor_seed,
    produce_harbor_materials,
    produce_harbor_pack,
)
from swe_factory.producers.suite_reporters import (
    GoSuiteReporter,
    PythonSuiteReporter,
    RustSuiteReporter,
    TypeScriptSuiteReporter,
    get_suite_reporter,
    grade_tool_label_for,
    list_reporter_languages,
    parse_with_reporter,
    reporter_info,
    reporter_registry_snapshot,
)


def _fixed_pair() -> tuple[SuiteOutcome, SuiteOutcome]:
    green = SuiteOutcome.from_summary(
        language="python",
        passed=("tests.a.test_f2p", "tests.a.test_p2p", "tests.b.test_other"),
        failed=(),
    )
    broken = SuiteOutcome.from_summary(
        language="python",
        passed=("tests.a.test_p2p", "tests.b.test_other"),
        failed=("tests.a.test_f2p",),
    )
    return green, broken


# ---------------------------------------------------------------------------
# VAL-LABEL-001 / VAL-LABEL-002 pure math
# ---------------------------------------------------------------------------


def test_val_label_001_f2p_fail_base_pass_gold() -> None:
    green, broken = _fixed_pair()
    labels = labels_from_suite_outcomes(green, broken)
    assert labels.f2p_node_ids == ("tests.a.test_f2p",)
    assert "tests.a.test_f2p" in broken.failed_set
    assert "tests.a.test_f2p" in green.passed_set
    assert labels.accepted is True


def test_val_label_001_empty_f2p_rejected() -> None:
    green = SuiteOutcome.from_summary(
        language="python",
        passed=("a", "b"),
        failed=(),
    )
    # broken also improves nothing — both pass both → empty F2P
    broken = SuiteOutcome.from_summary(
        language="python",
        passed=("a", "b"),
        failed=(),
    )
    with pytest.raises(HarborLabelError) as excinfo:
        labels_from_suite_outcomes(green, broken, require_nonempty_f2p=True)
    assert LABEL_EMPTY_F2P in excinfo.value.reason_codes


def test_val_label_002_p2p_pass_both_disjoint() -> None:
    green, broken = _fixed_pair()
    labels = labels_from_suite_outcomes(green, broken)
    assert set(labels.p2p_node_ids) == {"tests.a.test_p2p", "tests.b.test_other"}
    assert set(labels.f2p_node_ids).isdisjoint(labels.p2p_node_ids)
    # Every P2P passes green and broken
    for node in labels.p2p_node_ids:
        assert node in green.passed_set
        assert node in broken.passed_set


def test_label_cohorts_intersection_empty() -> None:
    f2p, p2p = label_cohorts_from_outcomes(
        green_passed={"a", "b", "c"},
        green_failed=set(),
        broken_passed={"b", "c"},
        broken_failed={"a"},
    )
    assert f2p == ("a",)
    assert p2p == ("b", "c")
    assert set(f2p).isdisjoint(p2p)


# ---------------------------------------------------------------------------
# VAL-LABEL-003 config.json persistence
# ---------------------------------------------------------------------------


def test_val_label_003_write_config_json(tmp_path: Path) -> None:
    green, broken = _fixed_pair()
    labels = labels_from_suite_outcomes(green, broken)
    path = write_tests_config_json(
        tmp_path / "tests" / "config.json",
        base_commit="a" * 40,
        f2p_node_ids=labels.f2p_node_ids,
        p2p_node_ids=labels.p2p_node_ids,
        grade={"format": "junit", "tool_label": "pytest", "node_id": "name"},
    )
    assert path.is_file()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"] == list(labels.f2p_node_ids)
    assert cfg["p2p_node_ids"] == list(labels.p2p_node_ids)
    assert cfg["base_commit"] == "a" * 40
    assert all(isinstance(n, str) and n.strip() for n in cfg["f2p_node_ids"])
    assert set(cfg["f2p_node_ids"]).isdisjoint(cfg["p2p_node_ids"])


def test_val_label_003_labels_to_tests_config(tmp_path: Path) -> None:
    green, broken = _fixed_pair()
    labels = labels_from_suite_outcomes(green, broken)
    payload = labels_to_tests_config(
        labels,
        base_commit="b" * 40,
        grade={"tool_label": "go-test"},
        dest=tmp_path / "config.json",
    )
    assert payload["f2p_node_ids"]
    on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert on_disk == payload


def test_write_config_rejects_empty_f2p(tmp_path: Path) -> None:
    with pytest.raises(HarborLabelError) as excinfo:
        write_tests_config_json(
            tmp_path / "config.json",
            base_commit="c" * 40,
            f2p_node_ids=[],
            p2p_node_ids=["x"],
        )
    assert LABEL_EMPTY_F2P in excinfo.value.reason_codes


def test_write_config_rejects_overlap(tmp_path: Path) -> None:
    with pytest.raises(HarborLabelError, match="overlap"):
        write_tests_config_json(
            tmp_path / "config.json",
            base_commit="d" * 40,
            f2p_node_ids=["same"],
            p2p_node_ids=["same", "other"],
        )


def test_produce_pack_config_has_dual_run_ids(tmp_path: Path) -> None:
    """VAL-LABEL-003 end-to-end via pack export."""
    result = produce_harbor_pack(
        "harbor_python_orders",
        out_dir=tmp_path / "out",
        work_root=tmp_path / "work",
        instance_suffix="lab003",
    )
    cfg = json.loads((result.pack_dir / "tests" / "config.json").read_text(encoding="utf-8"))
    assert len(cfg["f2p_node_ids"]) >= 1
    assert "p2p_node_ids" in cfg
    assert set(cfg["f2p_node_ids"]).isdisjoint(cfg.get("p2p_node_ids") or [])
    assert result.materials.f2p_node_ids
    assert list(result.materials.f2p_node_ids) == cfg["f2p_node_ids"]


# ---------------------------------------------------------------------------
# VAL-LABEL-004 held-out verifier-only
# ---------------------------------------------------------------------------


def test_val_label_004_held_out_not_in_agent(tmp_path: Path) -> None:
    seed = get_motor_seed("harbor_python_orders")
    result = produce_harbor_pack(
        seed,
        out_dir=tmp_path / "out",
        work_root=tmp_path / "work",
        instance_suffix="lab004",
    )
    assert result.pack_dir is not None
    env_repo = result.pack_dir / "environment" / "repo"
    hits = assert_held_out_verifier_only(
        agent_context=env_repo,
        test_patch_path=result.pack_dir / "tests" / "test.patch",
        held_out_relative_paths=(seed.held_out.relative_path,),
    )
    assert hits == []
    assert not (env_repo / seed.held_out.relative_path).exists()
    assert (result.pack_dir / "tests" / "test.patch").read_text(encoding="utf-8").strip()


def test_val_label_004_detects_leaked_test_patch(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "tests").mkdir()
    (agent / "tests" / "test.patch").write_text("leak\n", encoding="utf-8")
    hits = assert_held_out_verifier_only(agent_context=agent)
    assert any("test.patch" in h for h in hits)


def test_val_label_004_materials_writes_config_and_held_out_out_of_broken(
    tmp_path: Path,
) -> None:
    materials = produce_harbor_materials(
        get_motor_seed("harbor_go_kvstore"),
        work_root=tmp_path / "work",
        instance_suffix="lab004m",
    )
    held = materials.broken_workspace / "store" / "held_out_test.go"
    # go held-out path from seed
    seed = get_motor_seed("harbor_go_kvstore")
    assert not (materials.broken_workspace / seed.held_out.relative_path).exists()
    cfg = json.loads(
        (materials.broken_workspace.parent / "config.json").read_text(encoding="utf-8")
        if (materials.broken_workspace.parent / "config.json").is_file()
        else "{}"
    )
    # config sits next to green/broken under the case dir
    case = materials.broken_workspace.parent
    cfg = json.loads((case / "config.json").read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"]
    assert materials.test_patch.strip()
    _ = held


# ---------------------------------------------------------------------------
# VAL-LABEL-005 determinism
# ---------------------------------------------------------------------------


def test_val_label_005_recompute_deterministic() -> None:
    green, broken = _fixed_pair()
    first = labels_from_suite_outcomes(green, broken)
    second = labels_from_suite_outcomes(green, broken)
    assert first.f2p_node_ids == second.f2p_node_ids
    assert first.p2p_node_ids == second.p2p_node_ids
    # order stable (sorted)
    assert list(first.f2p_node_ids) == sorted(first.f2p_node_ids)
    assert list(first.p2p_node_ids) == sorted(first.p2p_node_ids)


def test_val_label_005_live_motor_twice(tmp_path: Path) -> None:
    """Two produce_harbor_materials calls yield identical node id sets."""
    seed = get_motor_seed("harbor_python_orders")
    a = produce_harbor_materials(seed, work_root=tmp_path / "a", instance_suffix="d1")
    b = produce_harbor_materials(seed, work_root=tmp_path / "b", instance_suffix="d2")
    assert a.f2p_node_ids == b.f2p_node_ids
    assert a.p2p_node_ids == b.p2p_node_ids


# ---------------------------------------------------------------------------
# Flake reject
# ---------------------------------------------------------------------------


def test_flake_detect_on_disagreement() -> None:
    r1 = SuiteOutcome.from_summary(
        language="python",
        passed=("a", "b"),
        failed=("c",),
    )
    r2 = SuiteOutcome.from_summary(
        language="python",
        passed=("a",),
        failed=("b", "c"),
    )
    is_flake, codes, details = detect_dual_run_flake([r1, r2], phase="gold")
    assert is_flake is True
    assert LABEL_G2_FLAKE in codes
    assert LABEL_FLAKE_REJECT in codes
    assert details["run_count"] == 2


def test_assert_no_dual_run_flake_raises() -> None:
    r1 = SuiteOutcome.from_summary(language="go", passed=("T1",), failed=("T2",))
    r2 = SuiteOutcome.from_summary(language="go", passed=("T1", "T2"), failed=())
    with pytest.raises(HarborLabelError) as excinfo:
        assert_no_dual_run_flake([r1, r2], phase="gold")
    assert LABEL_FLAKE_REJECT in excinfo.value.reason_codes


def test_stable_dual_run_no_flake() -> None:
    r1 = SuiteOutcome.from_summary(language="go", passed=("T1",), failed=("T2",))
    r2 = SuiteOutcome.from_summary(language="go", passed=("T1",), failed=("T2",))
    is_flake, codes, _ = detect_dual_run_flake([r1, r2])
    assert is_flake is False
    assert codes == []
    assert_no_dual_run_flake([r1, r2])


# ---------------------------------------------------------------------------
# VAL-LABEL-006 multi-lang reporters
# ---------------------------------------------------------------------------


def test_val_label_006_reporter_registry() -> None:
    langs = list_reporter_languages()
    assert set(langs) >= {"python", "go", "typescript", "javascript", "rust"}
    snap = reporter_registry_snapshot()
    assert snap["python"]["tool_label"] == "pytest"
    assert snap["go"]["tool_label"] == "go-test"
    assert snap["typescript"]["tool_label"] == "jest"
    assert snap["rust"]["tool_label"] == "cargo-test"
    assert grade_tool_label_for("python") == "pytest"
    assert grade_tool_label_for("go") == "go-test"
    for lang in langs:
        info = reporter_info(lang)
        assert info.reporter_id
        rep = get_suite_reporter(lang)
        assert rep.tool_label == info.tool_label


def test_val_label_006_python_reporter_parse() -> None:
    text = json.dumps(
        {
            "rc": 1,
            "passed": ["tests.a.test_ok"],
            "failed": ["tests.a.test_f2p"],
            "errors": [],
        }
    )
    out = PythonSuiteReporter().parse_log(text, returncode=1)
    assert "tests.a.test_ok" in out.passed
    assert "tests.a.test_f2p" in out.failed


def test_val_label_006_go_reporter_parse() -> None:
    log = "--- PASS: TestStoreSetGet (0.00s)\n--- FAIL: TestRouterUpsert (0.00s)\n"
    out = GoSuiteReporter().parse_log(log, returncode=1)
    assert out.passed == ("TestStoreSetGet",)
    assert out.failed == ("TestRouterUpsert",)
    # dual-run from parsed logs
    green = GoSuiteReporter().parse_log(
        "--- PASS: TestStoreSetGet (0.00s)\n--- PASS: TestRouterUpsert (0.00s)\n",
        returncode=0,
    )
    labels = labels_from_suite_outcomes(green, out)
    assert labels.f2p_node_ids == ("TestRouterUpsert",)
    assert labels.p2p_node_ids == ("TestStoreSetGet",)


def test_val_label_006_ts_reporter_parse() -> None:
    log = "  ✓ catalog add and get\n  ✕ catalog findByTag\n"
    out = TypeScriptSuiteReporter().parse_log(log, returncode=1)
    assert "catalog add and get" in out.passed
    assert "catalog findByTag" in out.failed
    green = TypeScriptSuiteReporter().parse_log(
        "  ✓ catalog add and get\n  ✓ catalog findByTag\n",
        returncode=0,
    )
    labels = labels_from_suite_outcomes(green, out)
    assert "catalog findByTag" in labels.f2p_node_ids
    assert "catalog add and get" in labels.p2p_node_ids


def test_val_label_006_js_and_rust_reporters() -> None:
    js = TypeScriptSuiteReporter(as_javascript=True)
    assert js.language == "javascript"
    rust_log = "test crate::util::it_works ... ok\ntest crate::util::broken_feature ... FAILED\n"
    rust = RustSuiteReporter().parse_log(rust_log, returncode=1)
    assert "crate::util::it_works" in rust.passed
    assert "crate::util::broken_feature" in rust.failed
    green = RustSuiteReporter().parse_log(
        "test crate::util::it_works ... ok\ntest crate::util::broken_feature ... ok\n",
        returncode=0,
    )
    labels = labels_from_suite_outcomes(green, rust)
    assert labels.f2p_node_ids == ("crate::util::broken_feature",)
    assert labels.p2p_node_ids == ("crate::util::it_works",)


def test_val_label_006_motor_languages_reporter_notes(tmp_path: Path) -> None:
    """Each adapted motor records reporter identity and non-empty F2P."""
    for seed_id in ("harbor_python_orders", "harbor_go_kvstore", "harbor_ts_registry"):
        mats = produce_harbor_materials(
            get_motor_seed(seed_id),
            work_root=tmp_path / seed_id,
            instance_suffix="rep",
        )
        assert mats.f2p_node_ids
        assert set(mats.f2p_node_ids).isdisjoint(mats.p2p_node_ids)
        rep = mats.notes.get("suite_reporter") or {}
        assert rep.get("language") in {
            "python",
            "go",
            "typescript",
            "javascript",
            "rust",
        }
        assert rep.get("tool_label")
        case = mats.broken_workspace.parent
        cfg = json.loads((case / "config.json").read_text(encoding="utf-8"))
        assert cfg["f2p_node_ids"] == list(mats.f2p_node_ids)
        assert cfg["grade"]["tool_label"] == grade_tool_label_for(mats.language)


def test_live_dual_run_cov_via_compute(tmp_path: Path) -> None:
    """Live suite dual-run still non-empty F2P for python motor (VAL-LABEL-001)."""
    seed = get_motor_seed("harbor_python_orders")
    from swe_factory.producers.harbor_motors import _apply_fault_plan, _copy_tree

    green = tmp_path / "green"
    broken = tmp_path / "broken"
    _copy_tree(seed.green_repo(), green)
    _copy_tree(seed.green_repo(), broken)
    _apply_fault_plan(broken, seed.fault)
    labels = compute_dual_run_labels(
        language=seed.language,
        green_repo=green,
        broken_repo=broken,
        held_out_relative_path=seed.held_out.relative_path,
        held_out_content=seed.held_out.content,
    )
    assert len(labels.f2p_node_ids) >= 1
    assert set(labels.f2p_node_ids).isdisjoint(labels.p2p_node_ids)
    assert labels.notes.get("reporter")


def test_parse_with_reporter_dispatch() -> None:
    out = parse_with_reporter(
        "go",
        "--- PASS: TestA (0s)\n--- FAIL: TestB (0s)\n",
        returncode=1,
    )
    assert out.language == "go"
    assert out.failed == ("TestB",)
