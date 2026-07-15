"""Real-PR dual-run on real language suites (VAL-RDUAL-001..004).

Covers:
- VAL-RDUAL-001: dual-run uses real suite reporter (pytest / go test / jest)
- VAL-RDUAL-002: F2P fail@base pass@gold; empty F2P rejected; |f2p|≥1 success
- VAL-RDUAL-003: P2P pass both; disjoint; tests/config.json node ids
- VAL-RDUAL-004: held-out test.patch applied only in verifier prepare path
- Flaky dual-run rejects candidate
- Multi-lang reporter hooks offline (go / jest / rust parsers)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from swe_factory.producers.harbor_labeling import (
    LABEL_EMPTY_F2P,
    LABEL_FLAKE_REJECT,
    LABEL_G2_FLAKE,
    HarborLabelError,
    SuiteOutcome,
)
from swe_factory.producers.real_dual_run import (
    RDUAL_AGENT_LEAK,
    RDUAL_EMPTY_TEST_PATCH,
    RDUAL_STUB_CONFIG,
    REAL_SUITE_COMMANDS,
    RealDualRunError,
    agent_context_excludes_test_patch,
    apply_unified_patch,
    assert_real_suite_reporter,
    assert_verifier_prepare_applies_test_patch,
    flake_reject_on_disagreement,
    label_real_pr_dual_run,
    label_real_pr_from_outcomes,
    labels_from_real_suite_outcomes,
    paths_touched_by_unified_diff,
    prepare_verifier_dual_workspaces,
    refuse_stub_only_config,
    suite_command_for,
)
from swe_factory.producers.suite_reporters import (
    grade_tool_label_for,
    list_reporter_languages,
    parse_with_reporter,
    reporter_info,
)

BASE_SHA = "a" * 40


def _pair(
    *,
    language: str = "python",
    f2p: str = "tests.mod.test_f2p",
    p2p: tuple[str, ...] = ("tests.mod.test_p2p",),
) -> tuple[SuiteOutcome, SuiteOutcome]:
    gold = SuiteOutcome.from_summary(
        language=language,
        passed=(f2p, *p2p),
        failed=(),
    )
    base = SuiteOutcome.from_summary(
        language=language,
        passed=p2p,
        failed=(f2p,),
    )
    return base, gold


# ---------------------------------------------------------------------------
# VAL-RDUAL-001 real suite reporter
# ---------------------------------------------------------------------------


def test_val_rdual_001_real_suite_commands_cover_langs() -> None:
    langs = set(list_reporter_languages())
    assert langs >= {"python", "go", "typescript", "javascript", "rust"}
    for lang in langs:
        cmd = suite_command_for(lang)
        assert cmd
        assert cmd == REAL_SUITE_COMMANDS[lang]
        meta = assert_real_suite_reporter(lang)
        assert meta["tool_label"] == grade_tool_label_for(lang)
        assert meta["suite_command"] == cmd
        assert meta["reporter_id"]
        # authentic tools, not motor stubs
        assert "fake" not in meta["tool_label"].lower()
        assert "stub" not in meta["reporter_id"].lower()


def test_val_rdual_001_refuse_stub_only_config() -> None:
    with pytest.raises(RealDualRunError) as excinfo:
        refuse_stub_only_config(
            f2p_node_ids=("tests.a.test_x",),
            source_track="real_pr",
            reporter=None,
            language=None,
        )
    assert RDUAL_STUB_CONFIG in excinfo.value.reason_codes

    with pytest.raises(RealDualRunError) as excinfo:
        refuse_stub_only_config(
            f2p_node_ids=("tests.a.test_x",),
            source_track="real_pr",
            reporter={"tool_label": "fake_suite", "reporter_id": "stub_only"},
        )
    assert RDUAL_STUB_CONFIG in excinfo.value.reason_codes

    # Happy path with real reporter meta
    refuse_stub_only_config(
        f2p_node_ids=("tests.a.test_x",),
        source_track="real_pr",
        reporter=reporter_info("python").to_dict(),
        language="python",
    )


def test_val_rdual_001_offline_result_names_real_suite_command(tmp_path: Path) -> None:
    base, gold = _pair()
    result = label_real_pr_from_outcomes(
        language="python",
        base_outcome=base,
        gold_outcome=gold,
        base_commit=BASE_SHA,
        test_patch="diff --git a/tests/held.py b/tests/held.py\n",
        config_dest=tmp_path / "tests" / "config.json",
    )
    assert "pytest" in result.suite_command
    assert result.reporter["tool_label"] == "pytest"
    assert result.reporter["reporter_id"] == "python_pytest_v1"
    assert result.config_payload["suite_command"] == result.suite_command
    assert result.config_payload["suite_reporter"]["reporter_id"] == "python_pytest_v1"


# ---------------------------------------------------------------------------
# VAL-RDUAL-002 F2P fail@base pass@gold
# ---------------------------------------------------------------------------


def test_val_rdual_002_f2p_fail_base_pass_gold() -> None:
    base, gold = _pair(f2p="tests.mod.test_new_feature")
    labels = labels_from_real_suite_outcomes(gold, base)
    assert labels.f2p_node_ids == ("tests.mod.test_new_feature",)
    assert "tests.mod.test_new_feature" in base.failed_set
    assert "tests.mod.test_new_feature" in gold.passed_set
    result = label_real_pr_from_outcomes(
        language="python",
        base_outcome=base,
        gold_outcome=gold,
        base_commit=BASE_SHA,
    )
    assert len(result.f2p_node_ids) >= 1
    assert result.accepted is True


def test_val_rdual_002_empty_f2p_rejected() -> None:
    gold = SuiteOutcome.from_summary(language="python", passed=("a", "b"), failed=())
    base = SuiteOutcome.from_summary(language="python", passed=("a", "b"), failed=())
    with pytest.raises(HarborLabelError) as excinfo:
        labels_from_real_suite_outcomes(gold, base, require_nonempty_f2p=True)
    assert LABEL_EMPTY_F2P in excinfo.value.reason_codes


def test_val_rdual_002_errors_on_base_count_as_fail() -> None:
    """Collection/setup errors on base should enter F2P when gold is clean."""
    node = "tests.mod.test_import_symbol"
    gold = SuiteOutcome.from_summary(
        language="python",
        passed=(node, "tests.mod.test_ok"),
        failed=(),
    )
    base = SuiteOutcome(
        language="python",
        passed=("tests.mod.test_ok",),
        failed=(),
        errors=(node,),
        returncode=2,
    )
    labels = labels_from_real_suite_outcomes(gold, base)
    assert node in labels.f2p_node_ids
    assert "tests.mod.test_ok" in labels.p2p_node_ids


# ---------------------------------------------------------------------------
# VAL-RDUAL-003 P2P + config.json
# ---------------------------------------------------------------------------


def test_val_rdual_003_p2p_disjoint_and_config(tmp_path: Path) -> None:
    base, gold = _pair(
        f2p="tests.a.test_f2p",
        p2p=("tests.a.test_p2p", "tests.b.test_regress"),
    )
    result = label_real_pr_from_outcomes(
        language="python",
        base_outcome=base,
        gold_outcome=gold,
        base_commit=BASE_SHA,
        config_dest=tmp_path / "tests" / "config.json",
    )
    assert set(result.p2p_node_ids) == {"tests.a.test_p2p", "tests.b.test_regress"}
    assert set(result.f2p_node_ids).isdisjoint(result.p2p_node_ids)
    assert result.config_path is not None
    cfg = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"] == list(result.f2p_node_ids)
    assert cfg["p2p_node_ids"] == list(result.p2p_node_ids)
    assert cfg["base_commit"] == BASE_SHA
    assert set(cfg["f2p_node_ids"]).isdisjoint(cfg["p2p_node_ids"])
    assert all(isinstance(n, str) and n.strip() for n in cfg["f2p_node_ids"])
    assert cfg["grade"]["tool_label"] == "pytest"
    assert cfg["source_track"] == "real_pr"


# ---------------------------------------------------------------------------
# VAL-RDUAL-004 held-out test.patch only in verifier prepare
# ---------------------------------------------------------------------------


def test_val_rdual_004_test_patch_paths_from_diff() -> None:
    patch = textwrap.dedent(
        """\
        diff --git a/tests/test_held.py b/tests/test_held.py
        new file mode 100644
        --- /dev/null
        +++ b/tests/test_held.py
        @@ -0,0 +1,3 @@
        +def test_held():
        +    assert True
        +
        """
    )
    paths = paths_touched_by_unified_diff(patch)
    assert paths == ["tests/test_held.py"]


def test_val_rdual_004_prepare_applies_test_patch_not_mutating_agent(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "pkg").mkdir()
    (agent / "pkg" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (agent / "tests").mkdir()
    (agent / "tests" / "test_core.py").write_text(
        "from pkg.core import VALUE\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )

    solution = textwrap.dedent(
        """\
        diff --git a/pkg/core.py b/pkg/core.py
        --- a/pkg/core.py
        +++ b/pkg/core.py
        @@ -1 +1 @@
        -VALUE = 1
        +VALUE = 2
        """
    )
    test_patch = textwrap.dedent(
        """\
        diff --git a/tests/test_held.py b/tests/test_held.py
        new file mode 100644
        --- /dev/null
        +++ b/tests/test_held.py
        @@ -0,0 +1,6 @@
        +from pkg.core import VALUE
        +
        +def test_held_upgraded():
        +    assert VALUE == 2
        +
        """
    )

    prep = prepare_verifier_dual_workspaces(
        base_repo=agent,
        solution_patch=solution,
        test_patch=test_patch,
        language="python",
        work_root=tmp_path / "work",
    )
    try:
        assert_verifier_prepare_applies_test_patch(prep.apply_log)
        assert any("test.patch" in line for line in prep.apply_log)
        # Verifier workspaces have held-out
        assert (prep.base_workspace / "tests" / "test_held.py").is_file()
        assert (prep.gold_workspace / "tests" / "test_held.py").is_file()
        # Gold has solution
        assert (prep.gold_workspace / "pkg" / "core.py").read_text(
            encoding="utf-8"
        ) == "VALUE = 2\n"
        # Base still broken for product (VALUE=1) but held-out present
        assert (prep.base_workspace / "pkg" / "core.py").read_text(
            encoding="utf-8"
        ) == "VALUE = 1\n"
        # Agent durable tree never received held-out or solution
        assert not (agent / "tests" / "test_held.py").exists()
        assert (agent / "pkg" / "core.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        hits = agent_context_excludes_test_patch(agent)
        assert hits == []
    finally:
        # work_root owned=false so explicit cleanup of children only
        pass


def test_val_rdual_004_agent_leak_detected(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "test.patch").write_text("leak\n", encoding="utf-8")
    hits = agent_context_excludes_test_patch(agent)
    assert any("test.patch" in h for h in hits)

    base = tmp_path / "base"
    base.mkdir()
    (base / "x.py").write_text("x=1\n", encoding="utf-8")
    sol = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x=1\n+x=2\n"
    tpatch = (
        "diff --git a/t.py b/t.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/t.py\n@@ -0,0 +1 @@\n+pass\n"
    )
    with pytest.raises(RealDualRunError) as excinfo:
        label_real_pr_dual_run(
            language="python",
            base_repo=base,
            solution_patch=sol,
            test_patch=tpatch,
            base_commit=BASE_SHA,
            agent_context=agent,
            offline_base_outcome=SuiteOutcome.from_summary(
                language="python", passed=(), failed=("t",)
            ),
            offline_gold_outcome=SuiteOutcome.from_summary(
                language="python", passed=("t",), failed=()
            ),
        )
    assert RDUAL_AGENT_LEAK in excinfo.value.reason_codes


def test_val_rdual_004_empty_test_patch_refused(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.py").write_text("1\n", encoding="utf-8")
    sol = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-1\n+2\n"
    with pytest.raises(RealDualRunError) as excinfo:
        prepare_verifier_dual_workspaces(
            base_repo=base,
            solution_patch=sol,
            test_patch="   ",
            language="python",
        )
    assert RDUAL_EMPTY_TEST_PATCH in excinfo.value.reason_codes


# ---------------------------------------------------------------------------
# Flake dual-run rejects
# ---------------------------------------------------------------------------


def test_flake_dual_run_rejects_disagreement() -> None:
    gold_a = SuiteOutcome.from_summary(language="python", passed=("f2p", "p2p"), failed=())
    gold_b = SuiteOutcome.from_summary(
        language="python",
        passed=("f2p",),
        failed=("p2p",),  # disagree
    )
    base = SuiteOutcome.from_summary(language="python", passed=("p2p",), failed=("f2p",))
    with pytest.raises(RealDualRunError) as excinfo:
        label_real_pr_from_outcomes(
            language="python",
            base_outcome=base,
            gold_outcome=gold_a,
            base_commit=BASE_SHA,
            dual_gold_outcomes=(gold_a, gold_b),
            dual_base_outcomes=(base, base),
        )
    assert LABEL_FLAKE_REJECT in excinfo.value.reason_codes or LABEL_G2_FLAKE in (
        excinfo.value.reason_codes
    )


def test_flake_reject_helper() -> None:
    r1 = SuiteOutcome.from_summary(language="go", passed=("T1",), failed=("T2",))
    r2 = SuiteOutcome.from_summary(language="go", passed=("T1", "T2"), failed=())
    with pytest.raises(RealDualRunError) as excinfo:
        flake_reject_on_disagreement([r1, r2], phase="gold")
    assert LABEL_FLAKE_REJECT in excinfo.value.reason_codes


def test_stable_dual_outcome_accepts() -> None:
    base, gold = _pair()
    result = label_real_pr_from_outcomes(
        language="python",
        base_outcome=base,
        gold_outcome=gold,
        base_commit=BASE_SHA,
        dual_base_outcomes=(base, base),
        dual_gold_outcomes=(gold, gold),
    )
    assert result.accepted is True
    assert len(result.f2p_node_ids) >= 1


# ---------------------------------------------------------------------------
# Multi-lang reporter hooks (offline) — expectedBehavior
# ---------------------------------------------------------------------------


def test_multi_lang_go_reporter_offline_dual_run(tmp_path: Path) -> None:
    go_base_log = "--- PASS: TestStoreSetGet (0.00s)\n--- FAIL: TestRouterUpsert (0.00s)\n"
    go_gold_log = "--- PASS: TestStoreSetGet (0.00s)\n--- PASS: TestRouterUpsert (0.00s)\n"
    base = parse_with_reporter("go", go_base_log, returncode=1)
    gold = parse_with_reporter("go", go_gold_log, returncode=0)
    result = label_real_pr_from_outcomes(
        language="go",
        base_outcome=base,
        gold_outcome=gold,
        base_commit=BASE_SHA,
        config_dest=tmp_path / "config.json",
    )
    assert result.suite_command.startswith("go test")
    assert result.reporter["tool_label"] == "go-test"
    assert result.f2p_node_ids == ("TestRouterUpsert",)
    assert result.p2p_node_ids == ("TestStoreSetGet",)
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"] == ["TestRouterUpsert"]
    assert cfg["grade"]["tool_label"] == "go-test"


def test_multi_lang_jest_reporter_offline_dual_run(tmp_path: Path) -> None:
    base_log = "  ✓ catalog add and get\n  ✕ catalog findByTag\n"
    gold_log = "  ✓ catalog add and get\n  ✓ catalog findByTag\n"
    base = parse_with_reporter("typescript", base_log, returncode=1)
    gold = parse_with_reporter("typescript", gold_log, returncode=0)
    result = label_real_pr_from_outcomes(
        language="typescript",
        base_outcome=base,
        gold_outcome=gold,
        base_commit=BASE_SHA,
        config_dest=tmp_path / "config.json",
    )
    assert "npm test" in result.suite_command or "jest" in result.reporter["tool_label"]
    assert "catalog findByTag" in result.f2p_node_ids
    assert "catalog add and get" in result.p2p_node_ids
    assert result.reporter["tool_label"] == "jest"
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["suite_reporter"]["reporter_id"] == "ts_jest_v1"


def test_multi_lang_js_and_rust_offline() -> None:
    js_base = parse_with_reporter(
        "javascript",
        "  ✓ keeps working\n  ✕ new feature\n",
        returncode=1,
    )
    js_gold = parse_with_reporter(
        "javascript",
        "  ✓ keeps working\n  ✓ new feature\n",
        returncode=0,
    )
    js = label_real_pr_from_outcomes(
        language="javascript",
        base_outcome=js_base,
        gold_outcome=js_gold,
        base_commit=BASE_SHA,
    )
    assert js.reporter["reporter_id"] == "js_npm_v1"
    assert "new feature" in js.f2p_node_ids

    rust_base = parse_with_reporter(
        "rust",
        "test crate::util::it_works ... ok\ntest crate::util::broken_feature ... FAILED\n",
        returncode=1,
    )
    rust_gold = parse_with_reporter(
        "rust",
        "test crate::util::it_works ... ok\ntest crate::util::broken_feature ... ok\n",
        returncode=0,
    )
    rust = label_real_pr_from_outcomes(
        language="rust",
        base_outcome=rust_base,
        gold_outcome=rust_gold,
        base_commit=BASE_SHA,
    )
    assert rust.suite_command.startswith("cargo test")
    assert rust.f2p_node_ids == ("crate::util::broken_feature",)
    assert rust.p2p_node_ids == ("crate::util::it_works",)


# ---------------------------------------------------------------------------
# Live python dual-run (real pytest reporter) on synthetic workspace
# ---------------------------------------------------------------------------


def _write_py_real_pr_fixture(root: Path) -> tuple[str, str]:
    """Base workspace with broken product; gold solution + held-out test return F2P≥1.

    Uses exact on-disk content so pure/git apply and pytest node collection agree.
    Gold fixes ``scale`` (returns wrong on base); held-out adds ``test_scale_extra``.
    """
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    core_body = "def add(a, b):\n    return a + b\n\n\ndef scale(x):\n    return x\n"
    (root / "pkg" / "core.py").write_text(core_body, encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_core.py").write_text(
        "from pkg.core import add, scale\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_scale():\n    assert scale(3) == 6\n",
        encoding="utf-8",
    )
    # Exact unified diff against core_body (6 lines).
    solution = (
        "diff --git a/pkg/core.py b/pkg/core.py\n"
        "--- a/pkg/core.py\n"
        "+++ b/pkg/core.py\n"
        "@@ -3,4 +3,4 @@\n"
        " \n"
        " \n"
        " def scale(x):\n"
        "-    return x\n"
        "+    return x * 2\n"
    )
    test_patch = (
        "diff --git a/tests/test_scale_extra.py b/tests/test_scale_extra.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_scale_extra.py\n"
        "@@ -0,0 +1,5 @@\n"
        "+from pkg.core import scale\n"
        "+\n"
        "+\n"
        "+def test_scale_extra():\n"
        "+    assert scale(4) == 8\n"
    )
    return solution, test_patch


def test_live_python_real_suite_dual_run(tmp_path: Path) -> None:
    """VAL-RDUAL-001/002 live: real pytest on base vs gold with held-out test.patch."""
    base = tmp_path / "base"
    base.mkdir()
    solution, test_patch = _write_py_real_pr_fixture(base)
    result = label_real_pr_dual_run(
        language="python",
        base_repo=base,
        solution_patch=solution,
        test_patch=test_patch,
        base_commit=BASE_SHA,
        config_dest=tmp_path / "tests" / "config.json",
        agent_context=base,
        dual_runs=1,
    )
    assert result.accepted is True
    assert len(result.f2p_node_ids) >= 1
    # F2P includes basefailing product tests and/or held-out scale extras.
    f2p_blob = " ".join(result.f2p_node_ids)
    assert "test_scale" in f2p_blob
    # P2P regression tests still pass both (add stays green)
    assert any("test_add" in n for n in result.p2p_node_ids)
    assert set(result.f2p_node_ids).isdisjoint(result.p2p_node_ids)
    assert "pytest" in result.suite_command
    assert result.test_patch_applied is True
    assert_verifier_prepare_applies_test_patch(result.apply_log)
    # Agent tree still lacks held-out test
    assert not (base / "tests" / "test_scale_extra.py").exists()
    cfg = json.loads((tmp_path / "tests" / "config.json").read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"]
    assert cfg["suite_command"]
    assert cfg["label_method"].startswith("real_pr_dual_run")


def test_live_python_dual_runs_flake_stable(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    solution, test_patch = _write_py_real_pr_fixture(base)
    result = label_real_pr_dual_run(
        language="python",
        base_repo=base,
        solution_patch=solution,
        test_patch=test_patch,
        base_commit=BASE_SHA,
        dual_runs=2,
    )
    assert result.accepted is True
    assert len(result.f2p_node_ids) >= 1


def test_apply_unified_patch_create_file(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    patch = textwrap.dedent(
        """\
        diff --git a/hello.txt b/hello.txt
        new file mode 100644
        --- /dev/null
        +++ b/hello.txt
        @@ -0,0 +1,2 @@
        +hi
        +there
        """
    )
    paths = apply_unified_patch(ws, patch, label="test.patch")
    assert paths == ["hello.txt"]
    assert (ws / "hello.txt").read_text(encoding="utf-8") == "hi\nthere\n"


def test_rewrite_legacy_pytest_ignore_collect_path_arg(tmp_path):
    """Legacy boltons-style conf hooks must soft-fix under pytest 8+ dual-run."""
    from swe_factory.producers.harbor_labeling import _rewrite_legacy_pytest_conf_hooks

    conf = tmp_path / "tests" / "conftest.py"
    conf.parent.mkdir(parents=True)
    conf.write_text(
        "def pytest_ignore_collect(path, config):\n"
        "    if path.basename() == '_benchmarks':\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    n = _rewrite_legacy_pytest_conf_hooks(tmp_path)
    assert n == 1
    body = conf.read_text(encoding="utf-8")
    assert "collection_path" in body
    assert "def pytest_ignore_collect(path, config)" not in body


def test_rewrite_legacy_pytest_basename_attr_without_call(tmp_path):
    """Boltons uses path.basename attribute (not basename()) — must rewrite."""
    from swe_factory.producers.harbor_labeling import _rewrite_legacy_pytest_conf_hooks

    conf = tmp_path / "tests" / "conftest.py"
    conf.parent.mkdir(parents=True)
    conf.write_text(
        "import sys\n"
        "import re\n"
        "_VERSION_MARKER = re.compile(r'_py(?P<major_version>\\d)')\n"
        "\n"
        "def pytest_ignore_collect(path, config):\n"
        "    filename = path.basename\n"
        "    modulename = filename.split('.', 1)[0]\n"
        "    match = _VERSION_MARKER.search(modulename)\n"
        "    if not match:\n"
        "        return False\n"
        "    return int(match.group('major_version')) != sys.version_info[0]\n",
        encoding="utf-8",
    )
    n = _rewrite_legacy_pytest_conf_hooks(tmp_path)
    assert n == 1
    body = conf.read_text(encoding="utf-8")
    assert "def pytest_ignore_collect(collection_path, config)" in body
    assert "path.basename" not in body or "collection_path" in body
    # Attribute access must be expanded to dual-compatible expression.
    assert "getattr(collection_path, 'name'" in body or "collection_path.basename()" in body
    # Ensure rewritten body is importable under pathlib Path semantics.
    namespace: dict[str, object] = {}
    exec(compile(body, str(conf), "exec"), namespace)
    hook = namespace["pytest_ignore_collect"]

    class _Cfg:
        pass

    assert hook(Path("tests/test_x.py"), _Cfg()) is False
    assert hook(Path("tests/test_x_py2.py"), _Cfg()) is True
