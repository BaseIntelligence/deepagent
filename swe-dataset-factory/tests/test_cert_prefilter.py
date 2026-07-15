"""TDD for host cert prefilter (patch apply + collector dry-run F2P potential)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from swe_factory.pipeline.cert_prefilter import (
    REASON_COLLECT_EMPTY,
    REASON_NO_TEST_SURFACE,
    REASON_OK,
    REASON_PATCH_APPLY_FAIL,
    REASON_SKIP_INCOMPLETE,
    PrefilterResult,
    check_patch_apply_at_base,
    collector_dry_run_python,
    has_held_out_test_surface,
    prefilter_material,
    summarize_prefilter_ledger,
)


def _init_git_repo(root: Path, files: dict[str, str]) -> str:
    root.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return sha


def test_has_held_out_test_surface_python_defs() -> None:
    assert has_held_out_test_surface(
        "diff --git a/tests/test_x.py b/tests/test_x.py\n"
        "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
        "@@ -0,0 +1,2 @@\n+def test_foo():\n+    assert True\n"
    )
    assert not has_held_out_test_surface("diff --git a/src/a.py b/src/a.py\n")
    assert has_held_out_test_surface("", test_files=["tests/test_mod.py"])


def test_check_patch_apply_ok_and_fail(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    core = "def add(a, b):\n    return a + b\n"
    tests = "from pkg.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    base = _init_git_repo(
        repo,
        {
            "pkg/core.py": core,
            "tests/test_core.py": tests,
        },
    )
    sol = textwrap.dedent(
        """\
        diff --git a/pkg/core.py b/pkg/core.py
        --- a/pkg/core.py
        +++ b/pkg/core.py
        @@ -1,2 +1,2 @@
         def add(a, b):
        -    return a + b
        +    return a + b + 0
        """
    )
    test_p = textwrap.dedent(
        """\
        diff --git a/tests/test_core.py b/tests/test_core.py
        --- a/tests/test_core.py
        +++ b/tests/test_core.py
        @@ -1,4 +1,7 @@
         from pkg.core import add
         
         def test_add():
             assert add(1, 2) == 3
        +
        +def test_new():
        +    assert add(2, 2) == 4
        """
    )
    ok, detail = check_patch_apply_at_base(
        repo, base_commit=base, solution_patch=sol, test_patch=test_p
    )
    assert ok, detail
    assert detail == "apply_ok"

    bad_sol = textwrap.dedent(
        """\
        diff --git a/pkg/missing.py b/pkg/missing.py
        --- a/pkg/missing.py
        +++ b/pkg/missing.py
        @@ -1 +1 @@
        -old
        +new
        """
    )
    ok2, detail2 = check_patch_apply_at_base(
        repo, base_commit=base, solution_patch=bad_sol, test_patch=test_p
    )
    assert not ok2
    assert "apply-check failed" in detail2


def test_collector_dry_run_collects_held_out(tmp_path: Path) -> None:
    repo = tmp_path / "pkg"
    core = "def add(a, b):\n    return a + b\n"
    tests = "from pkg.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    _init_git_repo(
        repo,
        {
            "pkg/__init__.py": "",
            "pkg/core.py": core,
            "tests/test_core.py": tests,
        },
    )
    count, detail, nodes = collector_dry_run_python(
        repo, test_files=["tests/test_core.py"], apply_test_patch=False
    )
    assert count >= 1, detail
    assert any("test_add" in n for n in nodes)


def test_prefilter_rejects_no_test_surface() -> None:
    result = prefilter_material(
        task_id="x",
        repository_url="https://github.com/example/demo.git",
        base_commit="a" * 40,
        language="python",
        solution_patch="diff --git a/a.py b/a.py\n",
        test_patch="diff --git a/README.md b/README.md\n",
        test_files=(),
        run_collect=False,
    )
    assert not result.ok
    assert result.reason_code == REASON_NO_TEST_SURFACE


def test_prefilter_fail_open_incomplete_identity() -> None:
    result = prefilter_material(
        task_id="x",
        repository_url="",
        base_commit="short",
        language="python",
        solution_patch="",
        test_patch="def test_x():\n    assert True\n",
        test_files=["tests/test_x.py"],
        run_collect=False,
    )
    assert result.ok
    assert result.reason_code == REASON_SKIP_INCOMPLETE


def test_prefilter_material_apply_and_collect(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    core = "def add(a, b):\n    return a - b\n"
    tests = "from pkg.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    base = _init_git_repo(
        repo,
        {
            "pkg/__init__.py": "",
            "pkg/core.py": core,
            "tests/test_core.py": tests,
        },
    )
    sol = textwrap.dedent(
        """\
        diff --git a/pkg/core.py b/pkg/core.py
        --- a/pkg/core.py
        +++ b/pkg/core.py
        @@ -1,2 +1,2 @@
         def add(a, b):
        -    return a - b
        +    return a + b
        """
    )
    test_p = textwrap.dedent(
        """\
        diff --git a/tests/test_core.py b/tests/test_core.py
        --- a/tests/test_core.py
        +++ b/tests/test_core.py
        @@ -1,4 +1,7 @@
         from pkg.core import add
         
         def test_add():
             assert add(1, 2) == 3
        +
        +def test_extra():
        +    assert add(0, 0) == 0
        """
    )

    def materialize(**kwargs: object) -> Path:
        del kwargs
        return repo

    result = prefilter_material(
        task_id="demo",
        repository_url="https://github.com/example/demo.git",
        base_commit=base,
        language="python",
        solution_patch=sol,
        test_patch=test_p,
        test_files=["tests/test_core.py"],
        materialize_repo=materialize,
        run_collect=True,
    )
    assert result.ok, result.to_dict()
    assert result.reason_code == REASON_OK
    assert result.collected >= 1


def test_prefilter_rejects_bad_solution_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base = _init_git_repo(repo, {"pkg/core.py": "x = 1\n"})
    bad = textwrap.dedent(
        """\
        diff --git a/missing.py b/missing.py
        --- a/missing.py
        +++ b/missing.py
        @@ -1 +1 @@
        -a
        +b
        """
    )
    test_ok = textwrap.dedent(
        """\
        diff --git a/tests/test_x.py b/tests/test_x.py
        new file mode 100644
        --- /dev/null
        +++ b/tests/test_x.py
        @@ -0,0 +1,2 @@
        +def test_x():
        +    assert True
        """
    )

    def materialize(**kwargs: object) -> Path:
        del kwargs
        return repo

    result = prefilter_material(
        task_id="bad",
        repository_url="https://github.com/example/demo.git",
        base_commit=base,
        language="python",
        solution_patch=bad,
        test_patch=test_ok,
        test_files=["tests/test_x.py"],
        materialize_repo=materialize,
        run_collect=False,
    )
    assert not result.ok
    assert result.reason_code == REASON_PATCH_APPLY_FAIL


def test_summarize_prefilter_ledger() -> None:
    summary = summarize_prefilter_ledger(
        [
            {"reason_code": REASON_PATCH_APPLY_FAIL},
            {"reason_code": REASON_COLLECT_EMPTY},
            {"reason_code": REASON_PATCH_APPLY_FAIL},
        ]
    )
    assert summary["total"] == 3
    assert summary["by_reason_code"][REASON_PATCH_APPLY_FAIL] == 2
    assert summary["by_reason_code"][REASON_COLLECT_EMPTY] == 1


def test_prefilter_result_to_dict() -> None:
    r = PrefilterResult(ok=True, reason_code=REASON_OK, detail="x", collected=3)
    d = r.to_dict()
    assert d["ok"] is True
    assert d["collected"] == 3


def test_base_fail_f2p_heuristic_detects_all_pass(tmp_path: Path) -> None:
    from swe_factory.pipeline.cert_prefilter import (
        REASON_EMPTY_F2P_HEURISTIC,
        base_fail_f2p_heuristic,
        prefilter_material,
    )

    repo = tmp_path / "repo"
    base = _init_git_repo(
        repo,
        {
            "pkg/__init__.py": "",
            "pkg/core.py": "def add(a, b):\n    return a + b\n",
            "tests/test_core.py": (
                "from pkg.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
            ),
        },
    )
    # Test already green at base; held-out adds nothing that fails@base.
    sol = textwrap.dedent(
        """\
        diff --git a/pkg/core.py b/pkg/core.py
        --- a/pkg/core.py
        +++ b/pkg/core.py
        @@ -1,2 +1,3 @@
         def add(a, b):
             return a + b
        +# no behavior change
        """
    )
    test_p = textwrap.dedent(
        """\
        diff --git a/tests/test_core.py b/tests/test_core.py
        --- a/tests/test_core.py
        +++ b/tests/test_core.py
        @@ -1,4 +1,7 @@
         from pkg.core import add
         
         def test_add():
             assert add(1, 2) == 3
        +
        +def test_also_green():
        +    assert add(0, 0) == 0
        """
    )

    def materialize(**kwargs: object) -> Path:
        del kwargs
        return repo

    # apply_ok then collect + base-fail heuristic should skip empty F2P
    result = prefilter_material(
        task_id="green-base",
        repository_url="https://github.com/example/demo.git",
        base_commit=base,
        language="python",
        solution_patch=sol,
        test_patch=test_p,
        test_files=["tests/test_core.py"],
        materialize_repo=materialize,
        run_collect=True,
    )
    assert not result.ok
    assert result.reason_code == REASON_EMPTY_F2P_HEURISTIC

    ok, detail, failed = base_fail_f2p_heuristic(repo, test_files=["tests/test_core.py"])
    # During prefilter, test.patch is applied; after that call repo still has it.
    # Direct heuristic with already green suite may fail-open.
    assert isinstance(detail, str)
    assert isinstance(failed, list)
