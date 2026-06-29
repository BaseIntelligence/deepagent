"""Unit tests for the concrete language adapters (offline, small fixtures).

Covers the m2-adapters contract (VAL-ENV-001..008): mutually-exclusive language
detection, unsupported-repo rejection, the pinned base images, non-empty
ecosystem install commands, selection-aware test commands, the mutation-tool
metadata hook, and test-file classification - across Python, JS/TS, and Go. The
`swe-forge forge detect` / `adapter-info` CLI surface is exercised too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_forge.forge.adapters import (
    GoAdapter,
    JavaScriptAdapter,
    NoAdapterFoundError,
    PythonAdapter,
    build_default_registry,
)
from swe_forge.forge.cli import app as forge_app

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Fixtures: minimal single-language repositories.
# --------------------------------------------------------------------------- #
def _write(root: Path, rel: str, content: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "py"
    _write(repo, "pyproject.toml", "[project]\nname = 'demo'\n")
    _write(repo, "src/demo/__init__.py", "def add(a, b):\n    return a + b\n")
    _write(repo, "tests/test_demo.py", "def test_add():\n    assert True\n")
    return repo


@pytest.fixture
def python_only_sources(tmp_path: Path) -> Path:
    """A Python repo with sources but no manifest (detection via *.py only)."""
    repo = tmp_path / "py_src"
    _write(repo, "main.py", "print('hi')\n")
    return repo


@pytest.fixture
def python_requirements_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "py_req"
    _write(repo, "requirements.txt", "pytest\n")
    _write(repo, "app.py", "x = 1\n")
    return repo


@pytest.fixture
def js_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "js"
    _write(repo, "package.json", '{"name": "demo", "version": "1.0.0"}\n')
    _write(repo, "package-lock.json", "{}\n")
    _write(repo, "index.js", "module.exports = (a, b) => a + b;\n")
    return repo


@pytest.fixture
def js_only_sources(tmp_path: Path) -> Path:
    """Plain JS files, no package.json (resolves to javascript)."""
    repo = tmp_path / "js_src"
    _write(repo, "index.js", "console.log('hi');\n")
    return repo


@pytest.fixture
def ts_repo(tmp_path: Path) -> Path:
    """A TS-bearing repo (tsconfig + *.ts) -> resolves to the javascript adapter."""
    repo = tmp_path / "ts"
    _write(repo, "tsconfig.json", "{}\n")
    _write(
        repo, "src/index.ts", "export const add = (a: number, b: number) => a + b;\n"
    )
    return repo


@pytest.fixture
def go_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "go"
    _write(repo, "go.mod", "module demo\n\ngo 1.22\n")
    _write(repo, "main.go", "package main\n\nfunc Add(a, b int) int { return a + b }\n")
    return repo


@pytest.fixture
def go_only_sources(tmp_path: Path) -> Path:
    repo = tmp_path / "go_src"
    _write(repo, "lib.go", "package lib\n")
    return repo


@pytest.fixture
def unknown_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "unknown"
    _write(repo, "README.md", "# nothing to see\n")
    _write(repo, "data.txt", "plain text\n")
    return repo


# --------------------------------------------------------------------------- #
# VAL-ENV-001 / 002: detection per ecosystem + mutual exclusivity.
# --------------------------------------------------------------------------- #
class TestDetection:
    @pytest.mark.parametrize(
        ("fixture_name", "expected"),
        [
            ("python_repo", "python"),
            ("python_only_sources", "python"),
            ("python_requirements_repo", "python"),
            ("js_repo", "javascript"),
            ("js_only_sources", "javascript"),
            ("ts_repo", "javascript"),
            ("go_repo", "go"),
            ("go_only_sources", "go"),
        ],
    )
    def test_registry_selects_expected_adapter(
        self, request: pytest.FixtureRequest, fixture_name: str, expected: str
    ) -> None:
        repo = request.getfixturevalue(fixture_name)
        registry = build_default_registry()
        assert registry.detect(repo).name == expected

    @pytest.mark.parametrize(
        ("fixture_name", "expected"),
        [
            ("python_repo", "python"),
            ("js_repo", "javascript"),
            ("ts_repo", "javascript"),
            ("go_repo", "go"),
        ],
    )
    def test_detection_is_mutually_exclusive(
        self, request: pytest.FixtureRequest, fixture_name: str, expected: str
    ) -> None:
        repo = request.getfixturevalue(fixture_name)
        registry = build_default_registry()
        table = {adapter.name: adapter.detect(repo) for adapter in registry}
        assert table[expected] is True
        assert [name for name, ok in table.items() if ok] == [expected]

    def test_unknown_repo_is_rejected(self, unknown_repo: Path) -> None:
        registry = build_default_registry()
        assert registry.detect_optional(unknown_repo) is None
        assert registry.detect_all(unknown_repo) == []
        with pytest.raises(NoAdapterFoundError):
            registry.detect(unknown_repo)

    def test_nonexistent_path_matches_nothing(self, tmp_path: Path) -> None:
        registry = build_default_registry()
        assert registry.detect_optional(tmp_path / "does-not-exist") is None


# --------------------------------------------------------------------------- #
# VAL-ENV-004: base images.
# --------------------------------------------------------------------------- #
class TestBaseImage:
    def test_pinned_base_images(self) -> None:
        assert PythonAdapter().base_image() == "python:3.12-slim"
        assert JavaScriptAdapter().base_image() == "node:22-slim"
        assert GoAdapter().base_image() == "golang:1.22"


# --------------------------------------------------------------------------- #
# VAL-ENV-005: install commands are non-empty + ecosystem-appropriate.
# --------------------------------------------------------------------------- #
class TestInstallCommands:
    def test_python_editable_install(self, python_repo: Path) -> None:
        commands = PythonAdapter().install_commands(python_repo)
        assert commands == ["pip install -e ."]

    def test_python_requirements_install(self, python_requirements_repo: Path) -> None:
        commands = PythonAdapter().install_commands(python_requirements_repo)
        assert commands == ["pip install -r requirements.txt"]

    def test_python_sources_only_fallback_nonempty(
        self, python_only_sources: Path
    ) -> None:
        commands = PythonAdapter().install_commands(python_only_sources)
        assert len(commands) >= 1
        assert all(cmd for cmd in commands)

    def test_js_uses_npm_ci_with_lockfile(self, js_repo: Path) -> None:
        assert JavaScriptAdapter().install_commands(js_repo) == ["npm ci"]

    def test_js_without_lockfile_uses_npm_install(self, ts_repo: Path) -> None:
        assert JavaScriptAdapter().install_commands(ts_repo) == ["npm install"]

    def test_go_mod_download(self, go_repo: Path) -> None:
        commands = GoAdapter().install_commands(go_repo)
        assert commands == ["go mod download"]
        assert len(commands) >= 1


# --------------------------------------------------------------------------- #
# VAL-ENV-006: test command + selection honoring.
# --------------------------------------------------------------------------- #
class TestTestCommand:
    def test_python_default_and_selection(self) -> None:
        adapter = PythonAdapter()
        assert adapter.test_command() == "python -m pytest"
        selected = adapter.test_command(["tests/test_demo.py::test_add"])
        assert selected == "python -m pytest tests/test_demo.py::test_add"
        assert "tests/test_demo.py::test_add" in selected

    def test_js_default_and_selection(self) -> None:
        adapter = JavaScriptAdapter()
        assert adapter.test_command() == "node --test"
        selected = adapter.test_command(["index.test.js"])
        assert selected == "node --test index.test.js"

    def test_go_default_and_package_selection(self) -> None:
        adapter = GoAdapter()
        assert adapter.test_command() == "go test ./..."
        selected = adapter.test_command(["./pkg/foo"])
        assert selected == "go test ./pkg/foo"
        assert "./..." not in selected

    def test_go_run_name_selection(self) -> None:
        adapter = GoAdapter()
        selected = adapter.test_command(["TestAdd"])
        assert selected == "go test -run '^(TestAdd)$' ./..."

    def test_go_package_and_name_selection(self) -> None:
        adapter = GoAdapter()
        selected = adapter.test_command(["./pkg/foo", "TestAdd"])
        assert selected == "go test -run '^(TestAdd)$' ./pkg/foo"
        assert "./..." not in selected


# --------------------------------------------------------------------------- #
# parse_test_failures: derive a structural mutation's collateral (per candidate).
# --------------------------------------------------------------------------- #
class TestParseTestFailures:
    def test_python_pytest_failures_reduced_to_k_names(self) -> None:
        adapter = PythonAdapter()
        output = (
            "=== short test summary info ===\n"
            "FAILED tests/test_strutils.py::test_slugify_basic - AssertionError\n"
            "FAILED tests/test_x.py::test_param[case-1] - ValueError\n"
            "FAILED boltons/strutils.py::boltons.strutils.slugify\n"
            "ERROR tests/test_setup.py\n"
        )
        names = adapter.parse_test_failures(output)
        # node ids reduced to bare -k keywords; doctest leaf taken; params stripped;
        # a whole-file collection ERROR (no per-test name) is dropped.
        assert names == ["test_slugify_basic", "test_param", "slugify"]

    def test_python_no_failures_empty(self) -> None:
        adapter = PythonAdapter()
        assert adapter.parse_test_failures("3 passed in 0.1s") == []

    def test_go_fail_lines_reduced_to_top_level_names(self) -> None:
        adapter = GoAdapter()
        output = (
            "--- FAIL: TestEllipsis (0.00s)\n"
            "    string_test.go:10: got x want y\n"
            "--- FAIL: TestRouter/subcase (0.00s)\n"
            "FAIL\n"
        )
        names = adapter.parse_test_failures(output)
        # Subtests reduce to their parent so -skip anchored on it skips the whole.
        assert names == ["TestEllipsis", "TestRouter"]

    def test_javascript_default_is_conservative_empty(self) -> None:
        adapter = JavaScriptAdapter()
        # The JS adapter keeps the conservative base default (no fragile Mocha
        # failure parsing); JS keeps come from pr_mirror, not structural faults.
        assert adapter.parse_test_failures("  1) some failing test:\n") == []


# --------------------------------------------------------------------------- #
# VAL-ENV-007: mutation-tool hook is language-correct and distinct.
# --------------------------------------------------------------------------- #
class TestMutationToolHook:
    def test_per_language_tools(self) -> None:
        assert PythonAdapter().mutation_tool == "mutmut"
        assert JavaScriptAdapter().mutation_tool == "stryker"
        assert GoAdapter().mutation_tool == "go-mutesting"

    def test_tools_are_distinct(self) -> None:
        tools = {
            PythonAdapter().mutation_tool,
            JavaScriptAdapter().mutation_tool,
            GoAdapter().mutation_tool,
        }
        assert len(tools) == 3

    def test_python_lists_cosmic_ray_alternative(self) -> None:
        assert PythonAdapter().mutation_tools == ("mutmut", "cosmic-ray")


# --------------------------------------------------------------------------- #
# VAL-ENV-008: is_test_file classification.
# --------------------------------------------------------------------------- #
class TestIsTestFile:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("test_demo.py", True),
            ("demo_test.py", True),
            ("tests/test_module.py", True),
            ("demo.py", False),
            ("src/module.py", False),
            ("conftest.py", False),
        ],
    )
    def test_python(self, path: str, expected: bool) -> None:
        assert PythonAdapter().is_test_file(path) is expected

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("index.test.js", True),
            ("component.spec.ts", True),
            ("widget.test.tsx", True),
            ("util.spec.jsx", True),
            ("__tests__/foo.js", True),
            ("index.js", False),
            ("src/component.ts", False),
            ("latest.js", False),
        ],
    )
    def test_javascript(self, path: str, expected: bool) -> None:
        assert JavaScriptAdapter().is_test_file(path) is expected

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("main_test.go", True),
            ("pkg/handler_test.go", True),
            ("main.go", False),
            ("pkg/handler.go", False),
        ],
    )
    def test_go(self, path: str, expected: bool) -> None:
        assert GoAdapter().is_test_file(path) is expected


# --------------------------------------------------------------------------- #
# CLI surface: `forge detect` and `forge adapter-info`.
# --------------------------------------------------------------------------- #
class TestDetectCli:
    def test_detect_python_json(self, python_repo: Path) -> None:
        result = runner.invoke(forge_app, ["detect", str(python_repo), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["matched"] is True
        assert payload["language"] == "python"
        assert payload["base_image"] == "python:3.12-slim"
        assert payload["install_commands"] == ["pip install -e ."]
        assert payload["test_command"] == "python -m pytest"
        assert payload["mutation_tool"] == "mutmut"
        assert payload["detection"] == {
            "python": True,
            "javascript": False,
            "go": False,
        }

    def test_detect_ts_resolves_javascript(self, ts_repo: Path) -> None:
        result = runner.invoke(forge_app, ["detect", str(ts_repo), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["language"] == "javascript"
        assert payload["base_image"] == "node:22-slim"

    def test_detect_go_json(self, go_repo: Path) -> None:
        result = runner.invoke(forge_app, ["detect", str(go_repo), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["language"] == "go"
        assert payload["base_image"] == "golang:1.22"
        assert payload["mutation_tool"] == "go-mutesting"

    def test_detect_with_selection(self, python_repo: Path) -> None:
        result = runner.invoke(
            forge_app,
            ["detect", str(python_repo), "--select", "tests/test_demo.py", "--json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["test_command_selection"] == (
            "python -m pytest tests/test_demo.py"
        )

    def test_detect_unknown_rejected_cleanly(self, unknown_repo: Path) -> None:
        result = runner.invoke(forge_app, ["detect", str(unknown_repo), "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["matched"] is False
        assert "unsupported language" in payload["reason"]
        assert payload["detection"] == {
            "python": False,
            "javascript": False,
            "go": False,
        }
        # No traceback leaked to the user.
        assert "Traceback" not in result.output

    def test_detect_missing_path(self, tmp_path: Path) -> None:
        result = runner.invoke(forge_app, ["detect", str(tmp_path / "nope"), "--json"])
        assert result.exit_code == 1
        assert "does not exist" in result.output


class TestAdapterInfoCli:
    def test_lists_all_adapters(self) -> None:
        result = runner.invoke(forge_app, ["adapter-info", "--json"])
        assert result.exit_code == 0
        records = json.loads(result.output)
        by_lang = {r["language"]: r for r in records}
        assert by_lang["python"]["base_image"] == "python:3.12-slim"
        assert by_lang["javascript"]["base_image"] == "node:22-slim"
        assert by_lang["go"]["base_image"] == "golang:1.22"
        assert by_lang["python"]["mutation_tool"] == "mutmut"
        assert by_lang["javascript"]["mutation_tool"] == "stryker"
        assert by_lang["go"]["mutation_tool"] == "go-mutesting"

    def test_single_language_with_selection_and_classify(self) -> None:
        result = runner.invoke(
            forge_app,
            [
                "adapter-info",
                "--language",
                "go",
                "--select",
                "TestAdd",
                "--classify",
                "main_test.go,main.go",
                "--json",
            ],
        )
        assert result.exit_code == 0
        record = json.loads(result.output)
        assert record["language"] == "go"
        assert record["test_command"] == "go test ./..."
        assert record["test_command_selection"] == "go test -run '^(TestAdd)$' ./..."
        assert record["classification"] == {"main_test.go": True, "main.go": False}

    def test_unknown_language_errors(self) -> None:
        result = runner.invoke(
            forge_app, ["adapter-info", "--language", "rust", "--json"]
        )
        assert result.exit_code == 1
        assert "no adapter for language" in result.output
