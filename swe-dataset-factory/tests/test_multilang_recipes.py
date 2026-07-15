"""Multi-lang agent recipes + reporter smoke + held-out isolation (M14 VAL-MLANG / VAL-LX-008).

Covers:
- VAL-MLANG-001: agent recipes for python/go/javascript/typescript/rust (base images)
- VAL-MLANG-002: rust recipe + cargo dual-run path (parser + base image, not python fallthrough)
- VAL-LHARD-002: suite reporter detectability including rust
- VAL-LX-008: agent image isolates held-out tests via test.patch only
- Path language bias: ``.rs`` → rust (pr_miner language_from_paths)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swe_factory.envbuild.agent_recipe import (
    ALLOW_INTERNET_FALSE,
    SUPPORTED_RECIPE_LANGUAGES,
    agent_dockerfile_bakes_held_out_tests,
    agent_recipe_isolates_held_out_tests,
    base_image_for_language,
    default_baseline_test_command,
    default_install_commands,
    normalize_recipe_language,
    render_agent_dockerfile,
    render_real_pr_agent_dockerfile,
)
from swe_factory.envbuild.fixture import (
    language_recipe_table,
    recipe_for_language,
    recipe_from_clone,
    recipe_from_rust_defaults,
)
from swe_factory.envbuild.sha import isolation_scan
from swe_factory.producers.hard_filter import suite_reporter_detectable
from swe_factory.producers.pr_miner import PrFileChange, _language_from_paths
from swe_factory.producers.suite_reporters import (
    RustSuiteReporter,
    get_suite_reporter,
    list_reporter_languages,
    parse_with_reporter,
    reporter_info,
    reporter_registry_snapshot,
)

# ---------------------------------------------------------------------------
# VAL-MLANG-001: five languages + correct base images
# ---------------------------------------------------------------------------


def test_supported_recipe_languages_cover_five() -> None:
    assert set(SUPPORTED_RECIPE_LANGUAGES) == {
        "python",
        "go",
        "javascript",
        "typescript",
        "rust",
    }


def test_base_image_for_each_language_not_python_fallthrough() -> None:
    assert base_image_for_language("python").startswith("python:")
    assert "golang" in base_image_for_language("go")
    assert base_image_for_language("javascript").startswith("node:")
    assert base_image_for_language("typescript").startswith("node:")
    rust_img = base_image_for_language("rust")
    assert "rust" in rust_img
    assert "python" not in rust_img
    # aliases
    assert base_image_for_language("rs") == rust_img
    assert base_image_for_language("golang") == base_image_for_language("go")
    assert base_image_for_language("js") == base_image_for_language("javascript")


def test_language_recipe_table_has_install_and_baseline() -> None:
    table = language_recipe_table()
    assert set(table) == set(SUPPORTED_RECIPE_LANGUAGES)
    assert "cargo" in str(table["rust"]["baseline_test_command"]).lower()
    assert "rust" in str(table["rust"]["base_image"]).lower()
    for lang, row in table.items():
        assert row["language"] == lang
        assert row["base_image"]
        assert row["install_commands"] is not None
        assert row["baseline_test_command"]


@pytest.mark.parametrize(
    "language,expect_base_fragment",
    [
        ("python", "python:"),
        ("go", "golang"),
        ("javascript", "node:"),
        ("typescript", "node:"),
        ("rust", "rust:"),
    ],
)
def test_render_agent_dockerfile_language_base(language: str, expect_base_fragment: str) -> None:
    df = render_agent_dockerfile(base_commit="a" * 40, language=language)
    assert expect_base_fragment in df
    assert ALLOW_INTERNET_FALSE in df
    assert f'swe_factory.language="{normalize_recipe_language(language)}"' in df
    # No silent python slim for non-python languages
    if language != "python":
        assert "FROM python:3.12-slim" not in df


def test_rust_dockerfile_uses_rust_base_not_python() -> None:
    df = render_agent_dockerfile(base_commit="b" * 40, language="rust")
    assert "FROM rust:" in df
    assert "FROM python:" not in df
    assert "cargo" in df.lower() or "Cargo.toml" in df
    assert agent_recipe_isolates_held_out_tests(df) is True


def test_real_pr_rust_clone_dockerfile() -> None:
    df = render_real_pr_agent_dockerfile(
        repository_url="https://github.com/example/rcool.git",
        base_commit="c" * 40,
        language="rust",
    )
    assert "git clone" in df
    assert "FROM rust:" in df
    assert "COPY repo/" not in df
    assert ALLOW_INTERNET_FALSE in df


def test_recipe_from_rust_defaults_and_dispatch() -> None:
    rec = recipe_from_rust_defaults()
    assert rec.language == "rust"
    assert "rust" in rec.base_image
    assert "cargo" in rec.baseline_test_command.lower()
    assert recipe_for_language("rust").language == "rust"
    assert recipe_for_language("rs").language == "rust"
    clone = recipe_from_clone(
        repo_id="tokio-rs/bytes",
        base_commit="d" * 40,
        language="rust",
        clone_url="https://github.com/tokio-rs/bytes.git",
    )
    assert clone.language == "rust"
    assert "rust" in clone.base_image
    assert clone.require_real_sha is True


def test_default_install_and_baseline_commands() -> None:
    assert any("pytest" in c for c in default_install_commands("python"))
    assert default_baseline_test_command("go").startswith("go test")
    assert "cargo test" in default_baseline_test_command("rust")
    assert "npm" in " ".join(default_install_commands("javascript"))
    # Modern trybuild/serde cargo trees need a current stable (edition2024 gate).
    assert "1.88" in base_image_for_language("rust") or "1.8" in base_image_for_language("rust")
    assert any("cargo fetch" in c for c in default_install_commands("rust"))


# ---------------------------------------------------------------------------
# VAL-MLANG-001 / path bias: .rs → rust
# ---------------------------------------------------------------------------


def test_language_from_paths_rust_rs_bias() -> None:
    files = [
        PrFileChange(path="src/lib.rs", status="modified", patch="@@ -1 +1 @@\n"),
        PrFileChange(path="src/util.rs", status="modified", patch="@@ -1 +1 @@\n"),
        PrFileChange(path="tests/smoke.rs", status="added", patch="@@ -0,0 +1 @@\n"),
    ]
    assert _language_from_paths(files) == "rust"


def test_language_from_paths_prefers_dominant_rust_over_markdown() -> None:
    files = [
        PrFileChange(path="src/a.rs", status="modified", patch="@@\n"),
        PrFileChange(path="src/b.rs", status="modified", patch="@@\n"),
        PrFileChange(path="README.md", status="modified", patch="@@\n"),
    ]
    assert _language_from_paths(files) == "rust"


def test_language_from_paths_python_still_works() -> None:
    files = [
        PrFileChange(path="pkg/mod.py", status="modified", patch="@@\n"),
        PrFileChange(path="tests/test_mod.py", status="modified", patch="@@\n"),
    ]
    assert _language_from_paths(files) == "python"


# ---------------------------------------------------------------------------
# Dual-run suite reporters: real node IDs per language
# ---------------------------------------------------------------------------


def test_reporter_languages_include_rust() -> None:
    langs = set(list_reporter_languages())
    assert langs >= set(SUPPORTED_RECIPE_LANGUAGES)
    snap = reporter_registry_snapshot()
    assert snap["rust"]["tool_label"] == "cargo-test"
    assert snap["rust"]["reporter_id"] == "rust_cargo_v1"


@pytest.mark.parametrize(
    "language,log,expect_pass,expect_fail",
    [
        (
            "python",
            (
                '{"rc": 1, "passed": ["tests.a.test_ok"], '
                '"failed": ["tests.a.test_f2p"], "errors": []}'
            ),
            ("tests.a.test_ok",),
            ("tests.a.test_f2p",),
        ),
        (
            "go",
            "--- PASS: TestStore (0.00s)\n--- FAIL: TestBroken (0.00s)\n",
            ("TestStore",),
            ("TestBroken",),
        ),
        (
            "typescript",
            "  ✓ catalog add\n  ✕ catalog find\n",
            ("catalog add",),
            ("catalog find",),
        ),
        (
            "javascript",
            "  ✓ maps bunkers\n  ✕ maps money\n",
            ("maps bunkers",),
            ("maps money",),
        ),
        (
            "rust",
            "test crate::util::it_works ... ok\ntest crate::util::broken ... FAILED\n",
            ("crate::util::it_works",),
            ("crate::util::broken",),
        ),
    ],
)
def test_reporter_smoke_fixtures_produce_real_node_ids(
    language: str,
    log: str,
    expect_pass: tuple[str, ...],
    expect_fail: tuple[str, ...],
) -> None:
    out = parse_with_reporter(language, log, returncode=1)
    for node in expect_pass:
        assert node in out.passed
    for node in expect_fail:
        assert node in out.failed
    # Dual-run needs non-empty real node ids (not synthetic test_always_ok).
    assert out.passed or out.failed
    info = reporter_info(language)
    assert info.language == normalize_recipe_language(language)
    rep = get_suite_reporter(language)
    assert rep.tool_label == info.tool_label


def test_suite_reporter_detectable_for_all_scale_langs() -> None:
    """VAL-LHARD-002: product hard filter can detect suite path per language."""
    for lang in SUPPORTED_RECIPE_LANGUAGES:
        ok, rep_id, cmd = suite_reporter_detectable(lang)
        assert ok is True, lang
        assert rep_id
        assert cmd
        if lang == "rust":
            assert "cargo" in cmd.lower()


def test_rust_reporter_class_parse_only() -> None:
    log = (
        "running 2 tests\n"
        "test foo::bar ... ok\n"
        "test foo::baz ... FAILED\n"
        "test result: FAILED. 1 passed; 1 failed\n"
    )
    rust = RustSuiteReporter().parse_log(log, returncode=101)
    assert rust.language == "rust"
    assert "foo::bar" in rust.passed
    assert "foo::baz" in rust.failed


# ---------------------------------------------------------------------------
# VAL-LX-008: agent isolation — tests only via test.patch
# ---------------------------------------------------------------------------


def test_agent_dockerfile_does_not_copy_test_patch() -> None:
    for lang in SUPPORTED_RECIPE_LANGUAGES:
        df = render_agent_dockerfile(base_commit="e" * 40, language=lang)
        assert agent_dockerfile_bakes_held_out_tests(df) is False
        assert agent_recipe_isolates_held_out_tests(df) is True
        # Fail-closed RUN asserts absence of durable test.patch
        assert "test ! -f test.patch" in df or "test.patch" in df


def test_agent_dockerfile_bake_detector_flags_bad_copy() -> None:
    dirty = "FROM rust:1.78-bookworm\nWORKDIR /app\nCOPY tests/test.patch /app/test.patch\n"
    assert agent_dockerfile_bakes_held_out_tests(dirty) is True
    assert agent_recipe_isolates_held_out_tests(dirty) is False


def test_isolation_scan_flags_test_patch_in_agent_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "lib.rs").write_text("fn main() {}\n", encoding="utf-8")
    clean = isolation_scan(workspace)
    assert clean["clean"] is True
    # Held-out test.patch must not appear in agent root
    (workspace / "test.patch").write_text("diff --git a\n", encoding="utf-8")
    dirty = isolation_scan(workspace)
    assert dirty["clean"] is False
    hits = dirty.get("hits") or []
    assert isinstance(hits, list)
    assert any("test.patch" in str(h) for h in hits)
