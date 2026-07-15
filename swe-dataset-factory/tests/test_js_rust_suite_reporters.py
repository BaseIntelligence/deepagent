"""VAL-MLANG-003: ava/tape + cargo dual-run reporters produce real node IDs.

Product N is currently python-only because JS suites (ava/tape) and cargo
result parsing/runtime PATH did not feed non-empty green/F2P node ids.
These offline parser + install-PATH unit tests lock the fixes without
network or fixture product pad.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from swe_factory.producers.harbor_labeling import (
    detect_js_test_framework,
    ensure_local_node_bin_on_path,
    run_language_suite,
)
from swe_factory.producers.real_dual_run import (
    labels_from_real_suite_outcomes,
    suite_command_for,
)
from swe_factory.producers.suite_reporters import (
    RustSuiteReporter,
    TypeScriptSuiteReporter,
    get_suite_reporter,
    parse_with_reporter,
    reporter_info,
)

AVA_LOG = """\
  ✔ return ANSI escape codes
  ✔ group related codes into categories
  ✖ support conversion to ansi (16 colors)

  2 tests passed
  1 test failed
"""

TAPE_LOG = """\
TAP version 13
# parse()
ok 1 parses a single nested string
ok 2 parses a double nested string
# stringify()
ok 3 stringifies a querystring object
not ok 4 refuses invalid allowEmptyArrays
  ---
    operator: equal
  ...

1..4
# tests 4
# pass  3
# fail  1
"""

JEST_JSON = json.dumps(
    {
        "testResults": [
            {
                "assertionResults": [
                    {"title": "maps money", "status": "passed"},
                    {"title": "maps bunkers", "status": "failed"},
                ]
            }
        ]
    }
)

CARGO_LOG = """\
running 3 tests
test tests::integration::it_works ... ok
test tests::integration::broken_feature ... FAILED
test src/lib.rs - (line 42) ... ok
test src/macros.rs - macros::log (line 16) ... ok
test result: FAILED. 3 passed; 1 failed; 0 ignored
"""

CARGO_IGNORED = """\
running 2 tests
test ui ... ignored, requires nightly
test test_brace_escape ... ok
test result: ok. 1 passed; 0 failed; 1 ignored
"""


def test_ava_parse_produces_real_node_ids() -> None:
    out = parse_with_reporter("javascript", AVA_LOG, returncode=1)
    assert "return ANSI escape codes" in out.passed
    assert "group related codes into categories" in out.passed
    assert "support conversion to ansi (16 colors)" in out.failed
    assert out.passed and out.failed
    # Dual-run F2P/P2P shapes from real node ids
    green = parse_with_reporter(
        "javascript",
        AVA_LOG.replace("✖", "✔").replace("1 test failed", "0 test failed"),
        returncode=0,
    )
    labels = labels_from_real_suite_outcomes(green, out)
    assert "support conversion to ansi (16 colors)" in labels.f2p_node_ids
    assert "return ANSI escape codes" in labels.p2p_node_ids


def test_tape_parse_produces_real_node_ids() -> None:
    out = parse_with_reporter("javascript", TAPE_LOG, returncode=1)
    assert "parses a single nested string" in out.passed
    assert "stringifies a querystring object" in out.passed
    assert "refuses invalid allowEmptyArrays" in out.failed
    assert out.passed and out.failed
    green = parse_with_reporter(
        "javascript",
        TAPE_LOG.replace(
            "not ok 4 refuses invalid allowEmptyArrays", "ok 4 refuses invalid allowEmptyArrays"
        ),
        returncode=0,
    )
    labels = labels_from_real_suite_outcomes(green, out)
    assert "refuses invalid allowEmptyArrays" in labels.f2p_node_ids
    assert "parses a single nested string" in labels.p2p_node_ids


def test_jest_json_still_parsed() -> None:
    out = TypeScriptSuiteReporter(as_javascript=True).parse_log(JEST_JSON, returncode=1)
    assert "maps money" in out.passed
    assert "maps bunkers" in out.failed


def test_cargo_parse_handles_doctest_spaces_and_ignored() -> None:
    rust = RustSuiteReporter().parse_log(CARGO_LOG, returncode=101)
    assert "tests::integration::it_works" in rust.passed
    assert "tests::integration::broken_feature" in rust.failed
    assert any("src/lib.rs" in n for n in rust.passed)
    assert any("macros::log" in n for n in rust.passed)
    # "test result:" summary lines must not pollute node ids
    assert all("result:" not in n for n in rust.passed + rust.failed)
    ignored = RustSuiteReporter().parse_log(CARGO_IGNORED, returncode=0)
    assert "test_brace_escape" in ignored.passed
    assert "ui" not in ignored.passed
    assert ignored.passed  # non-empty green


def test_cargo_f2p_p2p_from_parser() -> None:
    broken = parse_with_reporter("rust", CARGO_LOG, returncode=101)
    green_text = CARGO_LOG.replace("FAILED", "ok").replace("1 failed", "0 failed")
    green = parse_with_reporter("rust", green_text, returncode=0)
    labels = labels_from_real_suite_outcomes(green, broken)
    assert "tests::integration::broken_feature" in labels.f2p_node_ids
    assert "tests::integration::it_works" in labels.p2p_node_ids
    assert labels.f2p_node_ids and labels.p2p_node_ids


def test_cargo_trybuild_error_status_parsed_as_failed() -> None:
    """trybuild compile-pass emits ``... error``; must be F2P-eligible failed."""
    text = """\
running 2 tests
test pass ... FAILED
test tests/compile-pass/foo.rs ... error
test result: FAILED. 0 passed; 1 failed; 0 ignored
"""
    out = RustSuiteReporter().parse_log(text, returncode=101)
    assert "pass" in out.failed
    assert "tests/compile-pass/foo.rs" in out.failed
    green = RustSuiteReporter().parse_log(
        text.replace("FAILED", "ok").replace("error", "ok").replace("1 failed", "0 failed"),
        returncode=0,
    )
    labels = labels_from_real_suite_outcomes(green, out)
    assert "pass" in labels.f2p_node_ids
    assert "tests/compile-pass/foo.rs" in labels.f2p_node_ids


def test_detect_js_frameworks(tmp_path: Path) -> None:
    # ava
    ava = tmp_path / "ava_pkg"
    ava.mkdir()
    (ava / "package.json").write_text(
        json.dumps(
            {
                "scripts": {"test": "xo && ava && tsd"},
                "devDependencies": {"ava": "^6.1.3", "xo": "^0.58.0"},
            }
        ),
        encoding="utf-8",
    )
    det = detect_js_test_framework(ava)
    assert det["framework"] == "ava"
    assert "ava" in det["command"]

    # tape / nyc tape
    tape = tmp_path / "tape_pkg"
    tape.mkdir()
    (tape / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "test": "npm run tests-only",
                    "tests-only": "nyc tape 'test/**/*.js'",
                },
                "devDependencies": {"tape": "^5.10.2", "nyc": "^10.3.2"},
            }
        ),
        encoding="utf-8",
    )
    det_t = detect_js_test_framework(tape)
    assert det_t["framework"] == "tape"
    assert "tape" in det_t["command"] or "tests-only" in det_t["command"]

    # jest
    jest = tmp_path / "jest_pkg"
    jest.mkdir()
    (jest / "package.json").write_text(
        json.dumps(
            {
                "scripts": {"test": "jest"},
                "devDependencies": {"jest": "^29.0.0"},
            }
        ),
        encoding="utf-8",
    )
    det_j = detect_js_test_framework(jest)
    assert det_j["framework"] == "jest"


def test_local_node_bin_on_path(tmp_path: Path) -> None:
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    marker = bin_dir / "ava"
    marker.write_text("#!/bin/sh\necho ava\n", encoding="utf-8")
    marker.chmod(0o755)
    env = ensure_local_node_bin_on_path(tmp_path, env=os.environ.copy())
    parts = env["PATH"].split(os.pathsep)
    assert str(bin_dir.resolve()) == parts[0]
    # Caller env is not required to globally mutate os.environ
    assert str(bin_dir.resolve()) in env["PATH"]


def test_suite_command_prefers_js_not_only_jest() -> None:
    # registry still exposes authentic suite command for dual-run meta
    js_cmd = suite_command_for("javascript")
    assert "npm" in js_cmd.lower() or "ava" in js_cmd.lower() or "tape" in js_cmd.lower()
    rust_cmd = suite_command_for("rust")
    assert "cargo" in rust_cmd.lower()
    info = reporter_info("javascript")
    # tool_label may be framework-generic or jest for motors; id must not be empty
    assert info.reporter_id
    rep = get_suite_reporter("javascript")
    assert rep.language == "javascript"


def test_npm_install_puts_bin_on_path_for_ava_fixture(tmp_path: Path) -> None:
    """Offline mini ava tree: install PATH wiring without requiring jest."""
    pkg = tmp_path / "mini_ava"
    pkg.mkdir()
    (pkg / "package.json").write_text(
        json.dumps(
            {
                "name": "mini-ava",
                "private": True,
                "type": "module",
                "scripts": {"test": "ava"},
                "devDependencies": {"ava": "^6.1.3"},
            }
        ),
        encoding="utf-8",
    )
    (pkg / "test.js").write_text(
        "import test from 'ava';\n"
        "test('alpha', t => { t.pass(); });\n"
        "test('beta', t => { t.pass(); });\n",
        encoding="utf-8",
    )
    # May depend on network for npm install; skip if npm missing or install fails offline
    import shutil
    import subprocess

    if shutil.which("npm") is None:
        pytest.skip("npm not available")
    proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=str(pkg),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if proc.returncode != 0 or not (pkg / "node_modules" / ".bin" / "ava").exists():
        pytest.skip(f"npm install ava unavailable offline: {proc.stderr[-200:]}")
    out = run_language_suite(pkg, "javascript")
    assert out.language in {"javascript", "typescript"}
    assert out.passed, f"expected ava node ids, raw={out.raw_tail[-500:]!r}"
    assert "alpha" in " ".join(out.passed) or any("alpha" in p for p in out.passed)
