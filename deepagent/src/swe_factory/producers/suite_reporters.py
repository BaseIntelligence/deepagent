"""Multi-language suite reporters for dual-run F2P/P2P labeling (VAL-LABEL-006).

Each reporter produces structured node outcomes (pass/fail node selectors) that
:mod:`swe_factory.producers.harbor_labeling` consumes into
``f2p_node_ids`` / ``p2p_node_ids`` for ``tests/config.json`` and the verifier
grader whitelist.

Languages:
- python — pytest node ids (``tests.mod.func`` Harbor form)
- go — ``go test`` function names
- typescript / javascript — jest **and** ava/tape titles (VAL-MLANG-003)
- rust — ``cargo test`` filtered names including doctest paths
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable  # Literal used by ReporterLang

from swe_factory.producers.harbor_labeling import (
    HarborLabelError,
    SuiteOutcome,
    run_go_suite,
    run_javascript_suite,
    run_python_suite,
    run_typescript_suite,
)

ReporterLang = Literal["python", "go", "typescript", "javascript", "rust", "ts", "js"]


@runtime_checkable
class SuiteReporter(Protocol):
    """Language suite reporter → structured :class:`SuiteOutcome`."""

    language: str
    tool_label: str
    node_id_form: str

    def run(self, repo: Path) -> SuiteOutcome:
        """Execute the language suite under ``repo`` and return node outcomes."""
        ...

    def parse_log(self, text: str, *, returncode: int = 0) -> SuiteOutcome:
        """Parse a fixed suite log/json into node outcomes (deterministic offline)."""
        ...


@dataclass(frozen=True, slots=True)
class ReporterInfo:
    """Identity metadata used in pack meta / oracle evidence."""

    language: str
    tool_label: str
    node_id_form: str
    reporter_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "language": self.language,
            "tool_label": self.tool_label,
            "node_id_form": self.node_id_form,
            "reporter_id": self.reporter_id,
        }


def _dedupe(seq: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(n for n in seq if n and str(n).strip()))


class PythonSuiteReporter:
    """pytest → Harbor node ids (``tests.mod.fn``)."""

    language = "python"
    tool_label = "pytest"
    node_id_form = "mod.name"
    reporter_id = "python_pytest_v1"

    def run(self, repo: Path) -> SuiteOutcome:
        return run_python_suite(repo)

    def parse_log(self, text: str, *, returncode: int = 0) -> SuiteOutcome:
        """Accept JSON lines with passed/failed arrays or pytest short nodeids."""
        lines = [ln for ln in text.splitlines() if ln.strip().startswith("{")]
        if lines:
            data = json.loads(lines[-1])
            return SuiteOutcome(
                language="python",
                passed=_dedupe([str(x) for x in (data.get("passed") or [])]),
                failed=_dedupe([str(x) for x in (data.get("failed") or [])]),
                errors=_dedupe([str(x) for x in (data.get("errors") or [])]),
                returncode=int(data.get("rc", returncode)),
                raw_tail=text[-1500:],
            )
        # PASS/FAIL style fallback
        passed = re.findall(r"(?:PASSED|✓)\s+(\S+)", text)
        failed = re.findall(r"(?:FAILED|✕)\s+(\S+)", text)
        return SuiteOutcome(
            language="python",
            passed=_dedupe(passed),
            failed=_dedupe(failed),
            returncode=returncode,
            raw_tail=text[-1500:],
        )


class GoSuiteReporter:
    """go test → TestXxx function names."""

    language = "go"
    tool_label = "go-test"
    node_id_form = "name"
    reporter_id = "go_test_v1"

    def run(self, repo: Path) -> SuiteOutcome:
        return run_go_suite(repo)

    def parse_log(self, text: str, *, returncode: int = 0) -> SuiteOutcome:
        passed = list(dict.fromkeys(re.findall(r"--- PASS: (\w+)", text)))
        failed = list(dict.fromkeys(re.findall(r"--- FAIL: (\w+)", text)))
        return SuiteOutcome(
            language="go",
            passed=tuple(passed),
            failed=tuple(failed),
            returncode=returncode,
            raw_tail=text[-1500:],
        )


# Ava / tape / jest-friendly unicode marks and TAP lines
_AVA_PASS_RE = re.compile(r"[✓✔]\s+(.+)")
_AVA_FAIL_RE = re.compile(r"[✕✖×]\s+(.+)")
_TAPE_PASS_RE = re.compile(r"^ok\s+\d+\s+(.+)$", re.MULTILINE)
_TAPE_FAIL_RE = re.compile(r"^not ok\s+\d+\s+(.+)$", re.MULTILINE)
# cargo: allow doctest node ids with spaces ("src/lib.rs - (line 42)")
_CARGO_STATUS_RE = re.compile(
    r"^test\s+(.+?)\s+\.\.\.\s+(ok|FAILED|ignored|error)\b",
    re.MULTILINE,
)
_CARGO_SKIP_NAME_RE = re.compile(r"^result\b|^\s*$")


def _parse_js_suite_log(text: str, *, language: str, returncode: int = 0) -> SuiteOutcome:
    """Parse jest JSON, ava checkmarks, or tape TAP into node outcomes."""
    # jest --json style (whole blob or last JSON object)
    candidates: list[str] = [text]
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("{")]
    if lines:
        candidates.append(lines[-1])
    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        passed: list[str] = []
        failed: list[str] = []
        for tr in data.get("testResults") or []:
            for assertion in tr.get("assertionResults") or []:
                name = assertion.get("title") or assertion.get("fullName")
                if not name:
                    continue
                status = assertion.get("status")
                if status == "passed":
                    passed.append(str(name))
                elif status == "failed":
                    failed.append(str(name))
        if passed or failed:
            return SuiteOutcome(
                language=language,
                passed=_dedupe(passed),
                failed=_dedupe(failed),
                returncode=returncode,
                raw_tail=text[-1500:],
            )

    # tape TAP (prefer when TAP version present or many `ok N` lines)
    tape_pass = [m.group(1).strip() for m in _TAPE_PASS_RE.finditer(text)]
    tape_fail = [m.group(1).strip() for m in _TAPE_FAIL_RE.finditer(text)]
    # Drop TAP plan noise ("1.." already excluded by @ok N title@ shape)
    if tape_pass or tape_fail:
        # When both spectaular journalling styles coexist, prefer the richer set.
        ava_pass = [m.group(1).strip() for m in _AVA_PASS_RE.finditer(text)]
        ava_fail = [m.group(1).strip() for m in _AVA_FAIL_RE.finditer(text)]
        if (len(tape_pass) + len(tape_fail)) >= (len(ava_pass) + len(ava_fail)):
            return SuiteOutcome(
                language=language,
                passed=_dedupe(tape_pass),
                failed=_dedupe(tape_fail),
                returncode=returncode,
                raw_tail=text[-1500:],
            )
        return SuiteOutcome(
            language=language,
            passed=_dedupe(ava_pass),
            failed=_dedupe(ava_fail),
            returncode=returncode,
            raw_tail=text[-1500:],
        )

    # ava / jest pretty reporter checkmarks
    ava_pass = [m.group(1).strip() for m in _AVA_PASS_RE.finditer(text)]
    ava_fail = [m.group(1).strip() for m in _AVA_FAIL_RE.finditer(text)]
    return SuiteOutcome(
        language=language,
        passed=_dedupe(ava_pass),
        failed=_dedupe(ava_fail),
        returncode=returncode,
        raw_tail=text[-1500:],
    )


class TypeScriptSuiteReporter:
    """JS/TS suite titles: jest, ava, and tape (also used for javascript motors)."""

    language = "typescript"
    tool_label = "jest"
    node_id_form = "name"
    reporter_id = "ts_jest_v1"

    def __init__(self, *, as_javascript: bool = False) -> None:
        if as_javascript:
            self.language = "javascript"
            self.tool_label = "npm-test"
            self.reporter_id = "js_npm_v1"

    def run(self, repo: Path) -> SuiteOutcome:
        if self.language == "javascript":
            out = run_javascript_suite(repo)
            return SuiteOutcome(
                language="javascript",
                passed=out.passed,
                failed=out.failed,
                errors=out.errors,
                returncode=out.returncode,
                raw_tail=out.raw_tail,
            )
        out = run_typescript_suite(repo)
        return out

    def parse_log(self, text: str, *, returncode: int = 0) -> SuiteOutcome:
        lang = "javascript" if self.language == "javascript" else "typescript"
        return _parse_js_suite_log(text, language=lang, returncode=returncode)


class RustSuiteReporter:
    """cargo test filtered names (parser-first; run requires cargo when present)."""

    language = "rust"
    tool_label = "cargo-test"
    node_id_form = "name"
    reporter_id = "rust_cargo_v1"

    def run(self, repo: Path) -> SuiteOutcome:
        if shutil.which("cargo") is None:
            raise HarborLabelError("cargo binary not available for rust dual-run labeling")
        proc = subprocess.run(
            # --test-threads=1 keeps line order stable for trash dual-run
            ["cargo", "test", "--", "--nocapture", "--test-threads=1"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        return self.parse_log(proc.stdout + "\n" + proc.stderr, returncode=int(proc.returncode))

    def parse_log(self, text: str, *, returncode: int = 0) -> SuiteOutcome:
        """Parse cargo unit and doctest lines into node ids.

        Supports:
        - ``test foo::bar ... ok`` / ``FAILED``
        - doctests with spaces: ``test src/lib.rs - (line 42) ... ok``
        - ignores ``ignored`` status and summary ``test result:`` noise
        """
        passed: list[str] = []
        failed: list[str] = []
        for m in _CARGO_STATUS_RE.finditer(text):
            name = m.group(1).strip()
            status = m.group(2)
            if not name or _CARGO_SKIP_NAME_RE.match(name) or name.startswith("result"):
                continue
            # Summary lines look like "test result: ok. N passed..."
            if "result:" in name or name == "result":
                continue
            if status == "ok":
                passed.append(name)
            elif status in {"FAILED", "error"}:
                failed.append(name)
            # ignored: skip
        return SuiteOutcome(
            language="rust",
            passed=_dedupe(passed),
            failed=_dedupe(failed),
            returncode=returncode,
            raw_tail=text[-1500:],
        )


_REPORTERS: dict[str, Callable[[], SuiteReporter]] = {
    "python": PythonSuiteReporter,
    "go": GoSuiteReporter,
    "typescript": TypeScriptSuiteReporter,
    "ts": TypeScriptSuiteReporter,
    "javascript": lambda: TypeScriptSuiteReporter(as_javascript=True),
    "js": lambda: TypeScriptSuiteReporter(as_javascript=True),
    "rust": RustSuiteReporter,
}


def normalize_reporter_lang(language: str) -> str:
    lang = language.strip().lower()
    aliases = {
        "py": "python",
        "ts": "typescript",
        "js": "javascript",
        "golang": "go",
        "rs": "rust",
    }
    return aliases.get(lang, lang)


def get_suite_reporter(language: str) -> SuiteReporter:
    """Return the suite reporter for a language (VAL-LABEL-006 adapters)."""
    key = normalize_reporter_lang(language)
    factory = _REPORTERS.get(key)
    if factory is None:
        raise HarborLabelError(
            f"no suite reporter for language {language!r}; "
            f"supported={sorted(set(normalize_reporter_lang(k) for k in _REPORTERS))}"
        )
    return factory()


def list_reporter_languages() -> tuple[str, ...]:
    """Canonical languages with dual-run reporter adapters."""
    return ("python", "go", "typescript", "javascript", "rust")


def reporter_info(language: str) -> ReporterInfo:
    rep = get_suite_reporter(language)
    return ReporterInfo(
        language=normalize_reporter_lang(language),
        tool_label=rep.tool_label,
        node_id_form=rep.node_id_form,
        reporter_id=getattr(rep, "reporter_id", f"{rep.language}_v1"),
    )


def run_with_reporter(repo: Path, language: str) -> SuiteOutcome:
    """Execute suite via the registry adapters (multi-lang dual-run feed)."""
    return get_suite_reporter(language).run(Path(repo))


def parse_with_reporter(language: str, text: str, *, returncode: int = 0) -> SuiteOutcome:
    """Parse fixed suite output offline (determinism / fixture paths)."""
    return get_suite_reporter(language).parse_log(text, returncode=returncode)


def grade_tool_label_for(language: str) -> str:
    """Map language → tests/config.json grade.tool_label for verifier parity."""
    return reporter_info(language).tool_label


def reporter_registry_snapshot() -> Mapping[str, dict[str, str]]:
    """Stable map of language → reporter identity for pack meta / evidence."""
    return {lang: reporter_info(lang).to_dict() for lang in list_reporter_languages()}


__all__ = [
    "GoSuiteReporter",
    "PythonSuiteReporter",
    "ReporterInfo",
    "ReporterLang",
    "RustSuiteReporter",
    "SuiteReporter",
    "TypeScriptSuiteReporter",
    "get_suite_reporter",
    "grade_tool_label_for",
    "list_reporter_languages",
    "normalize_reporter_lang",
    "parse_with_reporter",
    "reporter_info",
    "reporter_registry_snapshot",
    "run_with_reporter",
]
