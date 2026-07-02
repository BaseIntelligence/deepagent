"""Stage 2 spec synthesis: test-conditioned backtranslation (``GeneratedSpec``).

A :class:`~swe_forge.forge.models.Candidate` carries the manufactured fault and
its by-construction gold fix; this module turns that into the *agent-facing* task
description without ever revealing the fix. It does so by **test-conditioned
backtranslation**: the problem statement is derived from the candidate's
FAIL->PASS (F2P) failure trace - the observable broken behavior of the hidden
failing tests - NOT from the mutation/oracle diff. Alongside it we emit:

* a ``requirements`` list, each item grounded in (and traceable to) a named F2P
  test, and
* an ``interface_block`` enumerating the expected public symbol name(s) and
  signature(s) of the *real* target API (parsed from the pristine gold source via
  the language adapter), so a correct solution is never failed for a mere naming
  difference.

The governing principle holds here too - *the teacher proposes, deterministic
execution disposes*: the teacher only ever sees the failing-test trace and the
public signatures (never the gold body or any diff), and a deterministic leak
scan disposes of any output that copies an implementation line, an ``oracle``/
``mutation`` hunk line, or the generator name into the three spec fields. A
:class:`~swe_forge.forge.models.GeneratedSpec` is produced ONLY for a Candidate
that passes its forward+inverse self-validation; a failed run emits neither file.

No provider/brand string and no caching live here; the teacher client is built
lazily from :class:`~swe_forge.forge.config.ForgeSettings` only when the
LLM-backed author runs, and an offline, trace-derived template author is provided
for deterministic use.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Protocol

from swe_forge.forge.adapters.base import LanguageAdapter, ParseError, Symbol
from swe_forge.forge.generators._llm import run_sync
from swe_forge.forge.models import (
    GENERATOR_NAMES,
    Candidate,
    GeneratedSpec,
    Provenance,
)
from swe_forge.forge.teacher import TeacherClient


class SpecError(RuntimeError):
    """Raised when a valid, non-leaking :class:`GeneratedSpec` cannot be produced.

    Signals the caller to abort and emit NO spec artifact (a spec is only ever
    written alongside a valid Candidate; a failed run produces neither file).
    """


# --------------------------------------------------------------------------- #
# F2P failure trace (the INPUT to backtranslation)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FailingTest:
    """One hidden F2P test and the observable behavior it asserts.

    ``name`` is the named test id (e.g. ``tests/test_calc.py::test_negative``)
    the requirements trace back to. ``message`` is the observed failure output
    (assertion text / error), and ``expected``/``observed`` capture the expected
    vs. actual result when known. Together these are the *observable behavior*
    the problem statement is backtranslated from - never the diff.
    """

    name: str
    file: str = ""
    message: str = ""
    expected: str = ""
    observed: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise SpecError("FailingTest.name must be non-empty")

    def behavior_line(self) -> str:
        """A one-line description of the observable behavior this test checks."""
        parts: list[str] = []
        if self.expected:
            if self.observed:
                parts.append(f"expected {self.expected}, but got {self.observed}")
            else:
                parts.append(f"expected {self.expected}")
        elif self.message:
            parts.append(self.message.strip().splitlines()[0])
        elif self.observed:
            parts.append(f"observed {self.observed}")
        detail = "; ".join(parts) if parts else "the assertion currently fails"
        return detail

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "file": self.file,
            "message": self.message,
            "expected": self.expected,
            "observed": self.observed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FailingTest:
        return cls(
            name=str(data.get("name", "")),
            file=str(data.get("file", "")),
            message=str(data.get("message", "")),
            expected=str(data.get("expected", "")),
            observed=str(data.get("observed", "")),
        )


@dataclass(frozen=True)
class F2PTrace:
    """The FAIL->PASS failure trace fed to backtranslation.

    Holds the named failing tests (``tests``) and, optionally, the ``raw`` trace
    text. At least one failing test is required - a problem statement cannot be
    test-conditioned without an observed failure.
    """

    tests: tuple[FailingTest, ...]
    raw: str = ""

    def test_names(self) -> tuple[str, ...]:
        return tuple(test.name for test in self.tests)

    def to_dict(self) -> dict[str, object]:
        return {"tests": [test.to_dict() for test in self.tests], "raw": self.raw}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> F2PTrace:
        raw_tests = data.get("tests")
        if raw_tests is None:
            raw_tests = data.get("fail_to_pass", [])
        tests: list[FailingTest] = []
        if isinstance(raw_tests, list):
            for item in raw_tests:
                if isinstance(item, dict):
                    tests.append(FailingTest.from_dict(item))
                elif isinstance(item, str) and item.strip():
                    tests.append(FailingTest(name=item.strip()))
        return cls(tests=tuple(tests), raw=str(data.get("raw", "")))


# --------------------------------------------------------------------------- #
# Interface block (deterministic, from the real gold API)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterfaceSymbol:
    """One expected public symbol of the target API (signature only, no body)."""

    name: str
    kind: str
    file: str
    signature: str

    def render(self) -> str:
        sig = self.signature.strip() if self.signature.strip() else self.name
        return f"- {sig}  [{self.kind}]  ({self.file})"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class InterfaceBlock:
    """The rendered interface text plus the structured symbols it lists."""

    text: str
    symbols: tuple[InterfaceSymbol, ...]

    def signatures(self) -> tuple[str, ...]:
        out: list[str] = []
        for sym in self.symbols:
            if sym.signature.strip():
                out.append(sym.signature.strip())
            out.append(sym.name)
        return tuple(out)

    def to_provenance(self) -> list[dict[str, str]]:
        return [sym.to_dict() for sym in self.symbols]


_OPEN_BRACKETS = "([{"
_CLOSE_BRACKETS = ")]}"


def _split_top_level_commas(text: str) -> list[str]:
    """Split ``text`` on commas that sit at bracket/quote depth zero."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    prev = ""
    for ch in text:
        if quote is not None:
            current.append(ch)
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
            current.append(ch)
        elif ch in _OPEN_BRACKETS:
            depth += 1
            current.append(ch)
        elif ch in _CLOSE_BRACKETS:
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        prev = ch
    if current:
        parts.append("".join(current))
    return parts


def _strip_param_default(param: str) -> str:
    """Drop a parameter's top-level ``= default`` value, keeping name+annotation.

    The default VALUE (e.g. ``lambda name: '*' + name``, ``{}``, ``0``) is an
    implementation token, not part of the public contract, so it is removed; the
    parameter name and any ``: annotation`` (the type contract) are preserved.
    Only a genuine default ``=`` at bracket/quote depth zero is cut - never a
    ``==``/``<=``/``>=``/``!=``/``:=`` operator that could appear inside an
    annotation, and never one nested in brackets or a string.
    """
    depth = 0
    quote: str | None = None
    prev = ""
    for i, ch in enumerate(param):
        if quote is not None:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in _OPEN_BRACKETS:
            depth += 1
        elif ch in _CLOSE_BRACKETS:
            depth -= 1
        elif ch == "=" and depth == 0:
            nxt = param[i + 1] if i + 1 < len(param) else ""
            if prev not in "=<>!:" and nxt != "=":
                return param[:i].strip()
        prev = ch
    return param.strip()


def _contract_signature(signature: str) -> str:
    """Return ``signature`` with every parameter default VALUE removed.

    The agent-visible interface block must be a signature/behavioral contract
    only: it lists the public name, parameters, and (return/param) type
    annotations, but NOT any default-value expression. Those default expressions
    are real implementation tokens that also appear verbatim as source/patch
    lines of the faulted symbol (especially on multi-line signatures in modular
    repos), so embedding them makes the leak auditor CORRECTLY reject an
    otherwise-legitimate candidate. Stripping them removes the SOURCE of that
    leak while leaving the auditor untouched. Language-agnostic: it operates on
    the parameter list between the first top-level ``(`` and its match, so Go
    (no defaults) is a no-op and Python/JS defaults are dropped.
    """
    sig = signature.strip()
    open_idx = sig.find("(")
    if open_idx == -1:
        return sig
    depth = 0
    quote: str | None = None
    prev = ""
    close_idx = -1
    for i in range(open_idx, len(sig)):
        ch = sig[i]
        if quote is not None:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in _OPEN_BRACKETS:
            depth += 1
        elif ch in _CLOSE_BRACKETS:
            depth -= 1
            if depth == 0:
                close_idx = i
                break
        prev = ch
    if close_idx == -1:
        return sig
    prefix = sig[:open_idx]
    params_str = sig[open_idx + 1 : close_idx]
    suffix = sig[close_idx + 1 :]
    if not params_str.strip():
        return f"{prefix}(){suffix}".rstrip()
    cleaned = [
        stripped
        for param in _split_top_level_commas(params_str)
        if (stripped := _strip_param_default(param))
    ]
    return f"{prefix}({', '.join(cleaned)}){suffix}".rstrip()


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def _changed_original_ranges(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Map each touched file to the original-side line ranges its hunks change.

    The original-side line numbers (the ``-`` side of each ``@@`` header) index
    the *pristine* source, so they line up with the adapter's parsed symbol
    spans. Returns ``{rel_path: [(start, end), ...]}``.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    current: str | None = None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current = None if path == "/dev/null" else path
            continue
        match = _HUNK_RE.match(line)
        if match and current is not None:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            end = start + count - 1 if count > 0 else start
            ranges.setdefault(current, []).append((min(start, end), max(start, end)))
    return ranges


def _overlaps(symbol: Symbol, spans: list[tuple[int, int]]) -> bool:
    for start, end in spans:
        if not (symbol.end_line < start or symbol.start_line > end):
            return True
    return False


def build_interface_block(
    candidate: Candidate, repo_root: Path, adapter: LanguageAdapter
) -> InterfaceBlock:
    """Build the interface block from the candidate's *pristine* target source.

    Parses the gold (pre-mutation) source of every target file with the language
    adapter and lists the expected signatures: the candidate's declared target
    symbols when present, else the symbols whose definition the mutation touches
    (inferred from the patch hunks), else every symbol in the file. Every listed
    signature therefore matches a real definition in the original target source,
    and the candidate's target symbol(s) always appear.
    """
    selected: list[InterfaceSymbol] = []
    seen: set[tuple[str, str]] = set()
    explicit = {s for s in candidate.target.symbols if s}

    for rel in candidate.target.files:
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            symbols = adapter.parse_symbols(path)
        except ParseError:
            symbols = []
        if not symbols:
            continue

        chosen = _choose_symbols(candidate, rel, symbols, explicit)
        for sym in chosen:
            key = (rel, sym.name)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                InterfaceSymbol(
                    name=sym.name,
                    kind=sym.kind,
                    file=rel,
                    signature=_contract_signature(
                        (sym.signature or f"{sym.name}").strip()
                    ),
                )
            )

    text = _render_interface(selected)
    return InterfaceBlock(text=text, symbols=tuple(selected))


def _choose_symbols(
    candidate: Candidate, rel: str, symbols: list[Symbol], explicit: set[str]
) -> list[Symbol]:
    """Pick the symbols to expose for ``rel`` (explicit, else touched, else all)."""
    if explicit:
        matched = [s for s in symbols if s.name in explicit]
        if matched:
            return matched
    spans = _changed_original_ranges(candidate.mutation_patch).get(rel, [])
    if spans:
        touched = [s for s in symbols if _overlaps(s, spans)]
        if touched:
            return touched
    return symbols


def _render_interface(symbols: list[InterfaceSymbol]) -> str:
    """Render the human-facing interface block (signatures only, grouped by file)."""
    header = (
        "Expected interface (implement these public symbols; signatures only - "
        "bodies are up to you):"
    )
    if not symbols:
        return header
    by_file: dict[str, list[InterfaceSymbol]] = {}
    for sym in symbols:
        by_file.setdefault(sym.file, []).append(sym)
    lines = [header]
    for rel in sorted(by_file):
        lines.append(f"# {rel}")
        for sym in by_file[rel]:
            lines.append(sym.render())
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Spec author (teacher proposes) protocol + implementations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpecAuthoringContext:
    """Inputs handed to a :class:`SpecAuthor` (the failing tests + interface only)."""

    language: str
    failing_tests: tuple[FailingTest, ...]
    interface_block: str
    target_files: tuple[str, ...]
    difficulty_hint: str = ""


@dataclass(frozen=True)
class AuthoredRequirement:
    """One drafted requirement plus the named F2P test it is meant to trace to."""

    text: str
    test: str = ""


@dataclass(frozen=True)
class AuthoredSpec:
    """A drafted problem statement + requirements plus authoring usage/cost."""

    problem_statement: str
    requirements: tuple[AuthoredRequirement, ...]
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0


class SpecAuthor(Protocol):
    """Drafts a problem statement + requirements from the F2P trace (sync surface)."""

    def __call__(self, ctx: SpecAuthoringContext) -> AuthoredSpec: ...


_SYSTEM_PROMPT = (
    "You write the task description for a software-engineering benchmark FROM "
    "FAILING TESTS ONLY. You are given the observed failures of hidden tests and "
    "the public interface (signatures only) the solver must implement. Write a "
    "`problem_statement` that describes the behavior the solver must achieve, "
    "phrased entirely from the failing tests' observable behavior (what is "
    "expected vs. what currently happens). Then write a `requirements` list where "
    "EACH item maps to exactly ONE of the provided failing test names and states "
    "the behavior that test checks. Hard constraints: describe WHAT must hold, "
    "never HOW; do NOT include, quote, paraphrase line-by-line, or invent any "
    "implementation/solution code, diffs, file contents, or fix details beyond "
    "the signatures you were given; do NOT mention how the bug was created. Output "
    "JSON only."
)

_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "problem_statement": {"type": "string"},
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "test": {"type": "string"},
                },
                "required": ["requirement", "test"],
            },
        },
    },
    "required": ["problem_statement", "requirements"],
}


def _format_failing_tests(tests: tuple[FailingTest, ...]) -> str:
    blocks: list[str] = []
    for test in tests:
        lines = [f"- test: {test.name}"]
        if test.file:
            lines.append(f"  file: {test.file}")
        if test.expected:
            lines.append(f"  expected: {test.expected}")
        if test.observed:
            lines.append(f"  observed: {test.observed}")
        if test.message:
            first = test.message.strip().splitlines()[0]
            lines.append(f"  failure: {first}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


class TeacherSpecAuthor:
    """Default :class:`SpecAuthor` backed by the LiteLLM teacher client."""

    def __init__(self, client: TeacherClient, *, max_tokens: int = 1024) -> None:
        self._client = client
        self._max_tokens = max_tokens

    @classmethod
    def from_settings(cls, *, max_tokens: int = 1024) -> TeacherSpecAuthor:
        return cls(TeacherClient.from_settings(), max_tokens=max_tokens)

    def __call__(self, ctx: SpecAuthoringContext) -> AuthoredSpec:
        prompt = (
            f"Language: {ctx.language}\n"
            f"Target files: {', '.join(ctx.target_files)}\n\n"
            f"Failing tests (observable behavior to satisfy):\n"
            f"{_format_failing_tests(ctx.failing_tests)}\n\n"
            f"Public interface the solver must implement:\n{ctx.interface_block}\n\n"
            "Allowed failing test names to map requirements to:\n"
            + "\n".join(f"- {t.name}" for t in ctx.failing_tests)
        )
        result = run_sync(
            self._client.complete_json(
                prompt,
                _SPEC_SCHEMA,
                system=_SYSTEM_PROMPT,
                schema_name="generated_spec",
                max_tokens=self._max_tokens,
            )
        )
        problem, requirements = _parse_authored_spec(result.text)
        return AuthoredSpec(
            problem_statement=problem,
            requirements=requirements,
            model=self._client.model,
            usage=result.usage.to_dict(),
            cost=result.cost,
        )


class TemplateSpecAuthor:
    """Offline, deterministic :class:`SpecAuthor` derived purely from the trace.

    Produces a problem statement and one requirement per failing test directly
    from the F2P trace (never the diff), so the deterministic spec machinery
    (interface block, leak scan, traceability, pairing) can be exercised without
    the live endpoint. Marked ``template/offline`` in provenance.
    """

    model = "template/offline"

    def __call__(self, ctx: SpecAuthoringContext) -> AuthoredSpec:
        tests = ctx.failing_tests
        intro = (
            f"A {ctx.language} target is currently failing "
            f"{len(tests)} hidden test"
            f"{'s' if len(tests) != 1 else ''} that exercise its expected "
            "behavior. Fix the target so the described behavior holds."
        )
        bullets = [f"- `{test.name}`: {test.behavior_line()}." for test in tests]
        problem = intro + "\n\nObserved failing behavior:\n" + "\n".join(bullets)
        requirements = tuple(
            AuthoredRequirement(
                text=(f"Satisfy `{test.name}`: {test.behavior_line()}."),
                test=test.name,
            )
            for test in tests
        )
        return AuthoredSpec(
            problem_statement=problem,
            requirements=requirements,
            model=self.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cost=0.0,
        )


def _parse_authored_spec(text: str) -> tuple[str, tuple[AuthoredRequirement, ...]]:
    """Parse a teacher JSON reply into ``(problem_statement, requirements)``.

    Tolerant of a surrounding Markdown code fence and of requirements given as
    plain strings (then the F2P mapping falls back to single-test grounding).
    """
    raw = (text or "").strip()
    fence = re.match(r"^```[^\n]*\n(?P<body>.*)\n```\s*$", raw, re.DOTALL)
    if fence:
        raw = fence.group("body").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpecError(f"spec author returned unparseable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError("spec author did not return a JSON object")

    problem = ""
    for key in ("problem_statement", "problem", "statement", "description"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            problem = value.strip()
            break

    requirements: list[AuthoredRequirement] = []
    raw_reqs = data.get("requirements")
    if isinstance(raw_reqs, list):
        for item in raw_reqs:
            if isinstance(item, str) and item.strip():
                requirements.append(AuthoredRequirement(text=item.strip()))
            elif isinstance(item, dict):
                req_text = ""
                for key in ("requirement", "text", "description", "behavior"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        req_text = value.strip()
                        break
                test = ""
                for key in ("test", "test_name", "f2p", "fail_to_pass"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        test = value.strip()
                        break
                if req_text:
                    requirements.append(AuthoredRequirement(text=req_text, test=test))
    return problem, tuple(requirements)


# --------------------------------------------------------------------------- #
# Leak scan (deterministic execution disposes)
# --------------------------------------------------------------------------- #
_STRUCTURAL = set("=(){}[];:<>+-*/%\"'`")


def _is_checkable_code_line(stripped: str) -> bool:
    """Return ``True`` iff ``stripped`` is a meaningful (non-trivial) code line."""
    if len(stripped) < 6:
        return False
    if not any(c.isalpha() for c in stripped):
        return False
    if any(c in _STRUCTURAL for c in stripped) or any(c.isdigit() for c in stripped):
        return True
    return False


def _normalize_sig(text: str) -> str:
    return "".join(text.strip().rstrip("{:").split())


def _patch_body_lines(patch: str) -> list[str]:
    """Return the content of every added/removed/context line in ``patch``."""
    skip = ("+++", "---", "diff ", "index ", "@@", "new file", "deleted file")
    out: list[str] = []
    for line in patch.splitlines():
        if line.startswith(skip) or line.startswith("\\"):
            continue
        if line and line[0] in "+- ":
            out.append(line[1:])
    return out


def _forbidden_code_lines(
    candidate: Candidate, signatures: tuple[str, ...]
) -> set[str]:
    """Collect the implementation/diff lines that must never appear in the spec.

    Drawn from both the oracle (gold) and mutation patches, restricted to
    non-trivial code lines, with the public signatures excluded (the interface
    legitimately exposes them).
    """
    sig_norms = {_normalize_sig(sig) for sig in signatures if sig.strip()}
    forbidden: set[str] = set()
    for patch in (candidate.oracle_patch, candidate.mutation_patch):
        for line in _patch_body_lines(patch):
            stripped = line.strip()
            if not _is_checkable_code_line(stripped):
                continue
            if _normalize_sig(stripped) in sig_norms:
                continue
            forbidden.add(stripped)
    return forbidden


def scan_spec_for_leaks(
    problem_statement: str,
    requirements: list[str],
    interface_block: str,
    candidate: Candidate,
    signatures: tuple[str, ...],
) -> list[str]:
    """Return a list of leak findings (empty when the three fields are clean).

    Flags any non-trivial oracle/mutation implementation line copied into a field,
    and any occurrence of a generator name. The interface legitimately exposes
    signatures, which are excluded from the forbidden set.
    """
    findings: list[str] = []
    forbidden = _forbidden_code_lines(candidate, signatures)
    fields = {
        "problem_statement": problem_statement,
        "requirements": "\n".join(requirements),
        "interface_block": interface_block,
    }
    for field_name, field_text in fields.items():
        for line in forbidden:
            if line in field_text:
                findings.append(
                    f"{field_name} copies an implementation/diff line: {line!r}"
                )
        for gname in GENERATOR_NAMES:
            if gname in field_text:
                findings.append(f"{field_name} mentions the generator name {gname!r}")
    return findings


# --------------------------------------------------------------------------- #
# Requirement -> F2P test grounding
# --------------------------------------------------------------------------- #
def _ground_requirement(requested_test: str, trace: F2PTrace) -> str | None:
    """Map a drafted requirement's test reference to a real F2P test name.

    Returns the matched F2P test name, or ``None`` when it cannot be grounded.
    Falls back to the sole failing test when the trace has exactly one.
    """
    names = trace.test_names()
    requested = requested_test.strip()
    if requested in names:
        return requested
    if requested:
        for name in names:
            if requested in name or name in requested:
                return name
        short = requested.rsplit("::", 1)[-1]
        for name in names:
            if short and short in name:
                return name
    if len(names) == 1:
        return names[0]
    return None


def _tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    with contextlib.suppress(metadata.PackageNotFoundError):
        versions["litellm"] = metadata.version("litellm")
    return versions


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def generate_spec(
    candidate: Candidate,
    f2p_trace: F2PTrace,
    repo_root: Path,
    adapter: LanguageAdapter,
    *,
    author: SpecAuthor | None = None,
) -> GeneratedSpec:
    """Produce a verified, non-leaking :class:`GeneratedSpec` for ``candidate``.

    Builds the interface block deterministically from the pristine gold source,
    drafts the problem statement + requirements from the F2P trace via ``author``
    (teacher-backed by default), grounds each requirement in a named F2P test,
    and runs the leak scan. Raises :class:`SpecError` (emitting nothing) when the
    trace is empty, the interface cannot be derived, no requirement can be
    grounded, or any of the three fields leaks oracle/implementation content.
    """
    if not f2p_trace.tests:
        raise SpecError(
            "F2P trace has no failing tests; cannot test-condition a problem statement"
        )

    interface = build_interface_block(candidate, repo_root, adapter)
    if not interface.symbols:
        raise SpecError(
            "could not derive any interface symbol from the target API "
            f"(files={list(candidate.target.files)})"
        )

    spec_author: SpecAuthor = author or TeacherSpecAuthor.from_settings()
    authored = spec_author(
        SpecAuthoringContext(
            language=candidate.language,
            failing_tests=f2p_trace.tests,
            interface_block=interface.text,
            target_files=tuple(candidate.target.files),
            difficulty_hint=candidate.difficulty_hint,
        )
    )

    problem = authored.problem_statement.strip()
    if not problem:
        raise SpecError("spec author produced an empty problem_statement")

    requirements: list[str] = []
    traceability: list[dict[str, str]] = []
    for drafted in authored.requirements:
        text = drafted.text.strip()
        if not text:
            continue
        grounded = _ground_requirement(drafted.test, f2p_trace)
        if grounded is None:
            continue
        requirements.append(text)
        traceability.append({"requirement": text, "test": grounded})
    if not requirements:
        raise SpecError(
            "no requirement could be grounded in a named F2P test "
            f"(trace tests: {list(f2p_trace.test_names())})"
        )

    leaks = scan_spec_for_leaks(
        problem, requirements, interface.text, candidate, interface.signatures()
    )
    if leaks:
        raise SpecError("spec leaks oracle/implementation content: " + "; ".join(leaks))

    authoring_mode = (
        "template" if authored.model == TemplateSpecAuthor.model else "teacher"
    )
    provenance = Provenance(
        generator=candidate.generator,
        seed=candidate.provenance.seed,
        language=candidate.language,
        tool_versions=_tool_versions(),
        details={
            "stage": "spec",
            "source": "test-conditioned backtranslation",
            "authoring_mode": authoring_mode,
            "candidate_generator": candidate.generator,
            "f2p_trace": f2p_trace.to_dict(),
            "f2p_tests": list(f2p_trace.test_names()),
            "interface_symbols": interface.to_provenance(),
            "requirement_traceability": traceability,
            "leak_scan": {"checked": True, "clean": True},
            "teacher": {
                "model": authored.model,
                "usage": dict(authored.usage),
                "cost": authored.cost,
            },
        },
    )

    return GeneratedSpec(
        problem_statement=problem,
        requirements=requirements,
        interface_block=interface.text,
        provenance=provenance,
    )


__all__ = [
    "AuthoredRequirement",
    "AuthoredSpec",
    "F2PTrace",
    "FailingTest",
    "InterfaceBlock",
    "InterfaceSymbol",
    "SpecAuthor",
    "SpecAuthoringContext",
    "SpecError",
    "TeacherSpecAuthor",
    "TemplateSpecAuthor",
    "build_interface_block",
    "generate_spec",
    "scan_spec_for_leaks",
]
