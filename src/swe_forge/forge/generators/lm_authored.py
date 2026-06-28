"""The ``lm_authored`` generator: a subtle, single-function, teacher-authored bug.

The teacher proposes one subtle logic bug confined to a single function (no
bug-signposting comments); deterministic execution disposes. The proposed edit is
spliced back over exactly the target symbol's line span, then validated through
the shared git-backed round-trip check (apply the forward ``mutation_patch``,
derive the inverse ``oracle_patch``, re-apply, require a byte-for-byte sha256
match). An edit that does not parse, does not change behavior, signposts the bug,
adds/removes other symbols, or fails to invert is rejected - never shipped.

Teacher usage/cost is recorded in provenance (no caching; per-call no-cache is
enforced by the teacher client). The generator is not deterministic across runs
(the teacher is the source of the edit); ``seed`` still orders target selection.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from swe_forge.forge.adapters.base import LanguageAdapter, Patch, Symbol
from swe_forge.forge.adapters._diff import make_patch
from swe_forge.forge.generators._llm import (
    contains_signposting,
    extract_code_field,
    normalize_code,
    run_sync,
    splice_symbol_lines,
)
from swe_forge.forge.generators._targeting import (
    SingleFileFault,
    parse_symbols_safe,
    resolve_target_files,
    sha256_bytes,
    tool_versions,
    verify_forward_patch,
)
from swe_forge.forge.generators.base import (
    BugGenerator,
    GenerationError,
    GenerationRequest,
)
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    Provenance,
    require_green_baseline,
)
from swe_forge.forge.teacher import TeacherClient

# Only target functions whose body is small enough to keep the diff bounded and
# the bug localized; huge functions are skipped (not single-fault friendly).
_MAX_TARGET_LINES = 80
# Reject an edit whose line count balloons relative to the original block - a
# subtle single-function bug should not rewrite the function wholesale.
_MAX_LINE_GROWTH = 6
# Bound teacher calls per generation for cost discipline.
_DEFAULT_MAX_ATTEMPTS = 4

_SYSTEM_PROMPT = (
    "You build hard but fair debugging benchmarks. Given one function, introduce "
    "EXACTLY ONE subtle logic bug so that at least one unit test of this function "
    "would fail. Hard constraints: keep the function name and signature identical; "
    "preserve the exact indentation of the original block; modify ONLY this one "
    "function and add no new functions; do NOT add, remove, or change comments; "
    "do NOT mention, hint at, or signpost the bug anywhere (no words like 'bug', "
    "'fixme', 'intentional', etc.). Return only the full modified function source."
)

_FUNCTION_SCHEMA = {
    "type": "object",
    "properties": {"function": {"type": "string"}},
    "required": ["function"],
}

_FUNCTION_KEYS = (
    "function",
    "source",
    "code",
    "buggy_function",
    "new_source",
    "mutated",
)


@dataclass(frozen=True)
class AuthoringContext:
    """Inputs handed to a :class:`BugAuthor` for one target function."""

    language: str
    rel: str
    symbol_name: str
    signature: str
    function_source: str
    seed: int
    attempt: int


@dataclass(frozen=True)
class AuthoredEdit:
    """A teacher-proposed replacement for a target function plus usage/cost."""

    new_source: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0


class BugAuthor(Protocol):
    """Authors a subtle single-function bug for one target (sync surface)."""

    def __call__(self, ctx: AuthoringContext) -> AuthoredEdit: ...


class TeacherBugAuthor:
    """Default :class:`BugAuthor` backed by the LiteLLM teacher client."""

    def __init__(self, client: TeacherClient, *, max_tokens: int = 1024) -> None:
        self._client = client
        self._max_tokens = max_tokens

    @classmethod
    def from_settings(cls, *, max_tokens: int = 1024) -> "TeacherBugAuthor":
        return cls(TeacherClient.from_settings(), max_tokens=max_tokens)

    def __call__(self, ctx: AuthoringContext) -> AuthoredEdit:
        prompt = (
            f"Language: {ctx.language}\n"
            f"Function `{ctx.symbol_name}` to mutate:\n\n{ctx.function_source}"
        )
        result = run_sync(
            self._client.complete_json(
                prompt,
                _FUNCTION_SCHEMA,
                system=_SYSTEM_PROMPT,
                schema_name="authored_bug",
                max_tokens=self._max_tokens,
            )
        )
        new_source = extract_code_field(result.text, _FUNCTION_KEYS)
        return AuthoredEdit(
            new_source=new_source,
            model=self._client.model,
            usage=result.usage.to_dict(),
            cost=result.cost,
        )


class LmAuthoredGenerator(BugGenerator):
    """Teacher-authored subtle single-function bug generator."""

    name = "lm_authored"

    def __init__(self, author: BugAuthor | None = None) -> None:
        self._author = author

    def _resolve_author(self) -> BugAuthor:
        if self._author is not None:
            return self._author
        return TeacherBugAuthor.from_settings()

    def generate(
        self, request: GenerationRequest, adapter: LanguageAdapter
    ) -> Candidate:
        if request.env_image is not None:
            require_green_baseline(request.env_image)

        repo_root = Path(request.repo_root).resolve()
        if not repo_root.is_dir():
            raise GenerationError(f"repo path is not a directory: {repo_root}")

        author = self._resolve_author()
        max_attempts = _max_attempts(request)

        with contextlib.chdir(repo_root):
            try:
                files = resolve_target_files(repo_root, adapter, request.file)
            except FileNotFoundError:
                raise GenerationError(
                    f"lm_authored: target file not found: {request.file}"
                ) from None
            sites = self._candidate_sites(adapter, files, request.symbol)
            if not sites:
                raise GenerationError(
                    f"lm_authored: no suitable function found for "
                    f"{self._describe_target(request)} in {repo_root}"
                )
            random.Random(request.seed).shuffle(sites)

            for attempt, (rel, symbol) in enumerate(sites[:max_attempts]):
                candidate = self._try_site(
                    request, adapter, author, rel, symbol, attempt
                )
                if candidate is not None:
                    return candidate

        raise GenerationError(
            f"lm_authored: the teacher produced no subtle, single-function, "
            f"round-tripping bug for {self._describe_target(request)} in {repo_root}"
        )

    def _try_site(
        self,
        request: GenerationRequest,
        adapter: LanguageAdapter,
        author: BugAuthor,
        rel: str,
        symbol: Symbol,
        attempt: int,
    ) -> Candidate | None:
        original = Path(rel).read_bytes()
        block = _symbol_block(original, symbol)
        if block is None:
            return None

        edit = author(
            AuthoringContext(
                language=adapter.name,
                rel=rel,
                symbol_name=symbol.name,
                signature=symbol.signature or "",
                function_source=block,
                seed=request.seed,
                attempt=attempt,
            )
        )
        new_source = (edit.new_source or "").strip("\n")
        if not new_source:
            return None
        # Subtlety gate: the planted bug must not be advertised.
        if contains_signposting(new_source):
            return None
        # Behavior gate: a comment/whitespace-only churn changes nothing.
        if normalize_code(new_source) == normalize_code(block):
            return None
        if _line_growth(block, new_source) > _MAX_LINE_GROWTH:
            return None

        try:
            mutated = splice_symbol_lines(original, symbol, edit.new_source)
        except ValueError:
            return None
        if mutated == original:
            return None
        # Single-function + target-match gate: the edit must parse and leave the
        # file's symbol set unchanged (no added/removed functions), with the
        # target symbol still present.
        if not self._symbol_set_preserved(adapter, rel, original, mutated, symbol):
            return None

        forward = Patch(diff=make_patch(rel, original, mutated), files=(rel,))
        fault = verify_forward_patch(rel, symbol, forward)
        if fault is None:
            return None
        return self._build_candidate(request, adapter, fault, edit)

    def _symbol_set_preserved(
        self,
        adapter: LanguageAdapter,
        rel: str,
        original: bytes,
        mutated: bytes,
        symbol: Symbol,
    ) -> bool:
        before = {s.name for s in parse_symbols_safe(adapter, rel)}
        path = Path(rel)
        path.write_bytes(mutated)
        try:
            after_symbols = parse_symbols_safe(adapter, rel)
        finally:
            path.write_bytes(original)
        after = {s.name for s in after_symbols}
        if not after_symbols and before:
            return False
        return after == before and symbol.name in after

    def _candidate_sites(
        self, adapter: LanguageAdapter, files: list[str], symbol_hint: str | None
    ) -> list[tuple[str, Symbol]]:
        sites: list[tuple[str, Symbol]] = []
        for rel in files:
            for symbol in parse_symbols_safe(adapter, rel):
                if symbol_hint is not None and symbol.name != symbol_hint:
                    continue
                if symbol.end_line - symbol.start_line + 1 > _MAX_TARGET_LINES:
                    continue
                if symbol.end_line <= symbol.start_line:
                    continue
                sites.append((rel, symbol))
        return sites

    def _build_candidate(
        self,
        request: GenerationRequest,
        adapter: LanguageAdapter,
        fault: SingleFileFault,
        edit: AuthoredEdit,
    ) -> Candidate:
        provenance = Provenance(
            generator=self.name,
            seed=request.seed,
            language=adapter.name,
            tool_versions=tool_versions(),
            details={
                "operation": "lm_authored",
                "file": fault.rel,
                "symbol": fault.symbol.name,
                "symbol_kind": fault.symbol.kind,
                "signature": fault.symbol.signature or "",
                "start_line": fault.symbol.start_line,
                "end_line": fault.symbol.end_line,
                "original_sha256": sha256_bytes(fault.original),
                "mutated_sha256": sha256_bytes(fault.mutated),
                "teacher": teacher_usage_details_from_edit(edit),
            },
        )
        return Candidate(
            language=adapter.name,
            generator=self.name,
            target=CandidateTarget(files=(fault.rel,), symbols=(fault.symbol.name,)),
            mutation_patch=fault.mutation_patch,
            oracle_patch=fault.oracle_patch,
            difficulty_hint="high",
            provenance=provenance,
        )

    def _describe_target(self, request: GenerationRequest) -> str:
        parts = []
        if request.file:
            parts.append(f"file={request.file}")
        if request.symbol:
            parts.append(f"symbol={request.symbol}")
        return ", ".join(parts) if parts else "any target"


def teacher_usage_details_from_edit(edit: AuthoredEdit) -> dict[str, object]:
    """Provenance ``teacher`` record from an :class:`AuthoredEdit` (no secrets)."""
    return {
        "model": edit.model,
        "usage": dict(edit.usage),
        "cost": edit.cost,
    }


def _symbol_block(original: bytes, symbol: Symbol) -> str | None:
    """Return the exact source text of ``symbol``'s line span, or ``None``."""
    lines = original.decode("utf-8").splitlines(keepends=True)
    start = max(symbol.start_line - 1, 0)
    end = min(symbol.end_line, len(lines))
    if start >= end:
        return None
    return "".join(lines[start:end])


def _line_growth(original_block: str, new_source: str) -> int:
    return len(new_source.splitlines()) - len(original_block.splitlines())


def _max_attempts(request: GenerationRequest) -> int:
    value = request.params.get("max_attempts")
    if isinstance(value, int) and value > 0:
        return value
    return _DEFAULT_MAX_ATTEMPTS


# Re-exported for callers that build provenance from an AuthoredEdit directly.
__all__ = [
    "AuthoredEdit",
    "AuthoringContext",
    "BugAuthor",
    "LmAuthoredGenerator",
    "TeacherBugAuthor",
]
