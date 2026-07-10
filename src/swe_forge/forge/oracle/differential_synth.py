"""Teacher-backed plausible-wrong variant generation + survivor-killing tests.

The differential-vs-gold gate needs two teacher-proposed inputs (and execution
disposes of both): a set of **plausible-but-wrong variants** of the gold code
(:class:`TeacherVariantGenerator`) and, when a variant survives the suite, a
**separating test** that passes on gold and fails on the variant
(:class:`DifferentialKillSynthesizer`). Every model call routes through the
env-driven LiteLLM teacher (:class:`swe_forge.forge.teacher.TeacherClient`) -
never the bespoke ``swe_forge.llm.*`` clients or any response cache.

Governing principle: *the teacher proposes, deterministic execution disposes*.
The variants here are only PROPOSALS; the gate runs each through the Docker
F2P+P2P suite and only a variant that actually passes the suite (a survivor)
drives strengthening, and each proposed separating test is confirmed by
execution (pass-on-gold AND fail-on-variant) before it is recorded.
"""

from __future__ import annotations

import re
from logging import getLogger

from swe_forge.forge.oracle.differential import (
    DifferentialSynthesisContext,
    Variant,
    VariantFile,
    VariantGenerationContext,
)
from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile
from swe_forge.forge.oracle.teacher_evidence import TeacherGateCallEvidence
from swe_forge.forge.oracle.teacher_regions import (
    TeacherSource,
    required_symbol,
    select_teacher_source,
)
from swe_forge.forge.teacher import LLMResult, TeacherClient, Usage

logger = getLogger(__name__)

DEFAULT_MAX_TOKENS = 3072
_CODE_FENCE_RE = re.compile(r"```(?:[\w.+-]*)\n(.*?)```", re.DOTALL)


_VARIANT_SYSTEM_PROMPT = """You produce PLAUSIBLE-BUT-WRONG variant implementations
of some correct code, for differential testing of a hidden test suite. The code
you are shown is CORRECT (the gold implementation). Your job: rewrite the SAME file
in several different ways that each look reasonable but contain a SUBTLE behavioral
bug - the kind of mistake a competent engineer might actually make.

Rules:
- Keep the SAME public interface (same symbol names and signatures) so the variant
  still imports and runs; only the internal behavior should be subtly wrong.
- Each variant must be BEHAVIORALLY DIFFERENT from the correct code on some input
  (off-by-one, wrong operator/comparison, dropped edge case, swapped branch, wrong
  default, etc.). Do NOT just reformat or rename - the result must compute a wrong
  answer somewhere.
- Each variant must be a COMPLETE replacement for the target source region shown.
- Output ONE fenced code block per variant, nothing else (no prose between blocks).
"""


_KILL_SYSTEM_PROMPT = """You strengthen a hidden test suite so it rejects a WRONG
implementation. You are shown the GOLD (correct) source and a WRONG variant that
the current tests FAILED to reject (it still passed). Write ONE new test file whose
assertions PASS on the correct (gold) code but FAIL on the wrong variant.

Rules:
- Assert RUNTIME behavior: import/require the target module, call the target
  symbol(s), and assert the CORRECT results for inputs where the wrong variant
  differs from gold. Do NOT read source files and assert on their text.
- The test MUST pass on the correct (gold) code shown.
- Output ONLY the complete test file inside a single fenced code block. No prose.
"""


def _extract_blocks(text: str) -> list[str]:
    return [m.strip("\n") for m in _CODE_FENCE_RE.findall(text) if m.strip()]


def _extract_code(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _implements_target(symbol: str, content: str) -> bool:
    """Reject fenced prose or a replacement that omits the published target."""
    return bool(content.strip()) and (not symbol or symbol in content)


def _kill_test_path(language: str, round_index: int) -> str:
    if language == "go":
        return f"swe_forge_diff_{round_index}_test.go"
    if language == "javascript":
        return f"swe_forge_diff_{round_index}.test.js"
    return f"test_swe_forge_diff_{round_index}.py"


class TeacherVariantGenerator:
    """Proposes plausible-but-wrong variants of the gold target via the teacher."""

    def __init__(
        self,
        client: TeacherClient | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self.teacher_calls: list[TeacherGateCallEvidence] = []

    def _resolve_client(self) -> TeacherClient:
        if self._client is None:
            self._client = TeacherClient.from_settings(max_tokens=self._max_tokens)
        return self._client

    async def __call__(self, ctx: VariantGenerationContext) -> list[Variant]:
        self.teacher_calls = []
        teacher_source = select_teacher_source(ctx.candidate, ctx.gold_sources)
        if teacher_source is None:
            self.teacher_calls.append(
                TeacherGateCallEvidence(
                    gate="differential",
                    call_kind="proposal",
                    real_teacher=False,
                    status="not_called",
                    response_kind="source_unavailable",
                    requested_proposals=ctx.num_variants,
                    invalid_proposals=1,
                )
            )
            return []
        try:
            result = await self._resolve_client().complete_text(
                self._user_message(ctx, teacher_source),
                system=_VARIANT_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/endpoint failures
            self.teacher_calls.append(
                TeacherGateCallEvidence(
                    gate="differential",
                    call_kind="proposal",
                    real_teacher=True,
                    status="error",
                    response_kind="error",
                    model=_teacher_model(self._resolve_client()),
                    requested_proposals=ctx.num_variants,
                    error_type=type(exc).__name__,
                )
            )
            logger.warning("variant generation aborted (%s)", type(exc).__name__)
            return []

        variants: list[Variant] = []
        raw_blocks = _CODE_FENCE_RE.findall(result.text)
        response_kind = "content"
        if not result.text.strip():
            response_kind = "empty"
        elif not raw_blocks:
            response_kind = "unparseable"
        identical = 0
        invalid = 0
        parsed = 0
        for index, raw_block in enumerate(raw_blocks[: ctx.num_variants]):
            block = raw_block.strip("\n")
            if not _implements_target(required_symbol(teacher_source), block):
                invalid += 1
                continue
            parsed += 1
            materialized = teacher_source.materialize(block)
            if materialized == ctx.gold_sources[teacher_source.path]:
                identical += 1
                continue
            variants.append(
                Variant(
                    variant_id=f"variant_{index + 1}",
                    files=(
                        VariantFile(path=teacher_source.path, content=materialized),
                    ),
                    description=f"teacher plausible-wrong variant #{index + 1}",
                )
            )
        if parsed and parsed == identical:
            response_kind = "identical"
        elif raw_blocks and not parsed:
            response_kind = "invalid"
        self.teacher_calls.append(
            _proposal_evidence(
                "differential",
                result,
                model=_teacher_model(self._resolve_client()),
                response_kind=response_kind,
                requested=ctx.num_variants,
                received=len(raw_blocks[: ctx.num_variants]),
                parsed=parsed,
                identical=identical,
                invalid=invalid,
                discarded=0,
            )
        )
        return variants

    @property
    def last_call(self) -> TeacherGateCallEvidence | None:
        """Most recent call evidence, useful to diagnostic CLI/test consumers."""
        return self.teacher_calls[-1] if self.teacher_calls else None

    def _user_message(
        self, ctx: VariantGenerationContext, teacher_source: TeacherSource
    ) -> str:
        symbol = teacher_source.symbol or "(unspecified)"
        return (
            f"Language: {ctx.language}\n"
            f"Target file: {teacher_source.path}\n"
            f"Target symbol: {symbol}\n"
            f"Produce {ctx.num_variants} plausible-but-wrong replacements for "
            "this source region.\n\n"
            "Gold (correct) target source region:\n"
            f"```\n{teacher_source.source}\n```\n\n"
            f"Return {ctx.num_variants} fenced code blocks, one complete wrong "
            "replacement for the shown source region per block. Preserve its "
            "indentation and public interface."
        )


class DifferentialKillSynthesizer:
    """Proposes one test that separates a surviving variant from gold."""

    def __init__(
        self,
        client: TeacherClient | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self.teacher_calls: list[TeacherGateCallEvidence] = []

    def _resolve_client(self) -> TeacherClient:
        if self._client is None:
            self._client = TeacherClient.from_settings(max_tokens=self._max_tokens)
        return self._client

    async def __call__(self, ctx: DifferentialSynthesisContext) -> list[HiddenTest]:
        if not ctx.survivors:
            return []
        path = _kill_test_path(ctx.language, ctx.round_index)
        if path in ctx.existing_test_paths:
            path = _kill_test_path(ctx.language, ctx.round_index * 100 + 1)

        try:
            result = await self._resolve_client().complete_text(
                self._user_message(ctx, path),
                system=_KILL_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/endpoint failures
            self.teacher_calls.append(
                TeacherGateCallEvidence(
                    gate="differential",
                    call_kind="strengthen",
                    real_teacher=True,
                    status="error",
                    response_kind="error",
                    model=_teacher_model(self._resolve_client()),
                    requested_proposals=1,
                    error_type=type(exc).__name__,
                )
            )
            logger.warning(
                "differential-kill synthesis aborted (%s)", type(exc).__name__
            )
            return []

        content = _extract_code(result.text)
        self.teacher_calls.append(
            _proposal_evidence(
                "differential",
                result,
                model=_teacher_model(self._resolve_client()),
                response_kind=(
                    "empty"
                    if not result.text.strip()
                    else "content"
                    if _CODE_FENCE_RE.search(result.text)
                    else "unparseable"
                ),
                requested=1,
                received=1 if _CODE_FENCE_RE.search(result.text) else 0,
                parsed=1 if content.strip() else 0,
                call_kind="strengthen",
                invalid=0 if content.strip() else 1,
                discarded=0 if content.strip() else 1,
            )
        )
        if not content.strip():
            return []
        test_id = ctx.adapter.test_command((path,))
        return [
            HiddenTest(
                test_id=test_id,
                files=(HiddenTestFile(path=path, content=content + "\n"),),
                origin="synthesized",
            )
        ]

    def _user_message(self, ctx: DifferentialSynthesisContext, path: str) -> str:
        symbols = ", ".join(ctx.candidate.target.symbols) or "(unspecified)"
        gold = "\n\n".join(
            f"### {name}\n```\n{body}\n```" for name, body in ctx.gold_sources.items()
        )
        survivor = ctx.survivors[0]
        variant_files = "\n\n".join(
            f"### {f.path}\n```\n{f.content}\n```" for f in survivor.files
        )
        return (
            f"Language: {ctx.language}\n"
            f"Target symbol(s): {symbols}\n"
            f"Write the new test file at path: {path}\n\n"
            "Gold (correct) source:\n"
            f"{gold}\n\n"
            "WRONG variant the current tests did NOT reject:\n"
            f"{variant_files}\n\n"
            "Write ONE test that PASSES on gold and FAILS on the wrong variant. "
            "Return ONLY the complete test file in one fenced code block."
        )

    @property
    def last_call(self) -> TeacherGateCallEvidence | None:
        """Most recent call evidence, useful to diagnostic CLI/test consumers."""
        return self.teacher_calls[-1] if self.teacher_calls else None


def _proposal_evidence(
    gate: str,
    result: LLMResult,
    *,
    model: str,
    response_kind: str,
    requested: int,
    received: int,
    parsed: int,
    call_kind: str = "proposal",
    identical: int = 0,
    invalid: int = 0,
    discarded: int = 0,
) -> TeacherGateCallEvidence:
    """Build a source-free metadata record from a teacher result."""
    return TeacherGateCallEvidence(
        gate=gate,
        call_kind=call_kind,
        real_teacher=True,
        status="success",
        response_kind=response_kind,
        model=model,
        usage=getattr(result, "usage", Usage()),
        cost=float(getattr(result, "cost", 0.0) or 0.0),
        finish_reason=getattr(result, "finish_reason", None),
        requested_proposals=requested,
        received_proposals=received,
        parsed_proposals=parsed,
        identical_proposals=identical,
        invalid_proposals=invalid,
        discarded_proposals=discarded,
    )


def _teacher_model(client: object) -> str:
    """Read a model id without making lightweight parser fakes implement it."""
    model = getattr(client, "model", "")
    return model.strip() if isinstance(model, str) else ""


__all__ = ["DifferentialKillSynthesizer", "TeacherVariantGenerator"]
