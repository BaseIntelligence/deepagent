"""Teacher-backed genuinely-correct alternative-implementation generation.

The alt-correct false-negative gate needs one teacher-proposed input (execution
disposes of it): a set of **genuinely-correct alternative** implementations of
the gold code (:class:`TeacherAltCorrectGenerator`) - same public Interface, but
different internal style and private symbol names. Every model call routes
through the env-driven LiteLLM teacher
(:class:`swe_forge.forge.teacher.TeacherClient`) - never the bespoke
``swe_forge.llm.*`` clients or any response cache.

Governing principle: *the teacher proposes, deterministic execution disposes*.
The alternatives here are only PROPOSALS; the gate runs each through the Docker
F2P+P2P suite and a correct alternative that the suite FAILS is treated as
evidence the suite is over-fit (a false-negative) - the gate then relaxes the
offending test(s) or rejects the candidate.
"""

from __future__ import annotations

import re
from logging import getLogger

from swe_forge.forge.oracle.alt_correct import (
    AltCorrectGenerationContext,
    AltImpl,
    AltImplFile,
)
from swe_forge.forge.oracle.teacher_evidence import TeacherGateCallEvidence
from swe_forge.forge.teacher import LLMResult, TeacherClient, Usage

logger = getLogger(__name__)

DEFAULT_MAX_TOKENS = 3072
_CODE_FENCE_RE = re.compile(r"```(?:[\w.+-]*)\n(.*?)```", re.DOTALL)


_ALT_CORRECT_SYSTEM_PROMPT = """You produce GENUINELY-CORRECT alternative
implementations of some correct code, to test that a hidden test suite is not
over-fit. The code you are shown is CORRECT (the gold implementation). Your job:
rewrite the SAME file in one or more different ways that are each FULLY CORRECT -
behaviorally identical to the gold code on every input - but written in a
DIFFERENT internal style.

Rules:
- Keep the SAME public interface: the exact public symbol names and signatures
  listed in the Interface block (and the public symbols of the gold file) must be
  preserved, so the alternative is a drop-in replacement that imports and runs.
- Change the INTERNALS: use different private/local variable and helper names, a
  different but equivalent algorithm or control flow, different formatting. The
  result MUST compute the SAME answer as the gold code for all inputs.
- Do NOT introduce any behavioral difference, bug, or edge-case regression. These
  are CORRECT alternatives, not wrong variants.
- Each alternative must be a COMPLETE drop-in replacement for the whole file shown.
- Output ONE fenced code block per alternative, nothing else (no prose between
  blocks).
"""


def _extract_blocks(text: str) -> list[str]:
    return [m.strip() for m in _CODE_FENCE_RE.findall(text) if m.strip()]


def _primary_target(ctx_files: tuple[str, ...], gold_sources: dict[str, str]) -> str:
    """Pick the source file the alternatives overwrite (a real, readable target)."""
    for path in ctx_files:
        if path in gold_sources and gold_sources[path].strip():
            return path
    if ctx_files:
        return ctx_files[0]
    if gold_sources:
        return next(iter(gold_sources))
    return ""


def _implements_target(ctx: AltCorrectGenerationContext, content: str) -> bool:
    """Reject fenced prose or a replacement that omits the published target."""
    required = {
        part
        for symbol in ctx.candidate.target.symbols
        for part in symbol.split(".")
        if part and part != "__init__"
    }
    return bool(content.strip()) and all(name in content for name in required)


class TeacherAltCorrectGenerator:
    """Proposes genuinely-correct alternatives of the gold target via the teacher."""

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

    async def __call__(self, ctx: AltCorrectGenerationContext) -> list[AltImpl]:
        self.teacher_calls = []
        target = _primary_target(ctx.candidate.target.files, ctx.gold_sources)
        if not target:
            self.teacher_calls.append(
                TeacherGateCallEvidence(
                    gate="alt_correct",
                    call_kind="proposal",
                    real_teacher=False,
                    status="not_called",
                    response_kind="source_unavailable",
                    requested_proposals=ctx.num_alternatives,
                    invalid_proposals=1,
                )
            )
            return []
        gold = ctx.gold_sources.get(target, "")
        if not gold.strip():
            self.teacher_calls.append(
                TeacherGateCallEvidence(
                    gate="alt_correct",
                    call_kind="proposal",
                    real_teacher=False,
                    status="not_called",
                    response_kind="source_unavailable",
                    requested_proposals=ctx.num_alternatives,
                    invalid_proposals=1,
                )
            )
            return []

        try:
            result = await self._resolve_client().complete_text(
                self._user_message(ctx, target, gold),
                system=_ALT_CORRECT_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/endpoint failures
            self.teacher_calls.append(
                TeacherGateCallEvidence(
                    gate="alt_correct",
                    call_kind="proposal",
                    real_teacher=True,
                    status="error",
                    response_kind="error",
                    model=_teacher_model(self._resolve_client()),
                    requested_proposals=ctx.num_alternatives,
                    error_type=type(exc).__name__,
                )
            )
            logger.warning("alt-correct generation aborted (%s)", type(exc).__name__)
            return []

        alternatives: list[AltImpl] = []
        raw_blocks = _CODE_FENCE_RE.findall(result.text)
        response_kind = "content"
        if not result.text.strip():
            response_kind = "empty"
        elif not raw_blocks:
            response_kind = "unparseable"
        parsed = 0
        identical = 0
        invalid = 0
        for index, raw_block in enumerate(raw_blocks[: ctx.num_alternatives]):
            block = raw_block.strip()
            if not block:
                invalid += 1
                continue
            if not _implements_target(ctx, block):
                invalid += 1
                continue
            parsed += 1
            if block == gold.strip():
                identical += 1
                continue
            alternatives.append(
                AltImpl(
                    impl_id=f"alt_{index + 1}",
                    files=(AltImplFile(path=target, content=block + "\n"),),
                    description=f"teacher genuinely-correct alternative #{index + 1}",
                )
            )
        if parsed and parsed == identical:
            response_kind = "identical"
        elif raw_blocks and not parsed:
            response_kind = "invalid"
        self.teacher_calls.append(
            _proposal_evidence(
                result,
                model=_teacher_model(self._resolve_client()),
                response_kind=response_kind,
                requested=ctx.num_alternatives,
                received=len(raw_blocks[: ctx.num_alternatives]),
                parsed=parsed,
                identical=identical,
                invalid=invalid,
                discarded=0,
            )
        )
        return alternatives

    @property
    def last_call(self) -> TeacherGateCallEvidence | None:
        """Most recent call evidence, useful to diagnostic CLI/test consumers."""
        return self.teacher_calls[-1] if self.teacher_calls else None

    def _user_message(
        self, ctx: AltCorrectGenerationContext, target: str, gold: str
    ) -> str:
        symbols = ", ".join(ctx.candidate.target.symbols) or "(unspecified)"
        interface = ctx.interface_block.strip()
        interface_section = (
            f"Interface (public symbols/signatures to preserve EXACTLY):\n"
            f"{interface}\n\n"
            if interface
            else ""
        )
        return (
            f"Language: {ctx.language}\n"
            f"Target file: {target}\n"
            f"Target symbol(s): {symbols}\n"
            f"Produce {ctx.num_alternatives} genuinely-correct alternative "
            "implementations of this file.\n\n"
            f"{interface_section}"
            "Gold (correct) source:\n"
            f"```\n{gold}\n```\n\n"
            f"Return {ctx.num_alternatives} fenced code blocks, one complete, "
            "fully-correct alternative version of the file per block (same public "
            "interface, different internals)."
        )


def _proposal_evidence(
    result: LLMResult,
    *,
    model: str,
    response_kind: str,
    requested: int,
    received: int,
    parsed: int,
    identical: int = 0,
    invalid: int = 0,
    discarded: int = 0,
) -> TeacherGateCallEvidence:
    """Build a source-free metadata record from a teacher result."""
    return TeacherGateCallEvidence(
        gate="alt_correct",
        call_kind="proposal",
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


__all__ = ["TeacherAltCorrectGenerator"]
