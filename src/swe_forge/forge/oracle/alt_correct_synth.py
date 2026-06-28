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
from swe_forge.forge.teacher import TeacherClient

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

    def _resolve_client(self) -> TeacherClient:
        if self._client is None:
            self._client = TeacherClient.from_settings(max_tokens=self._max_tokens)
        return self._client

    async def __call__(self, ctx: AltCorrectGenerationContext) -> list[AltImpl]:
        target = _primary_target(ctx.candidate.target.files, ctx.gold_sources)
        if not target:
            return []
        gold = ctx.gold_sources.get(target, "")
        if not gold.strip():
            return []

        try:
            result = await self._resolve_client().complete_text(
                self._user_message(ctx, target, gold),
                system=_ALT_CORRECT_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/endpoint failures
            logger.warning("alt-correct generation aborted: %s", exc)
            return []

        alternatives: list[AltImpl] = []
        blocks = _extract_blocks(result.text)[: ctx.num_alternatives]
        for index, block in enumerate(blocks):
            if block == gold.strip():
                continue
            alternatives.append(
                AltImpl(
                    impl_id=f"alt_{index + 1}",
                    files=(AltImplFile(path=target, content=block + "\n"),),
                    description=f"teacher genuinely-correct alternative #{index + 1}",
                )
            )
        return alternatives

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


__all__ = ["TeacherAltCorrectGenerator"]
