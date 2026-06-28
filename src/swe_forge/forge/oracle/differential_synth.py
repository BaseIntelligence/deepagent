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
from swe_forge.forge.teacher import TeacherClient

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
- Each variant must be a COMPLETE drop-in replacement for the whole file shown.
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
    return [m.strip() for m in _CODE_FENCE_RE.findall(text) if m.strip()]


def _extract_code(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _primary_target(ctx_files: tuple[str, ...], gold_sources: dict[str, str]) -> str:
    """Pick the source file the variants overwrite (a real, readable target)."""
    for path in ctx_files:
        if path in gold_sources and gold_sources[path].strip():
            return path
    if ctx_files:
        return ctx_files[0]
    if gold_sources:
        return next(iter(gold_sources))
    return ""


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

    def _resolve_client(self) -> TeacherClient:
        if self._client is None:
            self._client = TeacherClient.from_settings(max_tokens=self._max_tokens)
        return self._client

    async def __call__(self, ctx: VariantGenerationContext) -> list[Variant]:
        target = _primary_target(ctx.candidate.target.files, ctx.gold_sources)
        if not target:
            return []
        gold = ctx.gold_sources.get(target, "")
        if not gold.strip():
            return []

        try:
            result = await self._resolve_client().complete_text(
                self._user_message(ctx, target, gold),
                system=_VARIANT_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/endpoint failures
            logger.warning("variant generation aborted: %s", exc)
            return []

        variants: list[Variant] = []
        for index, block in enumerate(_extract_blocks(result.text)[: ctx.num_variants]):
            if block == gold.strip():
                continue
            variants.append(
                Variant(
                    variant_id=f"variant_{index + 1}",
                    files=(VariantFile(path=target, content=block + "\n"),),
                    description=f"teacher plausible-wrong variant #{index + 1}",
                )
            )
        return variants

    def _user_message(
        self, ctx: VariantGenerationContext, target: str, gold: str
    ) -> str:
        symbols = ", ".join(ctx.candidate.target.symbols) or "(unspecified)"
        return (
            f"Language: {ctx.language}\n"
            f"Target file: {target}\n"
            f"Target symbol(s): {symbols}\n"
            f"Produce {ctx.num_variants} plausible-but-wrong variants of this file.\n\n"
            "Gold (correct) source:\n"
            f"```\n{gold}\n```\n\n"
            f"Return {ctx.num_variants} fenced code blocks, one complete wrong "
            "version of the file per block."
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
            logger.warning("differential-kill synthesis aborted: %s", exc)
            return []

        content = _extract_code(result.text)
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


__all__ = ["DifferentialKillSynthesizer", "TeacherVariantGenerator"]
