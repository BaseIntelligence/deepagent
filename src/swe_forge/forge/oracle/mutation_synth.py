"""Teacher-backed synthesizer of survivor-killing tests for the mutation gate.

When the mutation-adequacy gate finds the hidden suite under-determined (mutants
survive), it asks a :class:`MutationTestSynthesizer` for extra behavioral tests
that pin the gold behavior down tightly enough to catch those mutants. This
implementation routes every model call through the env-driven LiteLLM teacher
(:class:`swe_forge.forge.teacher.TeacherClient`) - never the bespoke
``swe_forge.llm.*`` clients or any response cache.

Governing principle: *the teacher proposes, deterministic execution disposes*.
This module only PROPOSES a candidate test per round; the gate confirms it by
re-running the mutation tool and keeps it only if it actually reduces the
surviving-mutant count.
"""

from __future__ import annotations

import re
from logging import getLogger

from swe_forge.forge.oracle.mutation import (
    HiddenTest,
    HiddenTestFile,
    MutationSynthesisContext,
)
from swe_forge.forge.teacher import TeacherClient

logger = getLogger(__name__)

DEFAULT_MAX_TOKENS = 2048
_CODE_FENCE_RE = re.compile(r"```(?:[\w.+-]*)\n(.*?)```", re.DOTALL)

_SYSTEM_PROMPT = """You strengthen a hidden test suite so it pins down the exact
intended behavior of some code. The code under test is CORRECT (this is the gold
implementation). A mutation tool injected small faults ("mutants") that the
current tests FAILED to catch (they still passed). Write ONE new test file whose
assertions would FAIL if the code were subtly wrong, yet PASS on the correct code
shown to you.

Rules:
- Assert RUNTIME behavior: import/require the target module, call the target
  symbol(s), and assert the correct results for several inputs (edge cases,
  boundaries, operator/branch behavior). Do NOT read source files and assert on
  their text.
- Cover the behaviors the surviving mutants would have changed (e.g. off-by-one,
  swapped operator, removed branch).
- The test MUST pass on the correct code shown.
- Output ONLY the complete test file inside a single fenced code block. No prose.
"""


def _extract_code(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _test_path(language: str, round_index: int) -> str:
    if language == "go":
        return f"swe_forge_mut_{round_index}_test.go"
    if language == "javascript":
        return f"swe_forge_mut_{round_index}.test.js"
    return f"test_swe_forge_mut_{round_index}.py"


class MutationKillSynthesizer:
    """Proposes one survivor-killing test per round via the LiteLLM teacher."""

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

    async def __call__(self, ctx: MutationSynthesisContext) -> list[HiddenTest]:
        path = _test_path(ctx.language, ctx.round_index)
        if path in ctx.existing_test_paths:
            path = _test_path(ctx.language, ctx.round_index * 100 + 1)

        try:
            result = await self._resolve_client().complete_text(
                self._user_message(ctx, path),
                system=_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/endpoint failures
            logger.warning("mutation-kill synthesis aborted: %s", exc)
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

    def _user_message(self, ctx: MutationSynthesisContext, path: str) -> str:
        symbols = ", ".join(ctx.candidate.target.symbols) or "(unspecified)"
        survivors = (
            "\n".join(f"- {s}" for s in ctx.survivors[:20])
            or "- (none reported; strengthen coverage of the target behavior)"
        )
        sources = "\n\n".join(
            f"### {name}\n```\n{body}\n```" for name, body in ctx.sources.items()
        )
        return (
            f"Language: {ctx.language}\n"
            f"Target symbol(s): {symbols}\n"
            f"Write the new test file at path: {path}\n\n"
            "Gold (correct) source under test:\n"
            f"{sources}\n\n"
            "Surviving mutants the current tests did NOT catch:\n"
            f"{survivors}\n\n"
            "Return ONLY the complete test file in one fenced code block."
        )


__all__ = ["MutationKillSynthesizer"]
