"""Agentic hidden-test synthesizer, rewired onto :mod:`swe_forge.forge.teacher`.

The establish gate needs *discriminating* hidden tests when a manufactured fault
is not already covered: tests that FAIL on the broken tree and PASS on the gold
tree. This adapts the repository's agentic test-generator pattern (explore the
broken repo, write a behavioral test, iterate) but routes every model call through
the env-driven LiteLLM teacher (:class:`swe_forge.forge.teacher.TeacherClient`) -
it never imports the bespoke ``swe_forge.llm.*`` clients or any response cache, so
synthesis stays uncached and provider-agnostic.

Governing principle: *the teacher proposes, deterministic execution disposes*.
This module only PROPOSES candidate tests; the establish gate confirms each one
fail-on-broken / pass-on-gold in Docker before recording it. A proposal that
merely greps source (a string-matching pseudo-test) is rejected here so the
teacher is pushed toward genuine behavioral assertions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import PurePosixPath
from typing import Any

from swe_forge.forge.oracle.establish import (
    HiddenTest,
    HiddenTestFile,
    SynthesisContext,
)
from swe_forge.forge.teacher import NormalizedToolCall, TeacherClient

logger = getLogger(__name__)

DEFAULT_MAX_TURNS = 14
DEFAULT_MAX_PROPOSALS = 3
DEFAULT_MAX_TOKENS = 2048
_OUTPUT_LIMIT = 4000

_SYSTEM_PROMPT = """You write ONE behavioral regression test that catches a hidden bug.

The repository at the current working directory contains a BUG (a subtle code
fault has been introduced). Your job: write a test that FAILS now (on this buggy
code) and that WILL PASS once the bug is fixed to the intended behavior.

Rules:
- Explore with the `shell` and `read_file` tools to understand the intended
  behavior of the target symbol(s). Run the existing suite if helpful.
- The test MUST exercise RUNTIME behavior: import/require the module, call the
  function, assert the CORRECT (intended) result. Do NOT read source files and
  assert on their text (string-matching tests are rejected).
- Keep it small and focused on the one faulty behavior.
- Use `write_file` to create the test, then run it with `shell` to confirm it
  FAILS on the current buggy code.
- When the test fails for the RIGHT reason, call `submit_test` with its path.
- Submit at most a couple of tests. Stop once you have a failing behavioral test.
"""

# Minimal source-reading anti-patterns (a string-matching pseudo-test is no use:
# it would pass/fail on file text, not on the fixed behavior).
_STRING_MATCH_RE = re.compile(
    r"open\([^)]*\)\.read|read_text\(|readFileSync\(|\.read\(\)\s*\)?\s*;?\s*"
    r"(assert|expect)",
    re.IGNORECASE,
)


def _truncate(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    return text[:head] + "\n...[truncated]...\n" + text[-(limit - head) :]


def _is_safe_rel_path(path: str) -> bool:
    if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
        return False
    return True


def _looks_like_string_matching(content: str) -> bool:
    return bool(_STRING_MATCH_RE.search(content))


def _shell_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the repo working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run."}
                },
                "required": ["command"],
            },
        },
    }


def _read_file_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a repo file (relative path).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }


def _write_file_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a test file (relative path).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }


def _submit_test_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_test",
            "description": (
                "Submit the path of a test file (already written) that FAILS on "
                "the current buggy code and will PASS once fixed."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }


@dataclass
class _SynthSession:
    """Mutable per-run state: files written + tests submitted."""

    written: dict[str, str] = field(default_factory=dict)
    submitted: list[str] = field(default_factory=list)


class AgenticTestSynthesizer:
    """Proposes discriminating hidden tests via the LiteLLM teacher agentic loop.

    Implements the :class:`~swe_forge.forge.oracle.establish.HiddenTestSynthesizer`
    protocol. Runs a bounded multi-turn exchange (``shell`` / ``read_file`` /
    ``write_file`` / ``submit_test`` tools) against the broken tree, then returns
    the submitted tests as proposals - the establish gate confirms each one.
    """

    def __init__(
        self,
        client: TeacherClient | None = None,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_proposals: int = DEFAULT_MAX_PROPOSALS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self._max_turns = max_turns
        self._max_proposals = max_proposals
        self._max_tokens = max_tokens

    def _resolve_client(self) -> TeacherClient:
        if self._client is None:
            self._client = TeacherClient.from_settings(max_tokens=self._max_tokens)
        return self._client

    async def __call__(self, ctx: SynthesisContext) -> list[HiddenTest]:
        client = self._resolve_client()
        sandbox = ctx.recipe.sandbox
        session = _SynthSession()

        async def execute(call: NormalizedToolCall) -> str:
            return await self._dispatch(call, sandbox, session)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_message(ctx)},
        ]
        tools = [
            _shell_tool(),
            _read_file_tool(),
            _write_file_tool(),
            _submit_test_tool(),
        ]
        try:
            await client.agentic_turn(
                messages,
                tools,
                execute,
                max_turns=self._max_turns,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/endpoint failures
            logger.warning("hidden-test synthesis aborted: %s", exc)

        return self._collect(ctx, session)

    def _user_message(self, ctx: SynthesisContext) -> str:
        files = "\n".join(f"- {p}" for p in ctx.target_files) or "- (unknown)"
        symbols = ", ".join(ctx.candidate.target.symbols) or "(unspecified)"
        return (
            f"Language: {ctx.language}\n"
            f"Suspect source file(s):\n{files}\n"
            f"Suspect symbol(s): {symbols}\n\n"
            "Find the intended behavior and write a failing behavioral test."
        )

    async def _dispatch(
        self, call: NormalizedToolCall, sandbox: Any, session: _SynthSession
    ) -> str:
        name = call.name
        args = call.arguments if isinstance(call.arguments, dict) else {}
        if name == "shell":
            return await self._do_shell(sandbox, str(args.get("command", "")))
        if name == "read_file":
            return await self._do_read(sandbox, str(args.get("path", "")))
        if name == "write_file":
            return await self._do_write(
                sandbox,
                str(args.get("path", "")),
                str(args.get("content", "")),
                session,
            )
        if name == "submit_test":
            return self._do_submit(str(args.get("path", "")), session)
        return f"unknown tool: {name}"

    async def _do_shell(self, sandbox: Any, command: str) -> str:
        if not command.strip():
            return "error: empty command"
        try:
            result = await sandbox.run_command(command, timeout=180.0)
        except Exception as exc:
            return f"error running command: {exc}"
        return _truncate(
            f"exit={result.exit_code}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    async def _do_read(self, sandbox: Any, path: str) -> str:
        if not _is_safe_rel_path(path):
            return f"error: unsafe path {path!r}"
        try:
            content = await sandbox.read_file(path)
        except Exception as exc:
            return f"error reading {path}: {exc}"
        return _truncate(content)

    async def _do_write(
        self, sandbox: Any, path: str, content: str, session: _SynthSession
    ) -> str:
        if not _is_safe_rel_path(path):
            return f"error: unsafe path {path!r}"
        if not content.strip():
            return "error: empty content"
        try:
            await sandbox.write_file(path, content)
        except Exception as exc:
            return f"error writing {path}: {exc}"
        session.written[path] = content
        return f"wrote {path} ({len(content)} bytes)"

    def _do_submit(self, path: str, session: _SynthSession) -> str:
        if path not in session.written:
            return f"error: {path!r} was not written with write_file first"
        if _looks_like_string_matching(session.written[path]):
            return (
                "rejected: this test reads source text instead of asserting runtime "
                "behavior; import the module and assert the corrected result"
            )
        if path not in session.submitted:
            session.submitted.append(path)
        if len(session.submitted) >= self._max_proposals:
            return f"submitted {path}; proposal limit reached, stop now"
        return f"submitted {path}"

    def _collect(
        self, ctx: SynthesisContext, session: _SynthSession
    ) -> list[HiddenTest]:
        proposals: list[HiddenTest] = []
        for path in session.submitted[: self._max_proposals]:
            content = session.written.get(path, "")
            if not content:
                continue
            test_id = ctx.adapter.test_command((path,))
            proposals.append(
                HiddenTest(
                    test_id=test_id,
                    files=(HiddenTestFile(path=path, content=content),),
                    origin="synthesized",
                )
            )
        return proposals


__all__ = ["AgenticTestSynthesizer"]
