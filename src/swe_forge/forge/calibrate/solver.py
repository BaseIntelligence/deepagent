"""Agentic solver scaffold for difficulty calibration (architecture S6, Stage 4).

Calibration measures how hard a manufactured task is by letting a *panel* of
solver models attempt it and scoring how often they succeed. This module is the
per-rollout scaffold the panel runner drives: given the **broken** tree (the
candidate's forward ``mutation_patch`` applied) plus the agent-facing
:class:`~swe_forge.forge.models.GeneratedSpec` (problem statement + requirements
+ interface block, and *nothing else* - never the hidden tests or the gold
``oracle_patch``), it runs a bounded tool loop (``shell`` / ``read_file`` /
``write_file`` / ``finish``) through the env-driven LiteLLM teacher/panel surface
(:mod:`swe_forge.forge.teacher`) and captures the unified-diff patch the model
submits via ``finish``.

Each submitted patch is then scored through the SAME Docker FAIL->PASS primitives
the oracle gates use (:class:`~swe_forge.forge.oracle.establish.DockerOracleRecipe`,
``reconstruct_suite_tests``) so a "solve" means exactly the same thing everywhere:
the patch applies to the broken tree AND every hidden test (the full
``OracleReport.test_files[]`` set, not just the original F2P) passes AND the
P2P/regression suite stays green.

Governing principles:

* **Submit-gated.** Only the patch the model submits via ``finish`` is scored;
  the working-tree state mid-rollout is never credited. A rollout that exhausts
  its turn budget without finishing, submits an empty diff, or whose patch fails
  to apply records ``solve = false`` gracefully (it never crashes the runner).
* **No leak.** The solver only ever sees the broken source tree and the spec
  surface; the hidden test bodies and the ``oracle_patch`` are never written into
  the solver workspace nor placed in its prompt.
* **Reuse, don't rewrite.** Scoring goes through the shared Docker recipe, never
  an ad-hoc check, and never imports the bespoke ``swe_forge.llm.*`` clients or
  any response cache.
"""

from __future__ import annotations

import contextlib
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from swe_forge.artifacts import generated_artifact_gitignore_patterns
from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    GeneratedSpec,
    OracleReport,
    require_green_baseline,
)
from swe_forge.forge.oracle.differential import reconstruct_suite_tests
from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    HiddenTest,
    SandboxProtocol,
    TreeState,
)
from swe_forge.forge.teacher import NormalizedToolCall, TeacherClient, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from swe_forge.execution.docker_client import DockerClient

logger = getLogger(__name__)

#: Bounded budget for the solver tool loop (over-budget without ``finish`` =
#: a clean non-solve).
DEFAULT_MAX_TURNS = 20
DEFAULT_MAX_TOKENS = 4096
_OUTPUT_LIMIT = 6000

# Where the rollout/scoring stage their patches inside the workspace. Kept out of
# the repo's own test-discovery paths so a P2P run never collects them, and never
# committed into the broken baseline so a captured diff stays free of harness files.
_SOLVER_DIR = ".swe_forge_solver"
_MUTATION_PATCH_REL = f"{_SOLVER_DIR}/mutation.patch"
_SOLVER_PATCH_REL = f"{_SOLVER_DIR}/solver.patch"

# A throwaway git identity so the broken baseline can be committed (the captured
# diff is taken relative to it). Never a real user; only ever local to the
# throwaway container.
_GIT_IDENT = "-c user.email=forge@localhost -c user.name=forge"


def _gitignore_append_command() -> str:
    """Append the canonical generated-artifact ignores to ``.gitignore``.

    Runs before ``git add -A`` in the broken-baseline re-init so artifacts are
    never staged. ``printf`` creates ``.gitignore`` when absent; ``>>`` preserves
    any rules the repo already shipped; a leading newline keeps the first pattern
    on its own line even if an existing file lacked a trailing newline. Duplicate
    patterns (when the repo already ignores them) are harmless.
    """
    body = "\\n" + "\\n".join(generated_artifact_gitignore_patterns()) + "\\n"
    return f"printf '{body}' >> .gitignore"


# Non-solve reason keys (stable strings the runner/report may surface).
REASON_NOT_FINISHED = "not_finished"
REASON_EMPTY_PATCH = "empty_patch"
REASON_APPLY_FAILED = "apply_failed"
REASON_F2P_OR_P2P_FAILED = "f2p_or_p2p_failed"
REASON_ROLLOUT_ERROR = "rollout_error"
REASON_SCORE_ERROR = "score_error"

_SYSTEM_PROMPT = """You are an autonomous software engineer fixing a bug in a repository.

The repository at your current working directory contains a BUG: a function does
not behave as specified. Your job is to edit the source so the intended behavior
is restored.

Tools:
- `shell`: run a shell command in the repository working directory (explore, run
  the existing test suite, inspect files).
- `read_file`: read a repository file (relative path).
- `write_file`: create or overwrite a repository file (relative path) with new
  full content.
- `finish`: call this when your fix is complete to SUBMIT your patch for scoring.

Rules:
- Read the task statement, requirements, and interface carefully, then explore
  the code to find the function(s) that must be corrected.
- Implement the fix by editing the SOURCE files (not by editing or adding tests).
- Keep the public interface exactly as given (same symbol names and signatures).
- When, and only when, the behavior is correct, call `finish`. Only the code you
  have written when you call `finish` is submitted; nothing else is scored.
- If you do not call `finish`, no patch is submitted.
"""


class SolverError(RuntimeError):
    """Raised for an unrecoverable failure while driving a solver rollout."""


def _truncate(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    return text[:head] + "\n...[truncated]...\n" + text[-(limit - head) :]


def _is_safe_rel_path(path: str) -> bool:
    if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
        return False
    return True


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


# --------------------------------------------------------------------------- #
# Tool schemas
# --------------------------------------------------------------------------- #
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
            "description": "Create or overwrite a repo source file (relative path).",
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


def _finish_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Submit your completed fix for scoring. Call this only when the "
                "intended behavior is restored."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Optional short summary of the fix.",
                    }
                },
                "required": [],
            },
        },
    }


def solver_tools() -> list[dict[str, Any]]:
    """The four tools exposed to a solver rollout."""
    return [_shell_tool(), _read_file_tool(), _write_file_tool(), _finish_tool()]


def build_solver_prompt(spec: GeneratedSpec, language: str) -> str:
    """Render the agent-facing prompt from the spec surface ONLY.

    Includes the problem statement, requirements, and interface block - and
    nothing derived from the candidate's mutation/oracle patches or the hidden
    tests. This is the sole task description the solver receives (VAL-CAL-005).
    """
    requirements = "\n".join(f"{i}. {r}" for i, r in enumerate(spec.requirements, 1))
    return (
        f"Language: {language}\n\n"
        f"# Problem\n{spec.problem_statement}\n\n"
        f"# Requirements\n{requirements}\n\n"
        f"# Interface\n{spec.interface_block}\n\n"
        "Fix the source so all requirements hold, then call `finish`."
    )


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
@dataclass
class _SolverSession:
    """Mutable per-rollout state: files written, finished flag, captured patch."""

    written: dict[str, str] = field(default_factory=dict)
    finished: bool = False
    patch: str = ""


@dataclass
class SolverRollout:
    """The outcome of one solver rollout (before scoring).

    ``patch`` is the unified diff the model submitted via ``finish`` (empty when
    nothing was submitted or the submitted diff was empty). ``finished`` records
    whether ``finish`` was ever called (False = over-budget). ``usage``/``cost``
    are the LiteLLM token usage/cost accrued across the rollout's calls.
    """

    patch: str
    finished: bool
    turns: int
    usage: Usage
    cost: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "finished": self.finished,
            "turns": self.turns,
            "patch_bytes": len(self.patch),
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "error": self.error,
        }


@dataclass
class SolverContext:
    """Inputs handed to :meth:`AgenticSolver.solve`.

    ``sandbox`` is a running workspace whose tree is ALREADY broken (mutation
    applied) and committed as HEAD, so a captured diff reflects only the model's
    edits. The solver never receives the candidate's patches or the hidden tests.
    """

    spec: GeneratedSpec
    language: str
    sandbox: SandboxProtocol
    command_timeout: float = 600.0


class AgenticSolver:
    """Runs a bounded tool loop and captures the submitted unified-diff patch.

    Drives the env-driven LiteLLM teacher/panel surface
    (:meth:`swe_forge.forge.teacher.TeacherClient.agentic_turn`) with the
    ``shell`` / ``read_file`` / ``write_file`` / ``finish`` tools against the
    broken tree, then returns the diff the model submitted via ``finish``. It
    never imports the bespoke LLM clients or any response cache.
    """

    def __init__(
        self,
        client: TeacherClient | None = None,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self._max_turns = max_turns
        self._max_tokens = max_tokens

    def _resolve_client(self) -> TeacherClient:
        if self._client is None:
            self._client = TeacherClient.from_settings(max_tokens=self._max_tokens)
        return self._client

    async def solve(self, ctx: SolverContext) -> SolverRollout:
        client = self._resolve_client()
        sandbox = ctx.sandbox
        session = _SolverSession()

        async def execute(call: NormalizedToolCall) -> str:
            return await self._dispatch(call, sandbox, session, ctx.command_timeout)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_solver_prompt(ctx.spec, ctx.language)},
        ]
        error: str | None = None
        turns = 0
        usage = Usage()
        cost = 0.0
        try:
            result = await client.agentic_turn(
                messages,
                solver_tools(),
                execute,
                max_turns=self._max_turns,
                max_tokens=self._max_tokens,
            )
            turns = result.turns
            usage = result.usage
            cost = result.cost
        except Exception as exc:  # network/endpoint failure: clean non-solve
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("solver rollout aborted: %s", error)

        return SolverRollout(
            patch=session.patch if session.finished else "",
            finished=session.finished,
            turns=turns,
            usage=usage,
            cost=cost,
            error=error,
        )

    async def _dispatch(
        self,
        call: NormalizedToolCall,
        sandbox: SandboxProtocol,
        session: _SolverSession,
        timeout: float,
    ) -> str:
        name = call.name
        args = call.arguments if isinstance(call.arguments, dict) else {}
        if name == "shell":
            return await self._do_shell(sandbox, str(args.get("command", "")), timeout)
        if name == "read_file":
            return await self._do_read(sandbox, str(args.get("path", "")))
        if name == "write_file":
            return await self._do_write(
                sandbox,
                str(args.get("path", "")),
                str(args.get("content", "")),
                session,
            )
        if name == "finish":
            return await self._do_finish(sandbox, session, timeout)
        return f"unknown tool: {name}"

    async def _do_shell(
        self, sandbox: SandboxProtocol, command: str, timeout: float
    ) -> str:
        if not command.strip():
            return "error: empty command"
        try:
            result = await sandbox.run_command(command, timeout=timeout)
        except Exception as exc:
            return f"error running command: {exc}"
        return _truncate(
            f"exit={result.exit_code}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    async def _do_read(self, sandbox: SandboxProtocol, path: str) -> str:
        if not _is_safe_rel_path(path):
            return f"error: unsafe path {path!r}"
        try:
            content = await sandbox.read_file(path)
        except Exception as exc:
            return f"error reading {path}: {exc}"
        return _truncate(content)

    async def _do_write(
        self,
        sandbox: SandboxProtocol,
        path: str,
        content: str,
        session: _SolverSession,
    ) -> str:
        if not _is_safe_rel_path(path):
            return f"error: unsafe path {path!r}"
        try:
            await sandbox.write_file(path, content)
        except Exception as exc:
            return f"error writing {path}: {exc}"
        session.written[path] = content
        return f"wrote {path} ({len(content)} bytes)"

    async def _do_finish(
        self, sandbox: SandboxProtocol, session: _SolverSession, timeout: float
    ) -> str:
        patch = await capture_workspace_patch(sandbox, timeout=timeout)
        session.finished = True
        session.patch = patch
        if not patch.strip():
            return (
                "submitted: but the working tree has no changes relative to the "
                "starting (broken) code; the diff is empty"
            )
        return f"submitted patch ({len(patch)} bytes); scoring will use this diff"


async def capture_workspace_patch(
    sandbox: SandboxProtocol, *, timeout: float = 600.0
) -> str:
    """Capture the working tree's diff against HEAD as a unified diff.

    The broken baseline is committed as HEAD before the rollout (a single orphan
    root commit of the full source tree), so this returns exactly the model's
    edits to the existing source and never the mutation. The output applies
    cleanly with ``git apply`` at the repo root.

    Scope: only edits to tracked source files (``git add -u`` -- modifications and
    deletions of files that were in the broken baseline) are folded into the
    submitted patch. Stray scratch files the model created outside the source tree
    (scratch ``test_*`` files, debug logs, build output) are untracked, so they
    are deliberately excluded and only warned about; otherwise they would be
    materialized in the scorer container and perturb F2P/P2P collection.
    """
    others = await sandbox.run_command(
        "git ls-files --others --exclude-standard", timeout=timeout
    )
    if others.exit_code == 0:
        stray = sorted(line for line in others.stdout.splitlines() if line.strip())
        if stray:
            logger.warning(
                "solver wrote %d untracked file(s) outside the source tree; "
                "excluding them from the submitted patch: %s",
                len(stray),
                ", ".join(stray[:10]) + (" ..." if len(stray) > 10 else ""),
            )
    result = await sandbox.run_command(
        f"git {_GIT_IDENT} add -u && git {_GIT_IDENT} diff --cached HEAD",
        timeout=timeout,
    )
    if result.exit_code != 0:
        return ""
    return result.stdout


# --------------------------------------------------------------------------- #
# Scoring (shared Docker FAIL->PASS path)
# --------------------------------------------------------------------------- #
@dataclass
class SolveScore:
    """The verdict of scoring one submitted patch via the shared FAIL->PASS path.

    ``solved`` is True iff the patch applied to the broken tree AND every hidden
    test passed AND the P2P/regression suite stayed green. ``applied``/``empty``
    flag the degenerate non-solves (non-applying / empty diff). ``failing_test_ids``
    lists the hidden tests (and the P2P command) that failed.
    """

    solved: bool
    applied: bool
    empty: bool
    f2p_passed: bool
    p2p_passed: bool
    failing_test_ids: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "solved": self.solved,
            "applied": self.applied,
            "empty": self.empty,
            "f2p_passed": self.f2p_passed,
            "p2p_passed": self.p2p_passed,
            "failing_test_ids": list(self.failing_test_ids),
            "reason": self.reason,
        }


def _non_solve(
    reason: str, *, empty: bool = False, applied: bool = False
) -> SolveScore:
    return SolveScore(
        solved=False,
        applied=applied,
        empty=empty,
        f2p_passed=False,
        p2p_passed=False,
        reason=reason,
    )


async def _apply_solver_patch(
    recipe: DockerOracleRecipe, patch: str, *, timeout: float
) -> bool:
    """Stage and ``git apply`` the submitted patch onto the broken tree."""
    await recipe.sandbox.write_file(_SOLVER_PATCH_REL, _ensure_trailing_newline(patch))
    rel = shlex.quote(_SOLVER_PATCH_REL)
    primary = await recipe.sandbox.run_command(
        f"git apply --whitespace=nowarn {rel}", timeout=timeout
    )
    if primary.exit_code == 0:
        return True
    fallback = await recipe.sandbox.run_command(
        f"git apply --3way --whitespace=nowarn {rel}", timeout=timeout
    )
    return fallback.exit_code == 0


async def score_patch(
    recipe: DockerOracleRecipe,
    patch: str,
    base_tests: Sequence[HiddenTest],
    *,
    timeout: float = 600.0,
) -> SolveScore:
    """Score ``patch`` on the broken tree through the shared Docker recipe.

    Applies the candidate's forward mutation (``set_state(BROKEN)``), applies the
    submitted patch on top, runs the P2P/regression suite with NO hidden test
    present, then runs every hidden test (the full reconstructed suite). A solve
    requires the patch to apply AND all hidden tests to pass AND P2P to be green.
    """
    if not patch.strip():
        return _non_solve(REASON_EMPTY_PATCH, empty=True)

    await recipe.set_state(TreeState.BROKEN)
    applied = await _apply_solver_patch(recipe, patch, timeout=timeout)
    if not applied:
        return _non_solve(REASON_APPLY_FAILED)

    # P2P/regression runs with no hidden test present (a hidden test must not make
    # the repo's own suite look red).
    p2p = await recipe.run_p2p()

    failing: list[str] = []
    f2p_passed = True
    for test in base_tests:
        await recipe.write_test(test)
        run = await recipe.run_test(test)
        await recipe.remove_test(test)
        if not run.passed:
            failing.append(test.test_id)
            f2p_passed = False
    if not p2p.passed:
        failing.append(recipe.p2p_command)

    solved = f2p_passed and p2p.passed
    return SolveScore(
        solved=solved,
        applied=True,
        empty=False,
        f2p_passed=f2p_passed,
        p2p_passed=p2p.passed,
        failing_test_ids=tuple(failing),
        reason="" if solved else REASON_F2P_OR_P2P_FAILED,
    )


class DockerPatchScorer:
    """Scores a submitted solver patch via throwaway Docker sandboxes.

    Mirrors the oracle gates' Docker runners: opens a fresh ``--rm``
    :class:`~swe_forge.execution.sandbox.DockerSandbox` on the candidate's green
    ``EnvImage``, drives the shared
    :class:`~swe_forge.forge.oracle.establish.DockerOracleRecipe` to set the
    broken tree, applies the patch, and runs the full hidden suite + P2P.
    """

    def __init__(
        self,
        candidate: Candidate,
        env_image: EnvImage,
        adapter: LanguageAdapter,
        *,
        base_tests: Sequence[HiddenTest],
        p2p_command: str = "",
        command_timeout: float = 600.0,
        docker_client: "DockerClient | None" = None,
    ) -> None:
        self._candidate = candidate
        self._env_image = env_image
        self._adapter = adapter
        self._base_tests = list(base_tests)
        self._p2p_command = p2p_command or env_image.baseline_test_command
        self._timeout = command_timeout
        self._docker_client = docker_client

    async def score(self, patch: str) -> SolveScore:
        if not patch.strip():
            return _non_solve(REASON_EMPTY_PATCH, empty=True)
        async with self._recipe() as recipe:
            return await score_patch(
                recipe, patch, self._base_tests, timeout=self._timeout
            )

    @contextlib.asynccontextmanager
    async def _recipe(self) -> "AsyncIterator[DockerOracleRecipe]":
        from swe_forge.execution.docker_client import DockerClient
        from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

        client = self._docker_client or DockerClient()
        config = SandboxConfig(
            name="swe-forge-cal-solver-score",
            image=self._env_image.image_tag,
            workspace_dir=self._env_image.workspace_dir,
            command_timeout=self._timeout,
        )
        sandbox = DockerSandbox(client, config)
        async with sandbox:
            yield DockerOracleRecipe(
                sandbox,
                language=self._candidate.language,
                workspace_dir=self._env_image.workspace_dir,
                mutation_patch=self._candidate.mutation_patch,
                oracle_patch=self._candidate.oracle_patch,
                p2p_command=self._p2p_command,
                command_timeout=self._timeout,
            )


# --------------------------------------------------------------------------- #
# Orchestration: one rollout end to end (rollout -> submit-gate -> score)
# --------------------------------------------------------------------------- #
@dataclass
class RolloutOutcome:
    """One rollout end to end: the captured patch plus its FAIL->PASS verdict.

    This is the unit the panel runner collects ``k`` times per model. ``solved``
    is the submit-gated verdict (a non-finished / empty / non-applying / failing
    rollout is a clean non-solve). ``usage``/``cost`` are the rollout's billed
    LiteLLM usage so the runner can aggregate per-model cost.
    """

    model: str
    patch: str
    finished: bool
    solved: bool
    score: SolveScore
    turns: int
    usage: Usage
    cost: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "finished": self.finished,
            "solved": self.solved,
            "patch_bytes": len(self.patch),
            "score": self.score.to_dict(),
            "turns": self.turns,
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "error": self.error,
        }


async def _setup_broken_baseline(
    sandbox: SandboxProtocol, candidate: Candidate, *, timeout: float
) -> None:
    """Apply the forward mutation and RE-INIT git so the broken tree is a single
    orphan/root commit (no recoverable gold ancestor).

    forge builds the broken tree as a synthetic commit ON TOP of the pinned GOLD
    commit, and the EnvImage keeps the repo's full ``.git`` history -- so an
    agent-facing rollout exposing an unrestricted ``shell`` could otherwise recover
    the gold/oracle source via ``git show HEAD~1`` / ``git diff HEAD~1 HEAD`` /
    ``git checkout HEAD~1 -- <path>`` / ``git show HEAD:<path>`` and submit a
    gold-restoring patch (AGENTS.md "No gold leak via .git history"; VAL-CAL-005).
    After staging the broken working tree we therefore delete ``.git`` entirely
    and re-init a fresh repo whose ONLY commit is the broken baseline: exactly one
    ``HEAD`` remains (so the ``git add -u && git diff --cached HEAD`` patch capture
    still works), ``git show HEAD~1`` fails, and gold is unrecoverable from history.

    This is applied ONLY here, in the agent-facing rollout container. The persisted
    EnvImage and the deterministic scorer (``DockerOracleRecipe``) are untouched --
    they legitimately keep gold history for oracle scoring.

    To keep that capture source-only even when the candidate repo ships no
    ``.gitignore``, the canonical generated-artifact policy (Python, JS/TS, Go,
    Java/Gradle, coverage, logs, and binary outputs) is appended BEFORE
    ``git add -A``. Thus artifacts an EnvImage's baseline run generated never
    enter the baseline commit.
    Otherwise a tracked ``.pyc`` the model regenerates by running code would fold a
    stale binary diff into the submitted patch and make the scorer's ``git apply``
    fail (a real solve then mis-scored ``apply_failed``).

    The captured rollout diff is taken relative to this commit, so it contains
    only the model's source edits (never the mutation, never build artifacts).
    """
    await sandbox.write_file(
        _MUTATION_PATCH_REL, _ensure_trailing_newline(candidate.mutation_patch)
    )
    rel = shlex.quote(_MUTATION_PATCH_REL)
    primary = await sandbox.run_command(
        f"git apply --whitespace=nowarn {rel}", timeout=timeout
    )
    if primary.exit_code != 0:
        fallback = await sandbox.run_command(
            f"git apply --3way --whitespace=nowarn {rel}", timeout=timeout
        )
        if fallback.exit_code != 0:
            raise SolverError(
                "failed to apply mutation_patch to set the broken tree: "
                f"{(primary.stderr or primary.stdout or '').strip()[:300]}"
            )
    reinit = await sandbox.run_command(
        f"rm -rf {shlex.quote(_SOLVER_DIR)} .git && "
        f"{_gitignore_append_command()} && "
        f"git init -q && "
        f"git {_GIT_IDENT} add -A && "
        f"git {_GIT_IDENT} commit -q -m broken-baseline",
        timeout=timeout,
    )
    if reinit.exit_code != 0:
        raise SolverError(
            "failed to re-init git history for the broken baseline: "
            f"{(reinit.stderr or reinit.stdout or '').strip()[:300]}"
        )


async def run_solver_rollout(
    candidate: Candidate,
    env_image: EnvImage,
    spec: GeneratedSpec,
    oracle_report: OracleReport,
    *,
    model: str = "",
    solver: AgenticSolver | None = None,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 600.0,
) -> RolloutOutcome:
    """Run one solver rollout end to end and score it via the shared path.

    A green baseline is a hard precondition. The rollout runs in a throwaway
    sandbox whose tree is set broken (mutation applied, committed as HEAD); the
    solver sees only the spec surface and the broken source. The submitted patch
    is then scored in a fresh sandbox through the shared Docker recipe, enforcing
    the FULL ``OracleReport.test_files[]`` suite + P2P. Any rollout/scoring error
    becomes a clean non-solve rather than crashing the caller.
    """
    require_green_baseline(env_image)
    if adapter is None:
        adapter = build_default_registry().get(candidate.language)
    if solver is None:
        solver = AgenticSolver()

    base_tests = reconstruct_suite_tests(
        adapter, oracle_report.fail_to_pass, oracle_report.test_files
    )
    p2p_command = (
        oracle_report.pass_to_pass[0]
        if oracle_report.pass_to_pass
        else env_image.baseline_test_command
    )

    try:
        rollout = await _run_rollout_in_docker(
            candidate,
            env_image,
            spec,
            solver,
            command_timeout=command_timeout,
            docker_client=docker_client,
        )
    except Exception as exc:
        return RolloutOutcome(
            model=model,
            patch="",
            finished=False,
            solved=False,
            score=_non_solve(f"{REASON_ROLLOUT_ERROR}: {type(exc).__name__}: {exc}"),
            turns=0,
            usage=Usage(),
            cost=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Submit-gate: only a finished, non-empty patch is scored.
    if not rollout.finished:
        score = _non_solve(REASON_NOT_FINISHED)
    elif not rollout.patch.strip():
        score = _non_solve(REASON_EMPTY_PATCH, empty=True)
    else:
        score = await _score_rollout(
            candidate,
            env_image,
            adapter,
            rollout.patch,
            base_tests,
            p2p_command,
            command_timeout=command_timeout,
            docker_client=docker_client,
        )

    return RolloutOutcome(
        model=model,
        patch=rollout.patch,
        finished=rollout.finished,
        solved=score.solved,
        score=score,
        turns=rollout.turns,
        usage=rollout.usage,
        cost=rollout.cost,
        error=rollout.error,
    )


async def _run_rollout_in_docker(
    candidate: Candidate,
    env_image: EnvImage,
    spec: GeneratedSpec,
    solver: AgenticSolver,
    *,
    command_timeout: float,
    docker_client: "DockerClient | None",
) -> SolverRollout:
    """Set the broken tree in a throwaway sandbox and run the solver loop."""
    from swe_forge.execution.docker_client import DockerClient
    from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

    client = docker_client or DockerClient()
    config = SandboxConfig(
        name="swe-forge-cal-solver",
        image=env_image.image_tag,
        workspace_dir=env_image.workspace_dir,
        command_timeout=command_timeout,
    )
    sandbox = DockerSandbox(client, config)
    async with sandbox:
        await _setup_broken_baseline(sandbox, candidate, timeout=command_timeout)
        ctx = SolverContext(
            spec=spec,
            language=candidate.language,
            sandbox=sandbox,
            command_timeout=command_timeout,
        )
        return await solver.solve(ctx)


async def _score_rollout(
    candidate: Candidate,
    env_image: EnvImage,
    adapter: LanguageAdapter,
    patch: str,
    base_tests: Sequence[HiddenTest],
    p2p_command: str,
    *,
    command_timeout: float,
    docker_client: "DockerClient | None",
) -> SolveScore:
    scorer = DockerPatchScorer(
        candidate,
        env_image,
        adapter,
        base_tests=base_tests,
        p2p_command=p2p_command,
        command_timeout=command_timeout,
        docker_client=docker_client,
    )
    try:
        return await scorer.score(patch)
    except Exception as exc:
        return _non_solve(
            f"{REASON_SCORE_ERROR}: {type(exc).__name__}: {exc}", applied=False
        )


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TURNS",
    "REASON_APPLY_FAILED",
    "REASON_EMPTY_PATCH",
    "REASON_F2P_OR_P2P_FAILED",
    "REASON_NOT_FINISHED",
    "REASON_ROLLOUT_ERROR",
    "REASON_SCORE_ERROR",
    "AgenticSolver",
    "DockerPatchScorer",
    "RolloutOutcome",
    "SolveScore",
    "SolverContext",
    "SolverError",
    "SolverRollout",
    "build_solver_prompt",
    "capture_workspace_patch",
    "run_solver_rollout",
    "score_patch",
    "solver_tools",
]
