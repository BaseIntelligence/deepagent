"""Agentic test generator with multi-turn LLM loop.

This module provides the TestGenerator class that runs an agentic LLM loop
to generate test commands for SWE tasks. The agent explores the codebase,
installs dependencies, writes tests, and validates them.

Key features:
- TurnLimit enforcement (max 200 turns by default)
- Tool-based interaction (shell, submit_tests, write_file, etc.)
- Validation of generated tests (string-matching rejection)
- Mock-friendly design for unit testing
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Protocol

from swe_forge.llm.client import (
    GenerationRequest,
    LLMClient,
    ToolCall,
    ToolDefinition,
)
from swe_forge.llm.tools import (
    AgenticLoop,
    ShellArgs,
)

if TYPE_CHECKING:
    from swe_forge.swe.models import SweTask

logger = getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_AGENT_TURNS = 400
MAX_VALIDATION_RETRIES = 3
DEFAULT_TIMEOUT_MS = 60_000


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a test engineer writing verification tests for GitHub pull requests for the SWE-bench benchmark.

CONTEXT: You write tests that verify whether a coding agent correctly reproduced a PR's changes.
- fail_to_pass: tests that FAIL on the base commit (before PR), PASS after the PR is applied.
- pass_to_pass: tests that PASS on both the base commit and after the PR.

You have these tools:

FILE EXPLORATION (prefer these over shell for reading code -- they are structured and token-efficient):
- `read_file`: read a file with line numbers, supports offset/limit pagination.
- `list_dir`: list directory contents, supports recursive listing.
- `grep_files`: search file contents with regex (uses ripgrep/grep). Returns matching lines with line numbers.
- `search_files`: find files by glob pattern (e.g. "*.py", "**/*.test.js").

FILE MODIFICATION:
- `write_file`: create or overwrite a file in the repository (for writing test files).
- `apply_patch`: apply a unified diff patch to modify existing files.

EXECUTION:
- `shell`: execute a shell command in the cloned repository (for installing deps, running tests, etc.).

SUBMISSION:
- `submit_tests`: return your final validated test commands, the test files you wrote, AND the install commands that worked.

IMPORTANT: Use `read_file`, `list_dir`, `grep_files`, `search_files` instead of shell commands
like `cat`, `ls`, `grep`, `find` when exploring code. They return cleaner, more compact output.

ENVIRONMENT: You are running in a `ubuntu:24.04` Docker container.
You MUST first install Python and git via: apt-get update && apt-get install -y python3 python3-pip git
Then install all required dependencies yourself via `shell` before running tests.
The install_commands you submit will be replayed in a FRESH container, so they must be complete and
self-contained (apt-get install, pip install, etc.).

IMPORTANT: 
- Write test files RELATIVE to the repo root: test_swe_<feature>.py
- Test command: pytest -c /dev/null test_swe_<feature>.py -v
- The `-c /dev/null` flag is CRITICAL - it skips pyproject.toml config which may have incompatible plugins.

WORKFLOW:
1. SETUP — INSTALL PYTHON AND DEPENDENCIES (this is critical!):
   a. First install Python and git: apt-get update && apt-get install -y python3 python3-pip git
   b. Explore the repo to determine the correct installation procedure:
      - Check README.md, CONTRIBUTING.md, Makefile, Dockerfile, docker-compose.yml
      - Check setup.py, pyproject.toml, setup.cfg (Python)
      - Check package.json (JavaScript/TypeScript)
   c. Run installation commands via `shell` and carefully track which ones SUCCEED (exit code 0).
   d. If the first install attempt fails, read error output, fix the issue, and retry.
   e. ONLY include commands that exited with code 0 in your `install_commands` submission.
2. Use `shell` to explore the repo: project structure, existing tests, build system, dependencies.
3. Read the PR diff carefully: understand WHAT changed and WHY.
4. Find existing test suites covering code ADJACENT to the PR changes -- add them as pass_to_pass.
5. Write NEW test files that exercise the BEHAVIOR introduced by the PR.
6. Run your tests via `shell` to validate: fail_to_pass MUST fail, pass_to_pass MUST pass on base.
7. Call `submit_tests` with everything, including install_commands.

MANDATORY RULES FOR TEST QUALITY:

1. BEHAVIORAL TESTS ONLY
   - Every fail_to_pass test MUST exercise runtime behavior: import modules, call functions,
     instantiate classes, make HTTP requests, run CLI commands, check return values.

2. FORBIDDEN PATTERNS (your submission will be REJECTED if you use these):
   - Reading source files and asserting on their text content.
   - Checking that specific variable names, function names, or import statements exist in source code.
   - Using grep/cat/awk on source files as the test mechanism.
   - Any test whose only assertion is "this string exists in this file".

3. REGRESSION COVERAGE (pass_to_pass)
   - Include at least 1 pass_to_pass command running existing project tests.

4. ROBUSTNESS & EDGE CASES
   - If the PR adds input validation: test with null, empty, oversized, malformed inputs.
   - For bug fixes: test the specific bug scenario AND at least one related edge case.

5. COMPLETENESS
   - Write fail_to_pass tests that cover ALL distinct behaviors added by the PR, not just one.
   - Tests must be specific enough that a lazy agent who only partially implements the PR fails.

6. ANTI-HARDCODING
   - Test with DIFFERENT inputs than those shown in the PR description or diff.
   - This catches agents that hardcode return values instead of implementing real logic.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TestFile:
    """A test file generated by the agent."""

    path: str
    content: str


@dataclass
class GeneratedTests:
    """Result of test generation.

    Attributes:
        fail_to_pass: Test commands that FAIL on base commit, PASS after PR patch.
        pass_to_pass: Test commands that PASS on both base and PR commit.
        test_files: Test files written by the agent.
        install_commands: Shell commands that successfully installed dependencies.
        turn_count: Number of turns used during generation.
        success: Whether generation was successful.
    """

    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    test_files: list[TestFile] = field(default_factory=list)
    install_commands: list[str] = field(default_factory=list)
    turn_count: int = 0
    success: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox Protocol
# ─────────────────────────────────────────────────────────────────────────────


class SandboxProtocol(Protocol):
    """Protocol for sandbox implementations used by TestGenerator."""

    async def run_command(
        self, cmd: str, *, timeout: float | None = None
    ) -> "ExecResultProtocol":
        """Execute a command in the sandbox."""
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox."""
        ...

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        ...


class ExecResultProtocol(Protocol):
    """Protocol for command execution results."""

    @property
    def exit_code(self) -> int: ...

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# Tool Schemas
# ─────────────────────────────────────────────────────────────────────────────


def write_file_tool_schema() -> ToolDefinition:
    """Create the write_file tool schema."""
    return ToolDefinition.create(
        name="write_file",
        description="Create or overwrite a file in the repository. Use this to write test files.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path in the repo (e.g. 'tests/test_new_feature.py')",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write",
                },
            },
            "required": ["path", "content"],
        },
    )


def apply_patch_tool_schema() -> ToolDefinition:
    """Create the apply_patch tool schema."""
    return ToolDefinition.create(
        name="apply_patch",
        description="Apply a unified diff patch to modify files. Use standard unified diff format.",
        parameters={
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Unified diff content to apply",
                },
            },
            "required": ["patch"],
        },
    )


def read_file_tool_schema() -> ToolDefinition:
    """Create the read_file tool schema."""
    return ToolDefinition.create(
        name="read_file",
        description="Read a file from the repository with line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (optional)",
                },
            },
            "required": ["path"],
        },
    )


def list_dir_tool_schema() -> ToolDefinition:
    """Create the list_dir tool schema."""
    return ToolDefinition.create(
        name="list_dir",
        description="List directory contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the directory",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively (default: false)",
                },
            },
            "required": ["path"],
        },
    )


def grep_files_tool_schema() -> ToolDefinition:
    """Create the grep_files tool schema."""
    return ToolDefinition.create(
        name="grep_files",
        description="Search file contents with regex.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (optional, default: repo root)",
                },
            },
            "required": ["pattern"],
        },
    )


def search_files_tool_schema() -> ToolDefinition:
    """Create the search_files tool schema."""
    return ToolDefinition.create(
        name="search_files",
        description="Find files by glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '*.py', '**/*.test.js')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (optional, default: repo root)",
                },
            },
            "required": ["pattern"],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ShellOutput:
    """Result from a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int


@dataclass
class SubmitResult:
    """Result from submit_tests tool."""

    fail_to_pass: list[str]
    pass_to_pass: list[str]
    test_files: list[TestFile]
    install_commands: list[str]


@dataclass
class ToolResult:
    """Result from a tool execution."""

    content: str
    is_error: bool = False
    submit_result: SubmitResult | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Validation Functions
# ─────────────────────────────────────────────────────────────────────────────

STRING_MATCHING_PATTERNS: list[tuple[str, str]] = [
    # Python source-reading patterns
    (r"open\([^)]*\)\.read", "open().read() used to read source files"),
    (r"Path\([^)]*\)\.read_text", "Path().read_text() used to read source files"),
    (r"\.read\(\)[^;]*assert.*\bin\b", ".read() + assert...in (string-matching)"),
    # JavaScript/TypeScript source-reading patterns
    (r"readFileSync\(", "readFileSync() used to read source files"),
    (r"readFile\(", "readFile() used to read source files"),
    # Combined read + assert patterns
    (
        r"assert.*\bin\s+(source|content|text|code|file_content|src|contents)",
        "assert...in source/content (string-matching on file content)",
    ),
    (r"\.(includes|contains)\(['\"]", ".includes()/.contains() on source content"),
]


def reject_string_matching_tests(files: list[TestFile]) -> str | None:
    """Scan test files for string-matching anti-patterns.

    Returns rejection reason if forbidden patterns are found, None otherwise.
    """
    violations: list[str] = []

    for file in files:
        content = file.content
        for pattern, description in STRING_MATCHING_PATTERNS:
            try:
                matches = list(re.finditer(pattern, content))
                if matches:
                    # Check if this is actually testing source files
                    has_behavioral = any(
                        kw in content
                        for kw in ["import ", "require(", "from ", "fetch(", "request("]
                    )

                    # Count assertions
                    total_asserts = (
                        content.count("assert")
                        + content.count("expect(")
                        + content.count("Assert.")
                    )
                    string_match_count = len(matches)

                    # Reject if >50% of assertions are string-matching or no behavioral patterns
                    if total_asserts > 0 and (
                        string_match_count * 2 > total_asserts or not has_behavioral
                    ):
                        violations.append(
                            f"File '{file.path}': {description} "
                            f"({string_match_count} of {total_asserts} assertions)"
                        )
            except re.error:
                continue

    if not violations:
        return None

    return "Your tests use forbidden source-reading patterns:\n- " + "\n- ".join(
        violations
    )


def validate_test_scripts(files: list[TestFile]) -> str | None:
    """Validate generated test scripts for structural issues.

    Returns issue description if validation fails, None otherwise.
    """
    issues: list[str] = []
    known_paths: set[str] = {f.path for f in files}

    for file in files:
        is_shell = file.path.endswith((".sh", ".bash"))

        if is_shell:
            trimmed = file.content.lstrip()
            if not trimmed.startswith("#!"):
                issues.append(
                    f"Shell script '{file.path}' is missing a shebang line (e.g. #!/bin/bash)"
                )

            if not file.content.strip():
                issues.append(f"Shell script '{file.path}' is empty")

        # Check for references to missing test files
        for line in file.content.splitlines():
            trimmed = line.strip()
            for token in trimmed.split():
                if (
                    token.startswith("tests/") or token.startswith("./tests/")
                ) and token.endswith((".py", ".js", ".ts", ".sh")):
                    normalized = token.removeprefix("./")
                    if normalized not in known_paths:
                        issues.append(
                            f"File '{file.path}' references '{normalized}' which was not submitted"
                        )

    if not issues:
        return None

    return "Test script validation issues:\n- " + "\n- ".join(issues)


# ─────────────────────────────────────────────────────────────────────────────
# TestGenerator Class
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerator:
    """Agentic test generator using LLM multi-turn conversation.

    The TestGenerator runs an agentic loop where an LLM:
    1. Explores the repository structure
    2. Installs dependencies
    3. Writes behavioral tests
    4. Validates tests pass/fail correctly
    5. Submits final test commands

    The generator supports both real Docker sandboxes and mock implementations
    for testing.

    Example:
        generator = TestGenerator(llm_client)
        result = await generator.generate_tests(task, sandbox)
        if result.success:
            print(f"Generated {len(result.fail_to_pass)} fail_to_pass tests")
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_turns: int = MAX_AGENT_TURNS,
        model: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2000,
        max_context_tokens: int = 100000,
    ):
        """Initialize TestGenerator.

        Args:
            llm: LLM client for generation.
            max_turns: Maximum number of agent turns (default: 200).
            model: Model identifier to use.
            temperature: Generation temperature (default: 0.2).
            max_tokens: Maximum tokens per response (default: 2000).
            max_context_tokens: Maximum context tokens before compaction (default: 100k).
        """
        self._llm = llm
        self._max_turns = max_turns
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_context_tokens = max_context_tokens
        self._written_files: list[TestFile] = []

    def _get_tools(self) -> list[ToolDefinition]:
        """Get all tool schemas for the agent."""
        from swe_forge.llm.tools import shell_tool_schema, submit_tests_tool_schema

        return [
            shell_tool_schema(),
            submit_tests_tool_schema(),
            read_file_tool_schema(),
            list_dir_tool_schema(),
            grep_files_tool_schema(),
            search_files_tool_schema(),
            write_file_tool_schema(),
            apply_patch_tool_schema(),
        ]

    def _truncate(self, s: str, max_len: int = 4000) -> str:
        """Truncate string to max length with ellipsis."""
        if len(s) <= max_len:
            return s
        return s[:max_len] + "..."

    def _build_user_message(self, task: SweTask) -> str:
        """Build the initial user message for the agent."""
        return f"""Repository: {task.repo}
Language: {task.language}
PR description: {self._truncate(task.prompt, 1000)}

Diff (truncated):
```
{self._truncate(task.patch, 4000)}
```

== CRITICAL: YOU MUST USE TOOLS, NOT TEXT ==

DO NOT write text responses. DO NOT ask questions. DO NOT explain. 
USE TOOLS: shell, read_file, write_file, submit_tests.

== WORKFLOW (EXECUTE NOW) ==

Step 1: INSTALL (use shell tool)
shell: apt-get update && apt-get install -y python3 python3-pip git
shell: pip install -e . (or check pyproject.toml)

Step 2: READ the changed files (use read_file tool)
read_file: path to files from the diff

Step 3: WRITE TESTS (use write_file tool) 
write_file: path=test_swe_feature.py, content=your test code
Tests MUST import modules and call functions - behavioral tests only.

Step 4: RUN TESTS (use shell tool)
shell: pytest -c /dev/null test_swe_feature.py -v

Step 5: SUBMIT (use submit_tests tool) - THIS IS MANDATORY!
submit_tests: fail_to_pass=["pytest test_swe_feature.py"], 
              pass_to_pass=[], 
              test_files=[{{path, content}}], 
              install_commands=["apt-get...", "pip install..."]

== ABSOLUTE RULES ==
- EVERY response MUST include a tool call
- NO text explanations - only tool calls
- NO questions - just execute
- MUST call submit_tests to finish
- If unsure, try something with shell tool
- YOU HAVE 400 TURNS - use them to complete the task"""

    def _test_commands_for_language(self, language: str) -> tuple[list[str], list[str]]:
        """Get suggested build and test commands for a language.

        DEPRECATED: This method returns EMPTY LISTS. The agent MUST discover
        the actual commands by:
        1. Reading pyproject.toml, setup.py, package.json, etc.
        2. TRYING install commands and tracking which succeed (exit 0)
        3. TRYING test commands and tracking which work

        NO HARDCODED DEFAULTS - the LLM agent figures it out via tools.

        Use agentic_config.detect_repository_config() for real detection.
        """
        # NO DEFAULTS - agent must discover everything
        # Return empty lists as fallback - agent will populate via tools
        return ([], [])

    async def _execute_shell(
        self, args: ShellArgs, sandbox: SandboxProtocol
    ) -> ToolResult:
        """Execute a shell command in the sandbox."""
        timeout_sec = args.timeout_ms / 1000.0 if args.timeout_ms else None

        try:
            result = await sandbox.run_command(args.command, timeout=timeout_sec)
            stdout = self._truncate(result.stdout, 3000)
            stderr = self._truncate(result.stderr, 1500)
            return ToolResult(
                content=f"Exit code: {result.exit_code}\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}",
                is_error=result.exit_code != 0,
            )
        except Exception as e:
            return ToolResult(
                content=f"Error executing command: {e}",
                is_error=True,
            )

    async def _handle_write_file(
        self, arguments: dict[str, Any], sandbox: SandboxProtocol
    ) -> ToolResult:
        """Handle write_file tool call."""
        path = arguments.get("path", "")
        content = arguments.get("content", "")

        if not path:
            return ToolResult(content="Error: missing path parameter", is_error=True)

        try:
            await sandbox.write_file(path, content)

            # Track written files
            for existing in self._written_files:
                if existing.path == path:
                    existing.content = content
                    break
            else:
                self._written_files.append(TestFile(path=path, content=content))

            logger.debug(f"Agent wrote file: {path} ({len(content)} bytes)")
            return ToolResult(content=f"File written: {path}")
        except Exception as e:
            return ToolResult(content=f"Failed to write {path}: {e}", is_error=True)

    async def _handle_read_file(
        self, arguments: dict[str, Any], sandbox: SandboxProtocol
    ) -> ToolResult:
        """Handle read_file tool call."""
        path = arguments.get("path", "")

        if not path:
            return ToolResult(content="Error: missing file path", is_error=True)

        try:
            content = await sandbox.read_file(path)

            lines = content.splitlines()
            numbered = "\n".join(
                f"{i + 1}: {line}" for i, line in enumerate(lines[:200])
            )
            if len(lines) > 200:
                numbered += f"\n... [{len(lines) - 200} more lines truncated]"

            return ToolResult(
                content=self._truncate(f"File: {path}\n\n{numbered}", 5000)
            )
        except Exception as e:
            return ToolResult(content=f"Failed to read {path}: {e}", is_error=True)

    async def _handle_tool_call(
        self,
        tool_call: ToolCall,
        sandbox: SandboxProtocol,
    ) -> ToolResult:
        """Handle a single tool call from the agent."""
        tool_name = tool_call.function.name

        try:
            arguments = (
                json.loads(tool_call.function.arguments)
                if tool_call.function.arguments
                else {}
            )
        except json.JSONDecodeError as e:
            return ToolResult(content=f"Invalid JSON arguments: {e}", is_error=True)

        logger.debug(f"Tool call: {tool_name} with args: {list(arguments.keys())}")

        if tool_name == "shell":
            try:
                args = ShellArgs(
                    command=arguments.get("command", ""),
                    timeout_ms=arguments.get("timeout_ms", DEFAULT_TIMEOUT_MS),
                )
                if not args.command:
                    return ToolResult(content="Error: missing command", is_error=True)
                return await self._execute_shell(args, sandbox)
            except Exception as e:
                return ToolResult(
                    content=f"Error parsing shell args: {e}", is_error=True
                )

        elif tool_name == "write_file":
            return await self._handle_write_file(arguments, sandbox)

        elif tool_name == "read_file":
            return await self._handle_read_file(arguments, sandbox)

        elif tool_name == "list_dir":
            path = arguments.get("path", ".")
            try:
                result = await sandbox.run_command(f"ls -la {path}")
                return ToolResult(content=self._truncate(result.stdout, 3000))
            except Exception as e:
                return ToolResult(
                    content=f"Error listing directory: {e}", is_error=True
                )

        elif tool_name == "grep_files":
            pattern = arguments.get("pattern", "")
            path = arguments.get("path", ".")
            try:
                result = await sandbox.run_command(
                    f"grep -rn '{pattern}' {path} | head -100"
                )
                return ToolResult(content=self._truncate(result.stdout, 5000))
            except Exception as e:
                return ToolResult(content=f"Error grepping: {e}", is_error=True)

        elif tool_name == "search_files":
            pattern = arguments.get("pattern", "*")
            path = arguments.get("path", ".")
            try:
                result = await sandbox.run_command(
                    f"find {path} -name '{pattern}' | head -100"
                )
                return ToolResult(content=self._truncate(result.stdout, 2000))
            except Exception as e:
                return ToolResult(content=f"Error searching: {e}", is_error=True)

        elif tool_name == "apply_patch":
            patch = arguments.get("patch", "")
            if not patch:
                return ToolResult(content="Error: missing patch", is_error=True)
            try:
                # Write patch file and apply
                await sandbox.write_file(".swe_forge_tool_patch.tmp", patch)
                result = await sandbox.run_command(
                    "git apply --allow-empty .swe_forge_tool_patch.tmp && rm -f .swe_forge_tool_patch.tmp"
                )
                if result.exit_code == 0:
                    return ToolResult(content="Patch applied successfully.")
                return ToolResult(
                    content=f"git apply failed: {result.stdout}", is_error=True
                )
            except Exception as e:
                return ToolResult(content=f"Error applying patch: {e}", is_error=True)

        elif tool_name == "submit_tests":
            logger.info(
                f"submit_tests called with: fail_to_pass={arguments.get('fail_to_pass')}, install_commands={arguments.get('install_commands')}"
            )
            return self._handle_submit_tests(arguments)

        else:
            return ToolResult(content=f"Unknown tool: {tool_name}", is_error=True)

    def _handle_submit_tests(self, arguments: dict[str, Any]) -> ToolResult:
        """Handle submit_tests tool call."""
        fail_to_pass = arguments.get("fail_to_pass", [])
        pass_to_pass = arguments.get("pass_to_pass", [])
        test_files_raw = arguments.get("test_files", [])
        install_commands = arguments.get("install_commands", [])

        # Parse test files
        test_files: list[TestFile] = []
        for tf in test_files_raw:
            if isinstance(tf, dict) and "path" in tf and "content" in tf:
                test_files.append(TestFile(path=tf["path"], content=tf["content"]))

        # Combine with written files
        all_files = list(self._written_files)
        for tf in test_files:
            if not any(f.path == tf.path for f in all_files):
                all_files.append(tf)

        submit_result = SubmitResult(
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            test_files=all_files,
            install_commands=install_commands,
        )

        return ToolResult(
            content="Tests submitted for validation.",
            submit_result=submit_result,
        )

    async def generate_tests(
        self,
        task: SweTask,
        sandbox: SandboxProtocol,
    ) -> GeneratedTests:
        """Generate test commands for a SWE task.

        Runs an agentic loop where the LLM explores the repository,
        writes tests, and validates them.

        Args:
            task: The SWE task to generate tests for.
            sandbox: Sandbox for executing commands.

        Returns:
            GeneratedTests with the generated test commands.

        Raises:
            RuntimeError: If the turn limit is exhausted without successful generation.
        """
        loop = AgenticLoop(
            max_turns=self._max_turns,
            max_context_tokens=self._max_context_tokens,
        )
        self._written_files = []
        validation_retries = 0

        tools = self._get_tools()

        loop.add_system(SYSTEM_PROMPT)

        user_msg = self._build_user_message(task)
        loop.add_user(user_msg)

        while not loop.is_exhausted():
            loop.compact_if_needed(self._llm, self._model)

            request = GenerationRequest(
                model=self._model,
                messages=loop.messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=tools,
                tool_choice="auto",
            )

            try:
                response = await self._llm.complete(request)
                loop.increment_turn()  # Count each LLM call as a turn
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                break

            if not response.choices:
                break

            choice = response.choices[0]
            message = choice.message

            # Handle tool calls
            if message.tool_calls:
                loop.add_assistant_with_tool_calls(message.content, message.tool_calls)

                for tc in message.tool_calls:
                    result = await self._handle_tool_call(tc, sandbox)

                    if result.submit_result:
                        # Check validation
                        validation_error = self._validate_submission(
                            result.submit_result, validation_retries
                        )

                        if validation_error:
                            validation_retries += 1
                            if validation_retries < MAX_VALIDATION_RETRIES:
                                loop.add_tool_result(
                                    tc.id, f"REJECTED: {validation_error}"
                                )
                                continue
                            else:
                                # Max retries, return what we have
                                return self._create_result(
                                    result.submit_result, loop.turn_count, success=False
                                )

                        # Success!
                        return self._create_result(
                            result.submit_result,
                            loop.turn_count,
                            success=True,
                        )
                    else:
                        loop.add_tool_result(tc.id, result.content)

                continue

            # No tool calls - FORCE TOOL USAGE
            # Agent must use tools, not write text
            loop.add_assistant(message.content or "")
            loop.add_user(
                "ERROR: You must use tools (shell, read_file, write_file, submit_tests), not text. "
                "DO NOT explain. DO NOT ask questions. "
                "Execute: shell('apt-get update'), read_file('file.py'), write_file('test.py', '...'), submit_tests(...). "
                "Call a tool NOW."
            )
            continue

        # Exhausted turns without success
        return GeneratedTests(
            turn_count=loop.turn_count,
            success=False,
        )

    def _validate_submission(
        self,
        submit: SubmitResult,
        retry_count: int,
    ) -> str | None:
        """Validate a test submission.

        Returns None if valid, or an error message if invalid.
        """
        # Check fail_to_pass - ALWAYS required
        if not submit.fail_to_pass:
            return (
                "fail_to_pass must contain at least one test command. "
                "Write a test that FAILS on the base commit and PASSES "
                "after the PR patch is applied."
            )

        # Check install_commands - ALWAYS required
        if not submit.install_commands:
            return (
                "install_commands must contain at least one command. "
                "Run installation commands via shell first, verify they succeed "
                "(exit code 0), then include them in install_commands."
            )

        # Check for string-matching tests
        rejection = reject_string_matching_tests(submit.test_files)
        if rejection:
            return (
                f"{rejection}\n\n"
                "Rewrite your tests to check RUNTIME BEHAVIOR, not file contents. "
                "Import modules, call functions, check return values. "
                "Do NOT use open()/readFileSync() to read source and assert strings."
            )

        # Check test script validity
        script_issues = validate_test_scripts(submit.test_files)
        if script_issues:
            return f"{script_issues}\n\nFix the issues and resubmit."

        return None

    def _create_result(
        self,
        submit: SubmitResult,
        turn_count: int,
        success: bool,
    ) -> GeneratedTests:
        """Create GeneratedTests from a submit result."""
        return GeneratedTests(
            fail_to_pass=submit.fail_to_pass,
            pass_to_pass=submit.pass_to_pass,
            test_files=submit.test_files,
            install_commands=submit.install_commands,
            turn_count=turn_count,
            success=success,
        )
