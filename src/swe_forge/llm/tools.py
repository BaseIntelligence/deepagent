"""Function calling tools and agentic loop support for SWE-Forge.

This module provides:
- Tool schemas for shell and submit_tests (used by agentic test generation)
- Tool result parsing helpers
- Multi-turn conversation support (agentic loop, up to 200 turns max)
- Turn budget tracking and enforcement
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


from swe_forge.llm.client import (
    Message,
    ToolCall,
    ToolDefinition,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Schemas
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SHELL_TIMEOUT_MS = 30_000  # 30 seconds default timeout
MAX_TURNS_DEFAULT = 400


def shell_tool_schema() -> ToolDefinition:
    """Create the shell tool schema for command execution.

    The shell tool allows an agent to execute shell commands in a sandboxed
    environment. Used for exploration, installing dependencies, running tests, etc.

    Returns:
        ToolDefinition for the shell command execution tool.
    """
    return ToolDefinition.create(
        name="shell",
        description="Execute a shell command in the repository. Returns stdout, stderr, and exit code.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": f"Timeout in milliseconds (default: {DEFAULT_SHELL_TIMEOUT_MS})",
                    "default": DEFAULT_SHELL_TIMEOUT_MS,
                },
            },
            "required": ["command"],
        },
    )


def submit_tests_tool_schema() -> ToolDefinition:
    """Create the submit_tests tool schema.

    The submit_tests tool is used by the agentic test generator to submit
    the final validated test commands. This signals the end of the test
    generation process.

    Returns:
        ToolDefinition for the submit_tests tool.
    """
    return ToolDefinition.create(
        name="submit_tests",
        description="Submit the final validated test commands, test files, and install commands.",
        parameters={
            "type": "object",
            "properties": {
                "fail_to_pass": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Commands that FAIL on base commit, PASS after PR patch",
                },
                "pass_to_pass": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Commands that PASS on both base and PR commit",
                },
                "test_files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative file path",
                            },
                            "content": {
                                "type": "string",
                                "description": "Full file content",
                            },
                        },
                        "required": ["path", "content"],
                    },
                    "description": "Test files written during this session",
                },
                "install_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Shell commands that successfully installed all dependencies. "
                        "Only include commands that exited with code 0."
                    ),
                },
            },
            "required": [
                "fail_to_pass",
                "pass_to_pass",
                "test_files",
                "install_commands",
            ],
        },
    )


def set_dataset_prompt_tool_schema() -> ToolDefinition:
    """Create the set_dataset_prompt tool schema.
    
    This tool is used by the agent to set a natural description of the PR changes.
    The prompt describes what the PR does, in natural language.
    
    Returns:
        ToolDefinition for the set_dataset_prompt tool.
    """
    return ToolDefinition.create(
        name="set_dataset_prompt",
        description="Set a natural language description of what this PR does. Use this to describe the changes in simple terms.",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Natural description of the PR changes (e.g., 'Fixes null pointer exception in login when email is empty')",
                },
            },
            "required": ["prompt"],
        },
    )


# Module-level schema constants for easy import
SHELL_TOOL_SCHEMA: ToolDefinition = shell_tool_schema()
SUBMIT_TESTS_TOOL_SCHEMA: ToolDefinition = submit_tests_tool_schema()
SET_DATASET_PROMPT_TOOL_SCHEMA: ToolDefinition = set_dataset_prompt_tool_schema()


# ─────────────────────────────────────────────────────────────────────────────
# Docker Tools
# ─────────────────────────────────────────────────────────────────────────────

DOCKER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "docker_build",
        "description": "Build Docker image for testing",
        "parameters": {
            "type": "object",
            "properties": {
                "dockerfile": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["dockerfile"],
        },
    },
    {
        "name": "docker_run",
        "description": "Run command in Docker container",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "command": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["image", "command"],
        },
    },
]


def docker_build_tool_schema() -> ToolDefinition:
    """Create the docker_build tool schema."""
    return ToolDefinition.create(
        name="docker_build",
        description="Build Docker image for testing",
        parameters={
            "type": "object",
            "properties": {
                "dockerfile": {
                    "type": "string",
                    "description": "Dockerfile content",
                },
                "context": {
                    "type": "string",
                    "description": "Build context directory (optional)",
                },
            },
            "required": ["dockerfile"],
        },
    )


def docker_run_tool_schema() -> ToolDefinition:
    """Create the docker_run tool schema."""
    return ToolDefinition.create(
        name="docker_run",
        description="Run command in Docker container",
        parameters={
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Docker image to use",
                },
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command to run as array of strings",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds",
                    "default": DEFAULT_SHELL_TIMEOUT_MS,
                },
            },
            "required": ["image", "command"],
        },
    )


DOCKER_BUILD_TOOL_SCHEMA: ToolDefinition = docker_build_tool_schema()
DOCKER_RUN_TOOL_SCHEMA: ToolDefinition = docker_run_tool_schema()


def get_tool_schemas() -> list[ToolDefinition]:
    """Get the list of tool schemas for agentic test generation.

    Returns the standard tool schemas used by the SWE-Forge agentic
    test generation loop. Currently includes:
    - shell: Execute shell commands in the sandbox
    - submit_tests: Submit final test results

    Returns:
        List of ToolDefinition objects for the agent tools.
    """
    return [SHELL_TOOL_SCHEMA, SUBMIT_TESTS_TOOL_SCHEMA]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Argument Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ShellArgs:
    """Parsed arguments for the shell tool."""

    command: str
    timeout_ms: int = DEFAULT_SHELL_TIMEOUT_MS


@dataclass
class SubmittedTestFile:
    """A test file submitted by the agent."""

    path: str
    content: str


@dataclass
class SubmitTestsArgs:
    """Parsed arguments for the submit_tests tool."""

    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    test_files: list[SubmittedTestFile] = field(default_factory=list)
    install_commands: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Call Parsing
# ─────────────────────────────────────────────────────────────────────────────


class ToolParseError(Exception):
    """Raised when a tool call cannot be parsed."""

    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Failed to parse {tool_name}: {reason}")


def parse_tool_call(tool_call: ToolCall) -> ShellArgs | SubmitTestsArgs:
    """Parse and validate a tool call's arguments.

    Parses the JSON arguments string from a ToolCall and returns
    a typed arguments object based on the tool name.

    Args:
        tool_call: The ToolCall object to parse.

    Returns:
        ShellArgs if the tool is 'shell', SubmitTestsArgs if 'submit_tests'.

    Raises:
        ToolParseError: If the tool name is unknown or arguments are invalid.
    """
    tool_name = tool_call.function.name
    arguments_str = tool_call.function.arguments

    # Parse JSON arguments
    try:
        arguments = json.loads(arguments_str) if arguments_str else {}
    except json.JSONDecodeError as e:
        raise ToolParseError(tool_name, f"Invalid JSON: {e}")

    if tool_name == "shell":
        return _parse_shell_args(tool_name, arguments)
    elif tool_name == "submit_tests":
        return _parse_submit_tests_args(tool_name, arguments)
    else:
        raise ToolParseError(tool_name, f"Unknown tool: {tool_name}")


def _parse_shell_args(tool_name: str, arguments: dict[str, Any]) -> ShellArgs:
    """Parse shell tool arguments."""
    if "command" not in arguments:
        raise ToolParseError(tool_name, "Missing required parameter: command")

    command = arguments["command"]
    if not isinstance(command, str):
        raise ToolParseError(
            tool_name, f"command must be string, got {type(command).__name__}"
        )

    if command.strip() == "":
        raise ToolParseError(tool_name, "command cannot be empty")

    timeout_ms = arguments.get("timeout_ms", DEFAULT_SHELL_TIMEOUT_MS)
    if not isinstance(timeout_ms, int):
        raise ToolParseError(
            tool_name, f"timeout_ms must be integer, got {type(timeout_ms).__name__}"
        )

    if timeout_ms <= 0:
        raise ToolParseError(tool_name, "timeout_ms must be positive")

    return ShellArgs(command=command, timeout_ms=timeout_ms)


def _parse_submit_tests_args(
    tool_name: str, arguments: dict[str, Any]
) -> SubmitTestsArgs:
    """Parse submit_tests tool arguments."""
    # Get optional parameters with defaults
    fail_to_pass = arguments.get("fail_to_pass", [])
    pass_to_pass = arguments.get("pass_to_pass", [])
    test_files_raw = arguments.get("test_files", [])
    install_commands = arguments.get("install_commands", [])

    # Validate types
    if not isinstance(fail_to_pass, list):
        raise ToolParseError(
            tool_name, f"fail_to_pass must be array, got {type(fail_to_pass).__name__}"
        )
    if not isinstance(pass_to_pass, list):
        raise ToolParseError(
            tool_name, f"pass_to_pass must be array, got {type(pass_to_pass).__name__}"
        )
    if not isinstance(test_files_raw, list):
        raise ToolParseError(
            tool_name, f"test_files must be array, got {type(test_files_raw).__name__}"
        )
    if not isinstance(install_commands, list):
        raise ToolParseError(
            tool_name,
            f"install_commands must be array, got {type(install_commands).__name__}",
        )

    # Validate element types
    for i, cmd in enumerate(fail_to_pass):
        if not isinstance(cmd, str):
            raise ToolParseError(
                tool_name, f"fail_to_pass[{i}] must be string, got {type(cmd).__name__}"
            )

    for i, cmd in enumerate(pass_to_pass):
        if not isinstance(cmd, str):
            raise ToolParseError(
                tool_name, f"pass_to_pass[{i}] must be string, got {type(cmd).__name__}"
            )

    for i, cmd in enumerate(install_commands):
        if not isinstance(cmd, str):
            raise ToolParseError(
                tool_name,
                f"install_commands[{i}] must be string, got {type(cmd).__name__}",
            )

    # Parse test files
    test_files: list[SubmittedTestFile] = []
    for i, tf in enumerate(test_files_raw):
        if not isinstance(tf, dict):
            raise ToolParseError(
                tool_name, f"test_files[{i}] must be object, got {type(tf).__name__}"
            )
        if "path" not in tf:
            raise ToolParseError(
                tool_name, f"test_files[{i}] missing required field: path"
            )
        if "content" not in tf:
            raise ToolParseError(
                tool_name, f"test_files[{i}] missing required field: content"
            )
        if not isinstance(tf["path"], str):
            raise ToolParseError(
                tool_name,
                f"test_files[{i}].path must be string, got {type(tf['path']).__name__}",
            )
        if not isinstance(tf["content"], str):
            raise ToolParseError(
                tool_name,
                f"test_files[{i}].content must be string, got {type(tf['content']).__name__}",
            )
        test_files.append(SubmittedTestFile(path=tf["path"], content=tf["content"]))

    return SubmitTestsArgs(
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        test_files=test_files,
        install_commands=install_commands,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Turn Budget
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TurnBudget:
    """Track and enforce turn limits for agentic loops.

    Prevents infinite loops by enforcing a maximum number of turns
    in the conversation with the LLM.

    Attributes:
        max_turns: Maximum number of turns allowed (default: 200).
        current_turn: Current turn count (starts at 0).
    """

    max_turns: int = MAX_TURNS_DEFAULT
    current_turn: int = 0

    def remaining(self) -> int:
        """Get the number of remaining turns.

        Returns:
            Number of turns left before exhaustion.
        """
        return max(0, self.max_turns - self.current_turn)

    def increment(self) -> int:
        """Increment the turn counter.

        Returns:
            The new current turn value.

        Raises:
            RuntimeError: If the budget is already exhausted.
        """
        if self.is_exhausted():
            raise RuntimeError(
                f"Turn budget exhausted: {self.current_turn}/{self.max_turns} turns used"
            )
        self.current_turn += 1
        return self.current_turn

    def is_exhausted(self) -> bool:
        """Check if the turn budget has been exhausted.

        Returns:
            True if current_turn >= max_turns, False otherwise.
        """
        return self.current_turn >= self.max_turns

    def __repr__(self) -> str:
        return f"TurnBudget(turns={self.current_turn}/{self.max_turns}, remaining={self.remaining()})"


# ─────────────────────────────────────────────────────────────────────────────
# Agentic Loop
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_CONTEXT_TOKENS = 200000
DEFAULT_KEEP_LAST_N = 10


def _get_token_encoder():
    """Get tiktoken encoder, falling back to cl100k_base if model not found."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(messages: list[Message], encoder=None) -> int:
    """Estimate total tokens in message list.

    Args:
        messages: List of messages to count.
        encoder: Optional tiktoken encoder. If None, uses cl100k_base.

    Returns:
        Estimated token count.
    """
    if encoder is None:
        encoder = _get_token_encoder()

    if encoder is None:
        return sum(len((m.content or "")[:1000]) for m in messages)

    total = 0
    for msg in messages:
        total += 4
        total += len(encoder.encode(msg.content or ""))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += len(encoder.encode(tc.function.name))
                total += len(encoder.encode(tc.function.arguments or ""))
                total += 10
        if msg.tool_call_id:
            total += 10
    return total


class AgenticLoop:
    """Manage multi-turn conversation state for agentic workflows.

    Maintains conversation history and turn tracking for LLM-based
    agentic loops. Used by the test generator to manage the multi-turn
    conversation with the LLM.

    Features:
    - Turn budget tracking
    - Token estimation and auto-compaction
    - Message history management

    Example:
        loop = AgenticLoop(max_turns=200, max_context_tokens=100000)
        loop.add_system("You are a test engineer...")
        loop.add_user("Write tests for this PR...")

        while not loop.is_exhausted():
            if loop.should_compact():
                await loop.compact(llm_client)
            response = await llm.complete(loop.to_request())
            loop.add_assistant(response)
    """

    def __init__(
        self,
        max_turns: int = MAX_TURNS_DEFAULT,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        keep_last_n: int = DEFAULT_KEEP_LAST_N,
    ):
        """Initialize the agentic loop.

        Args:
            max_turns: Maximum number of turns before exhaustion.
            max_context_tokens: Maximum tokens before auto-compaction.
            keep_last_n: Number of recent turns to keep during compaction.
        """
        self._budget = TurnBudget(max_turns=max_turns)
        self._messages: list[Message] = []
        self._max_context_tokens = max_context_tokens
        self._keep_last_n = keep_last_n
        self._encoder = _get_token_encoder()
        self._compaction_count = 0

    @property
    def budget(self) -> TurnBudget:
        """Get the turn budget tracker."""
        return self._budget

    @property
    def messages(self) -> list[Message]:
        """Get the conversation history (read-only)."""
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        """Get the current turn count."""
        return self._budget.current_turn

    @property
    def max_turns(self) -> int:
        """Get the maximum allowed turns."""
        return self._budget.max_turns

    def is_exhausted(self) -> bool:
        """Check if the turn limit has been reached.

        Returns:
            True if no more turns are allowed.
        """
        return self._budget.is_exhausted()

    def remaining_turns(self) -> int:
        """Get remaining turns before exhaustion."""
        return self._budget.remaining()

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history.

        Args:
            message: The message to add.

        Raises:
            RuntimeError: If the turn budget is exhausted.
        """
        self._messages.append(message)

    def add_system(self, content: str) -> None:
        """Add a system message.

        Args:
            content: The system prompt content.
        """
        self._messages.append(Message.system(content))

    def add_user(self, content: str) -> None:
        """Add a user message.

        Args:
            content: The user message content.
        """
        self._messages.append(Message.user(content))

    def add_assistant(self, content: str) -> None:
        """Add an assistant message.

        Args:
            content: The assistant response content.
        """
        self._messages.append(Message.assistant(content))

    def add_assistant_with_tool_calls(
        self, content: str, tool_calls: list[ToolCall]
    ) -> None:
        """Add an assistant message with tool calls.

        Args:
            content: The assistant response content (often empty).
            tool_calls: List of tool calls to make.
        """
        self._messages.append(Message.assistant_with_tool_calls(content, tool_calls))

    def add_tool_result(self, call_id: str, content: str) -> None:
        """Add a tool result message.

        Args:
            call_id: The tool call ID this result corresponds to.
            content: The tool result content.
        """
        self._messages.append(Message.tool_result(call_id, content))

    def increment_turn(self) -> int:
        """Manually increment the turn counter after each LLM API call.

        Use this after each LLM invocation to track turns correctly,
        including when the LLM responds with tool calls.

        Returns:
            The new current turn value.

        Raises:
            RuntimeError: If the budget is already exhausted.
        """
        return self._budget.increment()

    def last_message(self) -> Message | None:
        """Get the last message in the conversation, if any.

        Returns:
            The most recent message, or None if empty.
        """
        return self._messages[-1] if self._messages else None

    def last_user_message(self) -> Message | None:
        """Get the last user message, if any.

        Returns:
            The most recent user message, or None.
        """
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg
        return None

    def last_assistant_message(self) -> Message | None:
        """Get the last assistant message, if any.

        Returns:
            The most recent assistant message, or None.
        """
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg
        return None

    def clear(self) -> None:
        """Clear all messages and reset turn counter."""
        self._messages.clear()
        self._budget.current_turn = 0

    def message_count(self) -> int:
        """Get the total number of messages in history."""
        return len(self._messages)

    def token_count(self) -> int:
        """Estimate token count of current message history."""
        return estimate_tokens(self._messages, self._encoder)

    def should_compact(self) -> bool:
        """Check if compaction is needed based on token count."""
        return self.token_count() > self._max_context_tokens

    def compact(self, llm_client=None, model: str = "gpt-4o-mini") -> int:
        """Compact message history to reduce token count.

        Strategy: Keep system message + recent turns, truncate middle.
        If llm_client is provided, uses LLM to summarize middle messages.

        Args:
            llm_client: Optional LLM client for summarization.
            model: Model to use for summarization.

        Returns:
            Number of tokens saved by compaction.
        """
        original_tokens = self.token_count()

        if len(self._messages) <= self._keep_last_n + 1:
            return 0

        system_msg = (
            self._messages[0]
            if self._messages and self._messages[0].role == "system"
            else None
        )
        recent_start = len(self._messages) - self._keep_last_n
        recent = self._messages[recent_start:]

        if system_msg:
            middle = self._messages[1:recent_start]
        else:
            middle = self._messages[:recent_start]

        if not middle:
            return 0

        summary_content = self._summarize_messages(middle, llm_client, model)
        summary_msg = Message.system(f"[Previous context summary]\n{summary_content}")

        if system_msg:
            self._messages = [system_msg, summary_msg] + recent
        else:
            self._messages = [summary_msg] + recent

        self._compaction_count += 1
        return original_tokens - self.token_count()

    def _summarize_messages(
        self, messages: list[Message], llm_client=None, model: str = "gpt-4o-mini"
    ) -> str:
        """Summarize a list of messages.

        Args:
            messages: Messages to summarize.
            llm_client: Optional LLM client for summarization.
            model: Model to use.

        Returns:
            Summary text.
        """
        if not messages:
            return "No previous context."

        message_preview = "\n".join(
            f"[{m.role}]: {(m.content or '')[:200]}..." for m in messages[:10]
        )

        if llm_client is not None:
            import asyncio

            try:
                from swe_forge.llm.client import GenerationRequest

                prompt = f"""Provide a detailed summary for continuing this conversation.
Focus on information that would be helpful for the next agent to continue the work.

When constructing the summary, use this template:
---
## Goal

[What goal(s) is the user trying to accomplish?]

## Instructions

- [What important instructions did the user give you that are relevant]
- [If there is a plan or spec, include information about it so next agent can continue using it]

## Discoveries

[What notable things were learned during this conversation that would be useful for the next agent to know when continuing the work]

## Accomplished

[What work has been completed, what work is still in progress, and what work is left?]

## Relevant files / directories

[Construct a structured list of relevant files that have been read, edited, or created that pertain to the task at hand.]
---

Messages:
{message_preview}

Summary:"""

                async def get_summary():
                    request = GenerationRequest(
                        model=model,
                        messages=[Message.user(prompt)],
                        max_tokens=800,
                    )
                    response = await llm_client.complete(request)
                    return response.first_content() or message_preview[:500]

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        return message_preview[:500]
                    return loop.run_until_complete(get_summary())
                except RuntimeError:
                    return message_preview[:500]
            except Exception:
                pass

        return message_preview[:500]

    def compact_if_needed(self, llm_client=None, model: str = "gpt-4o-mini") -> bool:
        """Compact if needed, return True if compaction occurred.

        Args:
            llm_client: Optional LLM client for summarization.
            model: Model to use for summarization.

        Returns:
            True if compaction occurred, False otherwise.
        """
        if self.should_compact():
            self.compact(llm_client, model)
            return True
        return False

    def __repr__(self) -> str:
        return f"AgenticLoop(messages={len(self._messages)}, turns={self.turn_count}/{self.max_turns})"
