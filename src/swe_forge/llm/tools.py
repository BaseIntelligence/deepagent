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

from pydantic import ValidationError

from swe_forge.llm.client import (
    FunctionCall,
    Message,
    ToolCall,
    ToolDefinition,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Schemas
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SHELL_TIMEOUT_MS = 30_000  # 30 seconds default timeout
MAX_TURNS_DEFAULT = 200


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


# Module-level schema constants for easy import
SHELL_TOOL_SCHEMA: ToolDefinition = shell_tool_schema()
SUBMIT_TESTS_TOOL_SCHEMA: ToolDefinition = submit_tests_tool_schema()


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


class AgenticLoop:
    """Manage multi-turn conversation state for agentic workflows.

    Maintains conversation history and turn tracking for LLM-based
    agentic loops. Used by the test generator to manage the multi-turn
    conversation with the LLM.

    Example:
        loop = AgenticLoop(max_turns=200)
        loop.add_system("You are a test engineer...")
        loop.add_user("Write tests for this PR...")

        while not loop.is_exhausted():
            response = await llm.complete(loop.to_request())
            loop.add_assistant(response)
            # Process response, maybe add tool results
            loop.add_tool_result(tool_call_id, result)
    """

    def __init__(self, max_turns: int = MAX_TURNS_DEFAULT):
        """Initialize the agentic loop.

        Args:
            max_turns: Maximum number of turns before exhaustion.
        """
        self._budget = TurnBudget(max_turns=max_turns)
        self._messages: list[Message] = []

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
        """Add a user message and increment turn counter.

        This represents a "turn" in the conversation where we're
        giving new input to the model.

        Args:
            content: The user message content.

        Raises:
            RuntimeError: If the turn budget is exhausted.
        """
        # Increment budget first (will raise if exhausted)
        self._budget.increment()
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

    def __repr__(self) -> str:
        return f"AgenticLoop(messages={len(self._messages)}, turns={self.turn_count}/{self.max_turns})"
