"""Tests for swe_forge.swe.test_generator module."""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock

from swe_forge.llm.client import (
    Choice,
    FunctionCall,
    GenerationResponse,
    Message,
    ToolCall,
    Usage,
)
from swe_forge.swe.models import SweTask
from swe_forge.swe.test_generator import (
    GeneratedTests,
    TestFile,
    TestGenerator,
    MAX_AGENT_TURNS,
    MAX_VALIDATION_RETRIES,
    reject_string_matching_tests,
    validate_test_scripts,
    write_file_tool_schema,
    apply_patch_tool_schema,
    read_file_tool_schema,
    list_dir_tool_schema,
    grep_files_tool_schema,
    search_files_tool_schema,
)


class TestTestFileDataclass:
    def test_test_file_creation(self):
        tf = TestFile(path="tests/test_foo.py", content="def test_foo(): pass")
        assert tf.path == "tests/test_foo.py"
        assert tf.content == "def test_foo(): pass"

    def test_test_file_equality(self):
        tf1 = TestFile(path="test.py", content="pass")
        tf2 = TestFile(path="test.py", content="pass")
        tf3 = TestFile(path="test.py", content="fail")
        assert tf1 == tf2
        assert tf1 != tf3


class TestGeneratedTestsDataclass:
    def test_generated_tests_creation(self):
        tf = TestFile(path="test.py", content="pass")
        result = GeneratedTests(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=["pytest other.py"],
            test_files=[tf],
            install_commands=["pip install -e ."],
            turn_count=5,
            success=True,
        )
        assert result.fail_to_pass == ["pytest test.py"]
        assert result.pass_to_pass == ["pytest other.py"]
        assert len(result.test_files) == 1
        assert result.install_commands == ["pip install -e ."]
        assert result.turn_count == 5
        assert result.success is True

    def test_generated_tests_defaults(self):
        result = GeneratedTests()
        assert result.fail_to_pass == []
        assert result.pass_to_pass == []
        assert result.test_files == []
        assert result.install_commands == []
        assert result.turn_count == 0
        assert result.success is False


class TestToolSchemas:
    def test_write_file_tool_schema_structure(self):
        tool = write_file_tool_schema()
        assert tool.type == "function"
        assert tool.function.name == "write_file"
        params = tool.function.parameters
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert "content" in params["properties"]
        assert "path" in params["required"]

    def test_apply_patch_tool_schema_structure(self):
        tool = apply_patch_tool_schema()
        assert tool.function.name == "apply_patch"
        params = tool.function.parameters
        assert "patch" in params["properties"]
        assert "patch" in params["required"]

    def test_read_file_tool_schema_structure(self):
        tool = read_file_tool_schema()
        assert tool.function.name == "read_file"
        params = tool.function.parameters
        assert params["properties"]["path"]["type"] == "string"

    def test_list_dir_tool_schema_structure(self):
        tool = list_dir_tool_schema()
        assert tool.function.name == "list_dir"
        params = tool.function.parameters
        assert "path" in params["required"]

    def test_grep_files_tool_schema_structure(self):
        tool = grep_files_tool_schema()
        assert tool.function.name == "grep_files"
        params = tool.function.parameters
        assert "pattern" in params["required"]

    def test_search_files_tool_schema_structure(self):
        tool = search_files_tool_schema()
        assert tool.function.name == "search_files"
        params = tool.function.parameters
        assert "pattern" in params["required"]


class TestRejectStringMatchingTests:
    def test_accepts_behavioral_tests(self):
        files = [
            TestFile(
                path="test.py",
                content="import mymodule\ndef test_feature():\n    assert mymodule.func() == 1",
            )
        ]
        result = reject_string_matching_tests(files)
        assert result is None

    def test_rejects_open_read_pattern(self):
        files = [
            TestFile(
                path="test.py",
                content="def test_bad():\n    content = open('src.py').read()\n    assert 'def foo' in content",
            )
        ]
        result = reject_string_matching_tests(files)
        assert result is not None
        assert "open().read()" in result

    def test_rejects_path_read_text_pattern(self):
        files = [
            TestFile(
                path="test.py",
                content="from pathlib import Path\ndef test_bad():\n    content = Path('src.py').read_text()\n    assert 'class' in content",
            )
        ]
        result = reject_string_matching_tests(files)
        assert result is not None
        assert "Path().read_text()" in result

    def test_rejects_readFileSync_pattern(self):
        files = [
            TestFile(
                path="test.js",
                content="const fs = require('fs');\nconst src = fs.readFileSync('src.js', 'utf8');\nassert(src.includes('function'));",
            )
        ]
        result = reject_string_matching_tests(files)
        assert result is not None
        assert "readFileSync()" in result

    def test_accepts_mixed_tests_with_majority_behavioral(self):
        files = [
            TestFile(
                path="test.py",
                content="""
import mymodule
def test_behavioral():
    result = mymodule.process()
    assert result.status == 'ok'
    with open('config.py') as f:
        config = f.read()
    assert 'DEBUG' in config
""",
            )
        ]
        result = reject_string_matching_tests(files)
        assert result is None

    def test_multiple_files_with_violations(self):
        files = [
            TestFile(
                path="test_a.py",
                content="content = open('a.py').read()\nassert 'x' in content",
            ),
            TestFile(
                path="test_b.py",
                content="content = open('b.py').read()\nassert 'y' in content",
            ),
        ]
        result = reject_string_matching_tests(files)
        assert result is not None
        assert "test_a.py" in result
        assert "test_b.py" in result


class TestValidateTestScripts:
    def test_accepts_valid_shell_script(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="#!/bin/bash\npython -m pytest tests/",
            )
        ]
        result = validate_test_scripts(files)
        assert result is None

    def test_rejects_missing_shebang(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="python -m pytest tests/",
            )
        ]
        result = validate_test_scripts(files)
        assert result is not None
        assert "shebang" in result.lower()

    def test_rejects_empty_shell_script(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="",
            )
        ]
        result = validate_test_scripts(files)
        assert result is not None
        assert "empty" in result.lower()

    def test_rejects_references_to_missing_files(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="#!/bin/bash\npython tests/test_missing.py",
            )
        ]
        result = validate_test_scripts(files)
        assert result is not None
        assert "test_missing.py" in result

    def test_accepts_when_referenced_file_exists(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="#!/bin/bash\npython tests/test_exists.py",
            ),
            TestFile(
                path="tests/test_exists.py",
                content="def test(): pass",
            ),
        ]
        result = validate_test_scripts(files)
        assert result is None


@dataclass
class MockExecResult:
    stdout: str
    stderr: str
    exit_code: int


class MockSandbox:
    def __init__(self):
        self.files: dict[str, str] = {}
        self.commands: list[tuple[str, float | None]] = []

    async def run_command(self, cmd: str, *, timeout: float | None = None):
        self.commands.append((cmd, timeout))
        return MockExecResult(stdout="output", stderr="", exit_code=0)

    async def write_file(self, path: str, content: str):
        self.files[path] = content

    async def read_file(self, path: str):
        return self.files.get(path, "")


class MockLLMClient:
    def __init__(self, responses: list[GenerationResponse] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.requests: list[object] = []

    async def complete(self, request):
        self.requests.append(request)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        raise RuntimeError("No more mock responses")


def create_response_with_tool_calls(tool_calls: list[ToolCall]) -> GenerationResponse:
    return GenerationResponse(
        id="resp_1",
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=tool_calls,
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


def create_response_with_text(text: str) -> GenerationResponse:
    return GenerationResponse(
        id="resp_1",
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


class TestTestGenerator:
    @pytest.mark.asyncio
    async def test_generate_tests_successful_submission(self):
        task = SweTask(
            id="test-1",
            repo="owner/repo",
            base_commit="abc123",
            language="python",
            patch="diff --git a/file.py",
            prompt="Test PR description",
        )

        submit_tool_call = ToolCall(
            id="call_submit",
            type="function",
            function=FunctionCall(
                name="submit_tests",
                arguments='{"fail_to_pass": ["pytest test.py"], "pass_to_pass": ["pytest other.py"], "test_files": [], "install_commands": ["pip install -e ."]}',
            ),
        )

        responses = [
            create_response_with_tool_calls([submit_tool_call]),
        ]

        mock_llm = MockLLMClient(responses)
        mock_sandbox = MockSandbox()

        generator = TestGenerator(mock_llm, max_turns=10)
        result = await generator.generate_tests(task, mock_sandbox)

        assert result.success is True
        assert result.fail_to_pass == ["pytest test.py"]
        assert result.pass_to_pass == ["pytest other.py"]
        assert result.install_commands == ["pip install -e ."]
        assert result.turn_count == 1

    @pytest.mark.asyncio
    async def test_generate_tests_handles_shell_tool(self):
        task = SweTask(
            id="test-2",
            repo="owner/repo",
            base_commit="abc123",
            language="python",
            prompt="Test",
        )

        shell_tool_call = ToolCall(
            id="call_shell",
            type="function",
            function=FunctionCall(
                name="shell",
                arguments='{"command": "ls -la"}',
            ),
        )

        mock_sandbox = MockSandbox()
        responses = [
            create_response_with_tool_calls([shell_tool_call]),
            create_response_with_text("Done"),
        ]

        mock_llm = MockLLMClient(responses)
        generator = TestGenerator(mock_llm, max_turns=5)

        await generator.generate_tests(task, mock_sandbox)

        assert len(mock_sandbox.commands) >= 1
        assert "ls -la" in mock_sandbox.commands[0][0]

    @pytest.mark.asyncio
    async def test_generate_tests_handles_write_file(self):
        task = SweTask(
            id="test-3",
            repo="owner/repo",
            base_commit="abc123",
            language="python",
            prompt="Test",
        )

        write_tool_call = ToolCall(
            id="call_write",
            type="function",
            function=FunctionCall(
                name="write_file",
                arguments='{"path": "test_new.py", "content": "def test_x(): pass"}',
            ),
        )

        mock_sandbox = MockSandbox()
        responses = [
            create_response_with_tool_calls([write_tool_call]),
            create_response_with_text("Done"),
        ]

        mock_llm = MockLLMClient(responses)
        generator = TestGenerator(mock_llm, max_turns=5)

        await generator.generate_tests(task, mock_sandbox)

        assert "test_new.py" in mock_sandbox.files
        assert "def test_x()" in mock_sandbox.files["test_new.py"]

    @pytest.mark.asyncio
    async def test_generate_tests_exhausts_turns(self):
        task = SweTask(
            id="test-4",
            repo="owner/repo",
            base_commit="abc123",
            language="python",
            prompt="Test",
        )

        shell_tool_call = ToolCall(
            id="call_shell",
            type="function",
            function=FunctionCall(
                name="shell",
                arguments='{"command": "echo hello"}',
            ),
        )

        responses = [
            create_response_with_tool_calls([shell_tool_call]),
            create_response_with_text("I'm exploring..."),
            create_response_with_tool_calls([shell_tool_call]),
            create_response_with_text("Still exploring..."),
        ]

        mock_llm = MockLLMClient(responses)
        mock_sandbox = MockSandbox()

        generator = TestGenerator(mock_llm, max_turns=3)
        result = await generator.generate_tests(task, mock_sandbox)

        assert result.success is False
        assert result.turn_count == 3

    @pytest.mark.asyncio
    async def test_generate_tests_validates_empty_fail_to_pass(self):
        task = SweTask(
            id="test-5",
            repo="owner/repo",
            base_commit="abc123",
            language="python",
            prompt="Test",
        )

        invalid_submit = ToolCall(
            id="call_submit",
            type="function",
            function=FunctionCall(
                name="submit_tests",
                arguments='{"fail_to_pass": [], "pass_to_pass": [], "test_files": [], "install_commands": []}',
            ),
        )

        valid_submit = ToolCall(
            id="call_submit2",
            type="function",
            function=FunctionCall(
                name="submit_tests",
                arguments='{"fail_to_pass": ["pytest x"], "pass_to_pass": [], "test_files": [], "install_commands": ["pip install"]}',
            ),
        )

        responses = [
            create_response_with_tool_calls([invalid_submit]),
            create_response_with_tool_calls([valid_submit]),
        ]

        mock_llm = MockLLMClient(responses)
        mock_sandbox = MockSandbox()

        generator = TestGenerator(mock_llm, max_turns=10)
        result = await generator.generate_tests(task, mock_sandbox)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_generate_tests_rejects_string_matching(self):
        task = SweTask(
            id="test-6",
            repo="owner/repo",
            base_commit="abc123",
            language="python",
            prompt="Test",
        )


        invalid_submit = ToolCall(
            id="call_submit",
            type="function",
            function=FunctionCall(
                name="submit_tests",
                arguments='{"fail_to_pass": ["pytest"], "pass_to_pass": [], "test_files": [{"path": "test_bad.py", "content": "content = open(\\"src.py\\").read()\\nassert \\"def\\" in content"}], "install_commands": ["pip install"]}',
            ),
        )

        valid_submit = ToolCall(
            id="call_submit2",
            type="function",
            function=FunctionCall(
                name="submit_tests",
                arguments='{"fail_to_pass": ["pytest"], "pass_to_pass": [], "test_files": [], "install_commands": ["pip install"]}',
            ),
        )

        responses = [
            create_response_with_tool_calls([invalid_submit]),
            create_response_with_tool_calls([valid_submit]),
        ]

        mock_llm = MockLLMClient(responses)
        mock_sandbox = MockSandbox()

        generator = TestGenerator(mock_llm, max_turns=10)
        await generator.generate_tests(task, mock_sandbox)

        assert mock_llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_generate_tests_language_specific_commands(self):
        task = SweTask(
            id="test-7",
            repo="owner/repo",
            base_commit="abc123",
            language="javascript",
            prompt="Test",
        )

        submit_tool_call = ToolCall(
            id="call_submit",
            type="function",
            function=FunctionCall(
                name="submit_tests",
                arguments='{"fail_to_pass": ["npm test"], "pass_to_pass": [], "test_files": [], "install_commands": ["npm install"]}',
            ),
        )

        responses = [
            create_response_with_tool_calls([submit_tool_call]),
        ]

        mock_llm = MockLLMClient(responses)
        mock_sandbox = MockSandbox()

        generator = TestGenerator(mock_llm, max_turns=10)
        result = await generator.generate_tests(task, mock_sandbox)

        assert result.success is True

    def test_test_generator_default_values(self):
        mock_llm = MagicMock()
        generator = TestGenerator(mock_llm)

        assert generator._max_turns == MAX_AGENT_TURNS
        assert generator._temperature == 0.2
        assert generator._max_tokens == 2000

    def test_test_generator_custom_values(self):
        mock_llm = MagicMock()
        generator = TestGenerator(
            mock_llm,
            max_turns=50,
            model="test-model",
            temperature=0.5,
            max_tokens=1000,
        )

        assert generator._max_turns == 50
        assert generator._model == "test-model"
        assert generator._temperature == 0.5
        assert generator._max_tokens == 1000

    def test_get_tools_returns_all_schemas(self):
        mock_llm = MagicMock()
        generator = TestGenerator(mock_llm)
        tools = generator._get_tools()

        tool_names = [t.function.name for t in tools]
        assert "shell" in tool_names
        assert "submit_tests" in tool_names
        assert "write_file" in tool_names
        assert "read_file" in tool_names
        assert "list_dir" in tool_names
        assert "grep_files" in tool_names
        assert "search_files" in tool_names
        assert "apply_patch" in tool_names

    def test_truncate_string(self):
        mock_llm = MagicMock()
        generator = TestGenerator(mock_llm)

        short = "short"
        assert generator._truncate(short) == short

        long = "x" * 5000
        truncated = generator._truncate(long, 100)
        assert len(truncated) == 103
        assert truncated.endswith("...")

    def test_test_commands_for_language(self):
        """Test that _test_commands_for_language returns empty (agentic)."""
        mock_llm = MagicMock()
        generator = TestGenerator(mock_llm)

        # NO MORE HARDCODED COMMANDS - returns empty lists
        # The agent must discover via tools
        build, test = generator._test_commands_for_language("python")
        assert build == []  # Empty, agent will populate
        assert test == []  # Empty, agent will populate

        build, test = generator._test_commands_for_language("javascript")
        assert build == []
        assert test == []

        build, test = generator._test_commands_for_language("go")
        assert build == []
        assert test == []

        build, test = generator._test_commands_for_language("unknown")
        assert build == []  # No defaults


class TestConstants:
    def test_max_agent_turns(self):
        assert MAX_AGENT_TURNS == 200

    def test_max_validation_retries(self):
        assert MAX_VALIDATION_RETRIES == 3
