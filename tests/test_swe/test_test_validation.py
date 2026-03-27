"""Tests for swe_forge.swe.test_validation module."""

import pytest
from dataclasses import dataclass

from swe_forge.swe.test_validation import (
    ValidationResult,
    TestValidator,
    detect_string_matching,
    detect_solution_leak,
    validate_script_structure,
    reject_string_matching_tests,
    validate_test_scripts,
)
from swe_forge.swe.test_generator import TestFile


class TestValidationResultDataclass:
    def test_validation_result_creation(self):
        result = ValidationResult(
            valid=True,
            reasons=[],
            passed=["pytest test.py"],
            failed=[],
        )
        assert result.valid is True
        assert result.reasons == []
        assert result.passed == ["pytest test.py"]
        assert result.failed == []

    def test_validation_result_defaults(self):
        result = ValidationResult(valid=False)
        assert result.valid is False
        assert result.reasons == []
        assert result.passed == []
        assert result.failed == []

    def test_validation_result_invalid_with_reasons(self):
        result = ValidationResult(
            valid=False,
            reasons=["Test failed", "String matching detected"],
            passed=[],
            failed=["pytest test.py"],
        )
        assert result.valid is False
        assert len(result.reasons) == 2
        assert result.failed == ["pytest test.py"]


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


@dataclass
class MockExecResult:
    stdout: str
    stderr: str
    exit_code: int


class MockSandbox:
    def __init__(self, results: list[MockExecResult] | None = None):
        self.results = results or []
        self.call_count = 0
        self.commands: list[tuple[str, float | None]] = []

    async def run_command(self, cmd: str, *, timeout: float | None = None):
        self.commands.append((cmd, timeout))
        if self.call_count < len(self.results):
            result = self.results[self.call_count]
            self.call_count += 1
            return result
        return MockExecResult(stdout="", stderr="", exit_code=0)

    async def write_file(self, path: str, content: str):
        pass


class TestDetectStringMatching:
    def test_accepts_behavioral_tests(self):
        content = (
            "import mymodule\ndef test_feature():\n    assert mymodule.func() == 1"
        )
        violations = detect_string_matching(content)
        assert violations == []

    def test_detects_open_read_pattern(self):
        content = "def test_bad():\n    content = open('src.py').read()\n    assert 'def foo' in content"
        violations = detect_string_matching(content)
        assert len(violations) > 0
        assert any("open().read()" in v for v in violations)

    def test_detects_path_read_text_pattern(self):
        content = "from pathlib import Path\ndef test_bad():\n    content = Path('src.py').read_text()\n    assert 'class' in content"
        violations = detect_string_matching(content)
        assert len(violations) > 0
        assert any("Path().read_text()" in v for v in violations)

    def test_detects_readFileSync_pattern(self):
        content = "const fs = require('fs');\nconst src = fs.readFileSync('src.js', 'utf8');\nassert(src.includes('function'));"
        violations = detect_string_matching(content)
        assert len(violations) > 0
        assert any("readFileSync()" in v for v in violations)

    def test_detects_multiple_patterns(self):
        content = """
def test_bad():
    src1 = open('a.py').read()
    assert 'x' in src1
    src2 = Path('b.py').read_text()
    assert 'y' in src2
"""
        violations = detect_string_matching(content)
        assert len(violations) >= 2


class TestDetectSolutionLeak:
    def test_accepts_normal_tests(self):
        content = "def test_feature():\n    result = mymodule.process()\n    assert result == 42"
        violations = detect_solution_leak(content)
        assert violations == []

    def test_detects_solution_comment(self):
        content = "# solution: return the hardcoded value 42\ndef test_feature(): pass"
        violations = detect_solution_leak(content)
        assert len(violations) > 0
        assert any("solution" in v.lower() for v in violations)

    def test_detects_long_expected_string(self):
        content = 'expected = "this is a very long expected output string that looks hardcoded"\nassert output == expected'
        violations = detect_solution_leak(content)
        assert len(violations) > 0


class TestValidateScriptStructure:
    def test_accepts_valid_shell_script(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="#!/bin/bash\npython -m pytest tests/",
            )
        ]
        issues = validate_script_structure(files)
        assert issues == []

    def test_rejects_missing_shebang(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="python -m pytest tests/",
            )
        ]
        issues = validate_script_structure(files)
        assert len(issues) > 0
        assert any("shebang" in i.lower() for i in issues)

    def test_rejects_empty_shell_script(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="",
            )
        ]
        issues = validate_script_structure(files)
        assert len(issues) > 0
        assert any("empty" in i.lower() for i in issues)

    def test_rejects_references_to_missing_files(self):
        files = [
            TestFile(
                path="run_tests.sh",
                content="#!/bin/bash\npython tests/test_missing.py",
            )
        ]
        issues = validate_script_structure(files)
        assert len(issues) > 0
        assert any("test_missing.py" in i for i in issues)

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
        issues = validate_script_structure(files)
        assert issues == []


class TestTestValidator:
    def test_validator_default_values(self):
        validator = TestValidator()
        assert validator._check_string_matching is True
        assert validator._check_solution_leak is True
        assert validator._check_script_structure is True
        assert validator._timeout_seconds == 120.0

    def test_validator_custom_values(self):
        validator = TestValidator(
            check_string_matching=False,
            check_solution_leak=False,
            check_script_structure=False,
            timeout_seconds=60.0,
        )
        assert validator._check_string_matching is False
        assert validator._check_solution_leak is False
        assert validator._check_script_structure is False
        assert validator._timeout_seconds == 60.0

    @pytest.mark.asyncio
    async def test_validate_tests_without_sandbox(self):
        validator = TestValidator()
        result = await validator.validate_tests(commands=["pytest test.py"])
        assert result.valid is True
        assert result.passed == []
        assert result.failed == []

    @pytest.mark.asyncio
    async def test_validate_tests_with_passing_sandbox(self):
        mock_results = [MockExecResult(stdout="OK", stderr="", exit_code=0)]
        sandbox = MockSandbox(results=mock_results)

        validator = TestValidator()
        result = await validator.validate_tests(
            commands=["pytest test.py"],
            sandbox=sandbox,
        )

        assert result.valid is True
        assert result.passed == ["pytest test.py"]
        assert result.failed == []

    @pytest.mark.asyncio
    async def test_validate_tests_with_failing_sandbox(self):
        mock_results = [MockExecResult(stdout="", stderr="AssertionError", exit_code=1)]
        sandbox = MockSandbox(results=mock_results)

        validator = TestValidator()
        result = await validator.validate_tests(
            commands=["pytest test.py"],
            sandbox=sandbox,
        )

        assert result.valid is False
        assert result.failed == ["pytest test.py"]
        assert len(result.reasons) > 0

    @pytest.mark.asyncio
    async def test_validate_tests_detects_string_matching(self):
        test_files = [
            TestFile(
                path="test_bad.py",
                content="content = open('src.py').read()\nassert 'def foo' in content",
            )
        ]

        validator = TestValidator()
        result = await validator.validate_tests(
            commands=["pytest test_bad.py"],
            test_files=test_files,
        )

        assert result.valid is False
        assert any("string-matching" in r for r in result.reasons)

    @pytest.mark.asyncio
    async def test_validate_tests_detects_script_issues(self):
        test_files = [
            TestFile(
                path="run_tests.sh",
                content="pytest tests/",
            )
        ]

        validator = TestValidator()
        result = await validator.validate_tests(
            commands=["./run_tests.sh"],
            test_files=test_files,
        )

        assert result.valid is False
        assert any("shebang" in r.lower() for r in result.reasons)


class TestValidateFailToPass:
    @pytest.mark.asyncio
    async def test_fail_to_pass_should_fail_on_base(self):
        mock_results = [MockExecResult(stdout="", stderr="FAILED", exit_code=1)]
        sandbox = MockSandbox(results=mock_results)

        validator = TestValidator()
        result = await validator.validate_fail_to_pass(
            commands=["pytest test_new_feature.py"],
            test_files=None,
            sandbox=sandbox,
        )

        assert result.valid is True
        assert "pytest test_new_feature.py" in result.failed

    @pytest.mark.asyncio
    async def test_fail_to_pass_invalid_if_passes_on_base(self):
        mock_results = [MockExecResult(stdout="OK", stderr="", exit_code=0)]
        sandbox = MockSandbox(results=mock_results)

        validator = TestValidator()
        result = await validator.validate_fail_to_pass(
            commands=["pytest test_existing.py"],
            test_files=None,
            sandbox=sandbox,
        )

        assert result.valid is False
        assert "pytest test_existing.py" in result.passed


class TestValidatePassToPass:
    @pytest.mark.asyncio
    async def test_pass_to_pass_should_pass(self):
        mock_results = [MockExecResult(stdout="OK", stderr="", exit_code=0)]
        sandbox = MockSandbox(results=mock_results)

        validator = TestValidator()
        result = await validator.validate_pass_to_pass(
            commands=["pytest test_regression.py"],
            sandbox=sandbox,
        )

        assert result.valid is True
        assert result.passed == ["pytest test_regression.py"]

    @pytest.mark.asyncio
    async def test_pass_to_pass_invalid_if_fails(self):
        mock_results = [MockExecResult(stdout="", stderr="FAILED", exit_code=1)]
        sandbox = MockSandbox(results=mock_results)

        validator = TestValidator()
        result = await validator.validate_pass_to_pass(
            commands=["pytest test_broken.py"],
            sandbox=sandbox,
        )

        assert result.valid is False
        assert "pytest test_broken.py" in result.failed


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


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_validation_workflow(self):
        test_files = [
            TestFile(
                path="tests/test_feature.py",
                content="import mymodule\ndef test_feature():\n    assert mymodule.process() == 42",
            ),
            TestFile(
                path="run_tests.sh",
                content="#!/bin/bash\npytest tests/",
            ),
        ]

        mock_results = [
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=mock_results)

        validator = TestValidator()
        result = await validator.validate_tests(
            commands=["pytest tests/test_feature.py", "./run_tests.sh"],
            test_files=test_files,
            sandbox=sandbox,
        )

        assert result.valid is True
        assert len(result.passed) == 2

    @pytest.mark.asyncio
    async def test_validation_with_disabling_checks(self):
        test_files = [
            TestFile(
                path="test_bad.sh",
                content="pytest tests/",
            )
        ]

        validator = TestValidator(check_script_structure=False)
        result = await validator.validate_tests(
            commands=["./test_bad.sh"],
            test_files=test_files,
        )

        assert result.valid is True
