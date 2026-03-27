from unittest.mock import AsyncMock, MagicMock

import pytest

from swe_forge.swe.shell_tool import (
    DEFAULT_TIMEOUT_MS,
    MAX_STDERR_SIZE,
    MAX_STDOUT_SIZE,
    SecurityError,
    ShellResult,
    ShellTool,
    _is_command_blocked,
    _truncate_output,
    validate_command,
)


class TestShellResult:
    def test_default_values(self):
        result = ShellResult(
            stdout="output",
            stderr="error",
            exit_code=0,
            duration_ms=100,
        )
        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.exit_code == 0
        assert result.duration_ms == 100
        assert result.timed_out is False

    def test_timed_out_flag(self):
        result = ShellResult(
            stdout="",
            stderr="",
            exit_code=124,
            duration_ms=30000,
            timed_out=True,
        )
        assert result.timed_out is True
        assert result.exit_code == 124


class TestTruncateOutput:
    def test_no_truncation_needed(self):
        text = "short text"
        result = _truncate_output(text, 100)
        assert result == text

    def test_truncation_applied(self):
        text = "a" * 100
        result = _truncate_output(text, 50)
        assert len(result) == 53
        assert result == "a" * 50 + "..."

    def test_exact_boundary(self):
        text = "a" * 50
        result = _truncate_output(text, 50)
        assert result == text


class TestCommandBlocking:
    def test_sudo_blocked(self):
        is_blocked, pattern = _is_command_blocked("sudo rm -rf /")
        assert is_blocked is True
        assert pattern is not None

    def test_su_blocked(self):
        is_blocked, pattern = _is_command_blocked("su root")
        assert is_blocked is True

    def test_rm_rf_root_blocked(self):
        is_blocked, pattern = _is_command_blocked("rm -rf /")
        assert is_blocked is True

    def test_rm_rf_path_allowed(self):
        is_blocked, _ = _is_command_blocked("rm -rf /tmp/test")
        assert is_blocked is False

    def test_shutdown_blocked(self):
        is_blocked, _ = _is_command_blocked("shutdown -h now")
        assert is_blocked is True

    def test_reboot_blocked(self):
        is_blocked, _ = _is_command_blocked("reboot")
        assert is_blocked is True

    def test_iptables_blocked(self):
        is_blocked, _ = _is_command_blocked("iptables -F")
        assert is_blocked is True

    def test_docker_exec_blocked(self):
        is_blocked, _ = _is_command_blocked("docker exec -it container bash")
        assert is_blocked is True

    def test_normal_command_allowed(self):
        is_blocked, _ = _is_command_blocked("ls -la")
        assert is_blocked is False

    def test_git_command_allowed(self):
        is_blocked, _ = _is_command_blocked("git status")
        assert is_blocked is False

    def test_python_command_allowed(self):
        is_blocked, _ = _is_command_blocked("python -m pytest")
        assert is_blocked is False

    def test_pip_install_allowed(self):
        is_blocked, _ = _is_command_blocked("pip install requests")
        assert is_blocked is False

    def test_fork_bomb_blocked(self):
        is_blocked, _ = _is_command_blocked(":(){ :|:& };:")
        assert is_blocked is True

    def test_chmod_setuid_blocked(self):
        is_blocked, _ = _is_command_blocked("chmod +s /bin/bash")
        assert is_blocked is True


class TestValidateCommand:
    def test_returns_true_for_safe_command(self):
        assert validate_command("ls -la") is True

    def test_returns_false_for_blocked_command(self):
        assert validate_command("sudo bash") is False


class TestShellTool:
    @pytest.fixture
    def mock_sandbox(self):
        from unittest.mock import MagicMock

        sandbox = MagicMock()
        sandbox.run_command = AsyncMock()
        return sandbox

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT_MS == 30000

    def test_output_size_constants(self):
        assert MAX_STDOUT_SIZE == 3000
        assert MAX_STDERR_SIZE == 1500

    def test_init_defaults(self):
        tool = ShellTool()
        assert tool.max_stdout_size == MAX_STDOUT_SIZE
        assert tool.max_stderr_size == MAX_STDERR_SIZE
        assert tool.enable_security is True

    def test_init_custom_params(self):
        tool = ShellTool(
            max_stdout_size=5000,
            max_stderr_size=2500,
            enable_security=False,
        )
        assert tool.max_stdout_size == 5000
        assert tool.max_stderr_size == 2500
        assert tool.enable_security is False

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_sandbox):
        mock_sandbox.run_command.return_value = MagicMock(
            stdout="file1\nfile2\n",
            stderr="",
            exit_code=0,
            duration=0.5,
        )

        tool = ShellTool()
        result = await tool.execute(mock_sandbox, "ls -la")

        assert result.exit_code == 0
        assert result.stdout == "file1\nfile2\n"
        assert result.stderr == ""
        assert result.timed_out is False
        mock_sandbox.run_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_error(self, mock_sandbox):
        mock_sandbox.run_command.return_value = MagicMock(
            stdout="",
            stderr="command not found",
            exit_code=127,
            duration=0.1,
        )

        tool = ShellTool()
        result = await tool.execute(mock_sandbox, "unknown_command")

        assert result.exit_code == 127
        assert result.stderr == "command not found"

    @pytest.mark.asyncio
    async def test_execute_timeout(self, mock_sandbox):
        mock_sandbox.run_command.side_effect = TimeoutError("Command timed out")

        tool = ShellTool()
        result = await tool.execute(mock_sandbox, "sleep 100", timeout_ms=1000)

        assert result.exit_code == 124
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_execute_blocked_command_raises_security_error(self, mock_sandbox):
        tool = ShellTool(enable_security=True)

        with pytest.raises(SecurityError, match="blocked"):
            await tool.execute(mock_sandbox, "sudo rm -rf /")

        mock_sandbox.run_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_blocked_command_with_disabled_security(self, mock_sandbox):
        mock_sandbox.run_command.return_value = MagicMock(
            stdout="",
            stderr="",
            exit_code=0,
            duration=0.1,
        )

        tool = ShellTool(enable_security=False)
        result = await tool.execute(mock_sandbox, "sudo ls")

        assert result.exit_code == 0
        mock_sandbox.run_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_passes_timeout(self, mock_sandbox):
        mock_sandbox.run_command.return_value = MagicMock(
            stdout="ok",
            stderr="",
            exit_code=0,
            duration=0.1,
        )

        tool = ShellTool()
        await tool.execute(mock_sandbox, "ls", timeout_ms=5000)

        call_args = mock_sandbox.run_command.call_args
        assert call_args.kwargs["timeout"] == 5.0

    @pytest.mark.asyncio
    async def test_execute_passes_cwd_and_env(self, mock_sandbox):
        mock_sandbox.run_command.return_value = MagicMock(
            stdout="ok",
            stderr="",
            exit_code=0,
            duration=0.1,
        )

        tool = ShellTool()
        await tool.execute(
            mock_sandbox,
            "ls",
            cwd="/custom/path",
            env={"MY_VAR": "value"},
        )

        call_args = mock_sandbox.run_command.call_args
        assert call_args.kwargs["cwd"] == "/custom/path"
        assert call_args.kwargs["env"] == {"MY_VAR": "value"}

    @pytest.mark.asyncio
    async def test_execute_truncates_long_output(self, mock_sandbox):
        long_output = "a" * 5000
        mock_sandbox.run_command.return_value = MagicMock(
            stdout=long_output,
            stderr="",
            exit_code=0,
            duration=0.1,
        )

        tool = ShellTool()
        result = await tool.execute(mock_sandbox, "cat large_file")

        assert len(result.stdout) == MAX_STDOUT_SIZE + 3
        assert result.stdout == "a" * MAX_STDOUT_SIZE + "..."

    @pytest.mark.asyncio
    async def test_execute_unsafe_bypasses_security(self, mock_sandbox):
        mock_sandbox.run_command.return_value = MagicMock(
            stdout="password123",
            stderr="",
            exit_code=0,
            duration=0.1,
        )

        tool = ShellTool(enable_security=True)
        result = await tool.execute_unsafe(mock_sandbox, "sudo cat /etc/shadow")

        assert result.exit_code == 0
        assert result.stdout == "password123"
        mock_sandbox.run_command.assert_called_once()


class TestBlockedPatterns:
    @pytest.mark.parametrize(
        "command",
        [
            "sudo bash",
            "sudo -u root bash",
            "su root",
            "rm -rf /",
            "rm -rf / --no-preserve-root",
            "shutdown -h now",
            "reboot",
            "init 6",
            "systemctl stop docker",
            "iptables -F",
            "docker exec -it container bash",
            "docker run -it ubuntu bash",
            "nsenter -t 1 -m -u -i -n sh",
            "chmod +s /bin/bash",
            "chmod u+s /bin/bash",
            "userdel -r user",
            "passwd root",
        ],
    )
    def test_blocked_commands(self, command):
        is_blocked, _ = _is_command_blocked(command)
        assert is_blocked is True, f"Command should be blocked: {command}"

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "git status",
            "python -m pytest",
            "pip install requests",
            "npm install",
            "cat /etc/hosts",
            "rm -rf /tmp/test",
            "rm file.txt",
            "chmod 755 script.sh",
            "docker ps",
            "git clone https://github.com/user/repo",
            "make build",
            "pytest tests/",
        ],
    )
    def test_allowed_commands(self, command):
        is_blocked, _ = _is_command_blocked(command)
        assert is_blocked is False, f"Command should be allowed: {command}"
